# Copyright 2026 KAS-lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://apache.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn

from aal_msgs.srv import AdaptArchitecture
from aal_msgs.srv import AdaptArchitectureTactical
from adapt_x_msgs.srv import FetchApplicable
from adapt_x_msgs.srv import FetchResolutionActions
from adapt_x_msgs.srv import SatisfiedTactic
from adapt_x_msgs.srv import UpdateInterfaces


class TacticalAdaptationNode(Node):
    """Minimal tactical adaptation node backed by an already-loaded KB."""

    def __init__(self):
        super().__init__("tactical_adaptation_node")
        self.declare_parameter("interfaces", ["navigate_to_pose"])
        self.declare_parameter("resolution_model", "")
        self.declare_parameter("kb_node", "tactical_retreat_kb_node")

        self.interfaces = set()
        self.resolution_model = ""
        self.kb_node = ""
        self.rule_dict = {}
        self.rule_to_actions = {}

        self.executor = None
        self.fetch_applicable_client = None
        self.fetch_actions_client = None
        self.satisfied_tactic_client = None
        self.aal_client = None

        self.create_service(
            AdaptArchitectureTactical,
            "/adapt_architecture_tactical",
            self.tactic_adaptation_callback,
        )
        self.create_service(
            UpdateInterfaces,
            "/update_interfaces",
            self.update_interfaces_callback,
        )

    def on_configure(self, state: State):
        self.interfaces = self._normalize_interfaces(
            self.get_parameter("interfaces").value
        )
        self.resolution_model = str(self.get_parameter("resolution_model").value)
        self.kb_node = str(self.get_parameter("kb_node").value)

        self.fetch_applicable_client = self.create_client(
            FetchApplicable,
            f"/{self.kb_node}/fetch_applicable_tactics",
        )
        self.fetch_actions_client = self.create_client(
            FetchResolutionActions,
            f"/{self.kb_node}/fetch_resolution_actions",
        )
        self.satisfied_tactic_client = self.create_client(
            SatisfiedTactic,
            f"/{self.kb_node}/satisfied_tactic",
        )
        self.aal_client = self.create_client(
            AdaptArchitecture,
            "/adapt_architecture",
        )

        try:
            self._refresh_interface_cache(self.interfaces)
        except RuntimeError as exc:
            self.get_logger().error(f"Failed to initialize interface cache: {exc}")
            return TransitionCallbackReturn.FAILURE

        return TransitionCallbackReturn.SUCCESS

    def tactic_adaptation_callback(self, request, response):
        try:
            interface_name = request.child_description.interface_name
            rules_for_interface = self.rule_dict.get(interface_name, [])

            if not rules_for_interface:
                self.get_logger().info(
                    f"No tactics cached for interface {interface_name}."
                )
                response.success = True
                return response

            satisfied_response = self._call_service(
                self.satisfied_tactic_client,
                SatisfiedTactic.Request(tactics=rules_for_interface),
                "Satisfied Tactic",
            )
            selected_rule = self._first_satisfied_rule(
                rules_for_interface,
                satisfied_response.satisfied,
            )

            if selected_rule is None:
                self.get_logger().info(
                    f"No satisfied tactics for interface {interface_name}."
                )
                response.success = True
                return response

            adaptations = self.rule_to_actions.get(selected_rule)
            if adaptations is None:
                raise RuntimeError(
                    f"No adaptations cached for selected tactic {selected_rule}"
                )

            adaptation_response = self._call_service(
                self.aal_client,
                AdaptArchitecture.Request(adaptations=adaptations),
                "Adapt Architecture",
            )
            response.success = bool(adaptation_response.success)
            return response
        except RuntimeError as exc:
            self.get_logger().error(f"Tactical adaptation failed: {exc}")
            response.success = False
            return response

    def update_interfaces_callback(self, request, response):
        current_interfaces = set(self.interfaces)
        target_interfaces = current_interfaces | self._normalize_interfaces(
            request.add_interfaces
        )
        target_interfaces -= self._normalize_interfaces(request.remove_interfaces)

        try:
            self._refresh_interface_cache(target_interfaces)
        except RuntimeError as exc:
            response.interfaces = sorted(current_interfaces)
            response.success = False
            response.message = f"Interface update rejected: {exc}"
            return response

        response.interfaces = sorted(self.interfaces)
        response.success = True
        response.message = "Interfaces updated"
        return response

    def _refresh_interface_cache(self, interfaces):
        new_rule_dict, new_rule_to_actions = self._build_interface_cache(interfaces)
        self.interfaces = set(interfaces)
        self.rule_dict = new_rule_dict
        self.rule_to_actions = new_rule_to_actions

    def _build_interface_cache(self, interfaces):
        rule_dict = {}
        rule_to_actions = {}

        for interface_name in sorted(interfaces):
            applicable_response = self._call_service(
                self.fetch_applicable_client,
                FetchApplicable.Request(
                    resolution_model=self.resolution_model,
                    interface_name=interface_name,
                ),
                "Fetch Applicable Tactics",
            )
            self._raise_if_unsuccessful(applicable_response, interface_name)
            tactics = list(applicable_response.tactics)
            rule_dict[interface_name] = tactics

            for tactic_name in tactics:
                if tactic_name in rule_to_actions:
                    continue

                actions_response = self._call_service(
                    self.fetch_actions_client,
                    FetchResolutionActions.Request(tactic_name=tactic_name),
                    "Fetch Resolution Actions",
                )
                self._raise_if_unsuccessful(actions_response, tactic_name)
                rule_to_actions[tactic_name] = list(actions_response.adaptations)

        return rule_dict, rule_to_actions

    def _call_service(self, client, request, identifier, timeout_sec=10.0):
        if client is None:
            raise RuntimeError(f"{identifier} client is not configured")

        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"{identifier} service is not available")

        future = client.call_async(request)
        if self.executor is not None:
            self.executor.spin_until_future_complete(future, timeout_sec=timeout_sec)
        else:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        result = future.result()
        if result is None:
            raise RuntimeError(f"{identifier} service call returned no result")
        return result

    @staticmethod
    def _normalize_interfaces(interfaces):
        return {interface for interface in interfaces if interface}

    @staticmethod
    def _first_satisfied_rule(rules, satisfied):
        for rule, is_satisfied in zip(rules, satisfied):
            if is_satisfied:
                return rule
        return None

    @staticmethod
    def _raise_if_unsuccessful(response, identifier):
        if hasattr(response, "success") and not response.success:
            raise RuntimeError(f"{identifier} returned success=false")


def main():
    rclpy.init()

    node = TacticalAdaptationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    node.executor = executor

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
