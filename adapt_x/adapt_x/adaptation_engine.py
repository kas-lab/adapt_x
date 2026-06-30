#!/usr/bin/env python3
import os
import sys
import rclpy
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.srv import GetParameters
from aal_msgs.srv import AdaptArchitectureTactical, AdaptArchitecture
from adapt_x_msgs.srv import FetchRules
from adapt_x_msgs.srv import UpdateRunning
from adapt_x_msgs.srv import FetchResolutions
from adapt_x_msgs.srv import SatisfiedRule
from adapt_x_msgs.srv import UpdateHasParamValue
from adapt_x_msgs.srv import HypotheticalReqFulfillment
from adapt_x_msgs.srv import HypotheticalReqsFulfillment
from aal_msgs.msg import Adaptation
from typing import List
from rclpy.parameter import parameter_value_to_python
import jpype
import jpype.imports
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
import numpy as np
from adapt_x.adapt_x.adaptation_engine_common import client_request_init
from adapt_x.adapt_x.adaptation_engine_common import mean_fulfillment
import functools

jars_path = os.path.join(get_package_share_directory("adapt_x"), "jars", "rem")

jpype.startJVM(
    classpath=[jars_path + "/*"]
)

from eu.coresense.adaptation.tactics import AtomicRule

from java.lang import Integer

RUNNING_PARAM_NAME = "require_nodes_running"


def some_adap(adap):
    if (adap.adaptation_target == Adaptation.ROSPARAMETER):
        return "ParameterAdaptation: " + str(adap.node_name) + " " + adap.parameter_adaptation.name +"->" + str(parameter_value_to_python(adap.parameter_adaptation.value))
    if (adap.adaptation_target == Adaptation.STATETRANSITION):
        return "LifecycleAdaptation"


Adaptation.__str__ = some_adap


class AdaptationEngine(Node):

    def __init__(self,
                 require_nodes_running: bool = False,
                 verify_parameters: bool = False,
                 system_name: str = "",
                 interfaces : List[str] = [""]):
        super().__init__("adaptation_engine")
        self.declare_parameter(RUNNING_PARAM_NAME, require_nodes_running)
        self.declare_parameter("verify_parameters", verify_parameters)
        self.declare_parameter("system_name", system_name)
        self.declare_parameter("interfaces", interfaces)

        self.kb_node = "tactical_retreat_kb_node"

        self.parallel_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self.satisfied_tactic_client = self.create_client(SatisfiedRule,
                                                          f"/{self.kb_node}/satisfied_tactic",
                                                          callback_group=self.parallel_cb_group)

        self.update_has_param_cli = self.create_client(UpdateHasParamValue,
                                                       f"/{self.kb_node}/update_has_param_value",
                                                       callback_group=self.parallel_cb_group)

        self.aal_client = self.create_client(AdaptArchitecture,
                                             "/adapt_architecture",
                                             callback_group=self.parallel_cb_group)

        self.current_feature = "HI_SPEED"

        if self.get_parameter("verify_parameters").value:
            if not self.check_all_node_parameters():
                self.get_logger().error("Parameter verification failed. Aborting.")
                sys.exit(1)

        self.previous_rule = None
        self.get_logger().info("Adaptation Engine initialized and ready.")

    def on_configure(self, state: State):
        """
        Configure the node, after a configuring transition is requested.

        on_configure callback is being called when the lifecycle node
        enters the "configuring" state.

        :return: The state machine either invokes a transition to the "inactive" state or stays
        in "unconfigured" depending on the return value.
        TransitionCallbackReturn.SUCCESS transitions to "inactive".
        TransitionCallbackReturn.FAILURE transitions to "unconfigured".
        TransitionCallbackReturn.ERROR or any uncaught exceptions to "errorprocessing"
        """

        self._start_fetching_adaptations()
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        self.get_logger().info("Activating...")
        self.create_service(AdaptArchitectureTactical,
                            '/adapt_architecture_tactical', self.tactic_adaptation_callback)
        return TransitionCallbackReturn.SUCCESS

    def _start_fetching_adaptations(self):
        """Start the async database initialization chain."""
        self.get_logger().info("_start_creating_learner() called")
        fetch_resolution_cli = self.create_client(FetchResolutions,
                                                  f"/{self.kb_node}/fetch_tactic_actions",
                                                  callback_group=self.parallel_cb_group)

        future = fetch_resolution_cli.call_async(
            FetchResolutions.Request()
        )
        self.get_logger().info("Called Fetch Resolutions service")

        future.add_done_callback(self._on_adaptations_fetched)

    def _on_adaptations_fetched(self, future):
        """Callback after running update - proceed to create measures."""
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().error("Failed to fetch adaptations")
                return

            self.resolutions = {}
            for resolution in result.resolutions:
                self.resolutions[resolution.resolution_action_name] = resolution.adaptations

            self.get_logger().info("Adaptations fetched successfully")

            interfaces = self.get_parameter("interfaces").value
            # ^ this should be derived from some awareness of the BT
            # ideally it being generated from a DSL
            self.return_count = 0
            sys_name = self.get_parameter("system_name").value
            fetch_applicable_client = self.create_client(FetchRules,
                                                         f"/{self.kb_node}/fetch_applicable_tactics")
            for interface in interfaces:
                future = fetch_applicable_client.call_async(FetchRules.Request(
                                                                        resolution_model=sys_name,
                                                                        interface_name=interface
                                                    ))
                future.add_done_callback(functools.partial(self.on_rules_fetched,
                                                           interface=interface))

            num_interfaces = len(interfaces)
            while (self.return_count < num_interfaces):
                self.get_logger().info(f"Fetching rules.. ({self.return_count}/{num_interfaces})",
                                       throttle_duration_sec=1)

            self.trigger_activate()
        except Exception as e:
            self.get_logger().error(f"Error in running update callback: {e}")

    def _on_rules_fetched(self, future, interface):
        result = future.result()
        if result is None or not result.success:
            self.get_logger().error("Failed to fetch rules")
            return

        rules = result.tactic_rules
        rule_dict = {}
        for rule in rules:
            rule_dict[rule.name] = rule.reconfigurations

        self.rule_dict[interface] = rule_dict

        self.return_count += 1

    def client_request(self, client, request, identifier, timeout_sec=10.0):
        if (client.wait_for_service(timeout_sec=15.0)):
            self.get_logger().info(f"Sending {identifier} query to Typedb...")
            future = client.call_async(request)
            self.executor.spin_until_future_complete(future, timeout_sec=timeout_sec)
            if future.result() is not None:
                self.get_logger().info(f"Received response for {identifier} query.")
                result = future.result()

                if hasattr(result, 'success') and not result.success:
                    self.get_logger().error(f"Typedb {identifier} query failed.")
                    raise Exception(f"Typedb {identifier} query had success false.")
                return result
            else:
                self.get_logger().error(f"Service call failed for {identifier} query.\
                                        Potential timeout")
                raise Exception(f"Typedb query service call for {identifier} failed.")
        else:
            self.get_logger().error(f"Typedb query service not available for {identifier} query.")
            raise Exception(f"Typedb query service not available for {identifier}")

    def request_adaptation(self, adaptations: List[Adaptation]):
        request = AdaptArchitecture.Request(adaptations=adaptations)

        if self.aal_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info("")
            future = self.aal_client.call_async(request)
            future.add_done_callback(self.postprocess)
            # self.executor.spin_until_future_complete(self, future, timeout_sec=0.5)
            # if future.result() is not None:
            #     # self.get_logger().info("Received response for AAL adaptation request")
            #     result = future.result()

            #     if not result.success:
            #         self.get_logger().error("AAL adaptation request failed.")
            #         raise Exception("AAL adaptation request had success false.")
            #     return result
            # else:
            #     self.get_logger().error("Service call failed for AAL adaptation request.")
            #     raise Exception("AAL adaptation service call failed.")
        else:
            self.get_logger().error("AAL adaptation service not available")
            raise RuntimeError("AAL adaptation service not available")

    def postprocess(self, future):
        # self.get_logger().info("Postprocessing adaptation results.")
        result = future.result()

        if (result is None or not result.success):
            self.get_logger().error("Adaptation failed, cannot postprocess.")
            return

        # update_query = self.generator.generateUpdateHasParamValue(self.current_feature)

        # self.request_typedb_query(str(update_query), Query.Request.UPDATE,
        #  "Update has_param_value")

        # req_ful_res = client_request(
        #         self, self.req_ful_client,
        #         RequirementsFulfillment.Request(system_name=str(self.systemName),
        #  in_effect=True),
        #         "Update Has Param Value")

        # update_hasparam_res = self.client_request(
        #         self.update_has_param_value_client,
        #         UpdateHasParamValue.Request(feature_name=self.current_feature),
        #         "Update Has Param Value")

        # self.executor.create_task(self._update_param_value_async)
        self._schedule_param_update()

    async def _update_param_value_async(self):
        """Async method to update parameter value without blocking the executor."""
        try:
            _ = self.client_request(
                    self.update_has_param_cli,
                    UpdateHasParamValue.Request(feature_name=self.current_feature),
                    "Update Has Param Value")

        except Exception as e:
            self.get_logger().error(f"Error updating parameter value: {str(e)}")

    def tactic_adaptation_callback(self, request, response):
        try:
            interface_name = request.child_description.interface_name

            rules_for_interface = self.rule_dict.get(interface_name, [])

            if (not rules_for_interface):
                self.get_logger().info(f"No applicable tactics found for interface \
                                        {interface_name}. No adaptation will be applied.")
                response.success = True
                return response

            sat_res = self.client_request(
                self.satisfied_tactic_client,
                SatisfiedRule.Request(tactic_rules=list(rules_for_interface.keys())),
                "Satisfied Tactic", 1.1)

            satisfied = sat_res.satisfied
            if (not any(satisfied)):
                self.get_logger().info(f"No applicable tactics found for interface \
                                        {interface_name} that satisfy their guard.")
                response.success = True
                return response

            #  satisfied is a list of bools, rules_for interface, we want the first rule satisfied

            selected_rule = None
            for i, is_satisfied in enumerate(satisfied):
                if is_satisfied:
                    selected_rule = rules_for_interface[i]
                    break

            feature_name = self.resulting_feature[selected_rule]
            selected_actions = self.rule_to_actions[selected_rule]

            self.current_feature = feature_name

            self.request_adaptation(selected_actions)

            response.success = True
            return response

        except RuntimeError as e:
            self.get_logger().error(f"RuntimeError during tactic adaptation: {str(e)}")
            response.success = False
            return response

    def _schedule_param_update(self):
        """Schedule parameter update as a non-blocking operation."""
        try:
            # Create the request but don't wait for it synchronously
            update_request = UpdateHasParamValue.Request(feature_name=self.current_feature)
            # Call async and add a callback to handle the result
            future = self.update_has_param_cli.call_async(update_request)
            future.add_done_callback(self._on_param_update_complete)

            self.get_logger().info(f"Scheduled parameter update for: {self.current_feature}")
        except RuntimeError as e:
            self.get_logger().error(f"Error scheduling parameter update: {str(e)}")

    def _on_param_update_complete(self, future):
        """Callback when parameter update completes."""
        try:
            result = future.result()
            if result and hasattr(result, 'success') and result.success:
                self.get_logger().info("Parameter update completed successfully")
            else:
                self.get_logger().error("Parameter update failed")
        except RuntimeError as e:
            self.get_logger().error(f"Error in parameter update callback: {str(e)}")


def main():
    rclpy.init()

    action_client = AdaptationEngine()
    mt_executor = MultiThreadedExecutor()
    mt_executor.add_node(action_client)
    action_client.executor = mt_executor
    mt_executor.spin()

    action_client.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
