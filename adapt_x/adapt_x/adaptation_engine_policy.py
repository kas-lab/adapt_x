#!/usr/bin/env python3
import os
import sys
import rclpy
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.srv import GetParameters
from ros_typedb_msgs.srv import Query
from std_srvs.srv import Trigger
from adapt_x_msgs.msg import RelaxedRequirements
from adapt_x_msgs.msg import SymbolicState
from aal_msgs.srv import AdaptArchitecture
from aal_msgs.action import AdaptArchitectureConcurrent
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action import ActionServer
from adapt_x_msgs.srv import FetchApplicable
from adapt_x_msgs.srv import UpdateRunning
from adapt_x_msgs.srv import FetchAdaptations
from adapt_x_msgs.srv import SatisfiedTactic
from adapt_x_msgs.srv import UpdateHasParamValue
from adapt_x_msgs.srv import HypotheticalReqFulfillment
from adapt_x_msgs.srv import HypotheticalReqsFulfillment
from aal_msgs.msg import Adaptation
from typing import List
from datetime import datetime
from rclpy.parameter import parameter_value_to_python
from rclpy.time import Time
import time
import jpype
import jpype.imports
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
import numpy as np
import pandas as pd
from adapt_x.adaptation_engine_common import client_request_init
from adapt_x.adaptation_engine_common import mean_fulfillment
from statistics import mean, StatisticsError


jars_path = os.path.join(get_package_share_directory("adapt_x"), "jars", "rem")

jpype.startJVM(
    classpath=[jars_path + "/*"]
)

from org.ros2.rcljava.rebet_java import RosToolingSupport
from de.fraunhofer.ipa.ros import RosStandaloneSetupGenerated
from de.fraunhofer.ipa.rossystem import RosSystemStandaloneSetupGenerated
from de.fraunhofer.ipa.ros2 import Ros2StandaloneSetupGenerated
from eu.coresense.variability import FeatureModelStandaloneSetupGenerated
from eu.coresense.resolution import ResolutionModelStandaloneSetupGenerated
from eu.coresense.context import ContextModelStandaloneSetupGenerated
from eu.coresense.requirements import RequirementsModelStandaloneSetupGenerated
from eu.coresense.adaptation import TacticsStandaloneSetupGenerated

from org.eclipse.xtext.resource import XtextResourceSet
from org.eclipse.xtext.xbase.lib import Pair
from eu.coresense.common import NameProvider
from eu.coresense.context.generator import ContextNameProvider
from eu.coresense.resolution.generator import ResolutionNameProvider
from eu.coresense.adaptation.generator import TacticsNameProvider
from eu.coresense.common import RosNameProvider
from eu.coresense.adaptation.generator import TQLGenerator
from eu.coresense.adaptation.tactics import TacticsModel
from eu.coresense.requirements.generator import RequirementsModelGenerator
from eu.coresense.requirements.requirementsModel import RequirementsModel
from eu.coresense.adaptation.tactics import AtomicRule
from uma.caosd.rhea.AdvancedConstraintsFMmetamodel import AdvancedConstraintsFMmetamodelPackage

from java.lang import String, Integer
from java.util import HashSet

RUNNING_PARAM_NAME = "require_nodes_running"


def some_adap(adap):
    if (adap.adaptation_target == Adaptation.ROSPARAMETER):
        return "ParameterAdaptation: " + str(adap.node_name) + " " + adap.parameter_adaptation.name +"->" + str(parameter_value_to_python(adap.parameter_adaptation.value))
    if (adap.adaptation_target == Adaptation.STATETRANSITION):
        return "LifecycleAdaptation"


Adaptation.__str__ = some_adap


class AdaptationEngine(Node):

    def __init__(self, resolution_path: str = "",
                 variability_path: str = "",
                 context_path: str = "",
                 requirements_path: str = "",
                 ros_path: List[str] = [""],
                 ros2_path: List[str] = [""],
                 rossystem_path: List[str] = [""],
                 tactics_path: str = "",
                 require_nodes_running: bool = False,
                 verify_parameters: bool = False,
                 policy_mode: bool = True,
                 policy_learner: str = "egreedy"):
        super().__init__("adaptation_engine")
        self.declare_parameter("tactics_path", tactics_path)
        self.declare_parameter("resolution_path", resolution_path)
        self.declare_parameter("variability_path", variability_path)
        self.declare_parameter("context_path", context_path)
        self.declare_parameter("requirements_path", requirements_path)
        self.declare_parameter("ros_path", ros_path)
        self.declare_parameter("ros2_path", ros2_path)
        self.declare_parameter("rossystem_path", rossystem_path)
        self.declare_parameter(RUNNING_PARAM_NAME, require_nodes_running)
        self.declare_parameter("verify_parameters", verify_parameters)
        self.declare_parameter("policy_learner", policy_learner)
        self.reward = None
        self.reward_window = []
        self.resSet = XtextResourceSet()
        self.previous_requirements = None
        self.epsilon = 0.0  # TODO: parameterize
        RosToolingSupport.registerEPackages()
        AdvancedConstraintsFMmetamodelPackage.eINSTANCE.eClass()
        self.kb_node = "tactical_retreat_kb_node"
        self.exclusive_cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()

        self.parallel_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self.satisfied_tactic_client = self.create_client(SatisfiedTactic,
                                                          f"/{self.kb_node}/satisfied_tactic",
                                                          callback_group=self.parallel_cb_group)
        self.dream_client_two = self.create_client(HypotheticalReqsFulfillment,
                                                   f"/{self.kb_node}/fetch_hyp_requirements",
                                                   callback_group=self.parallel_cb_group)

        self.update_has_param_cli = self.create_client(UpdateHasParamValue,
                                                       f"/{self.kb_node}/update_has_param_value",
                                                       callback_group=self.parallel_cb_group)
        self.dream_client = self.create_client(HypotheticalReqFulfillment,
                                               f"/{self.kb_node}/fetch_hyp_requirement")

        self.aal_client = self.create_client(AdaptArchitecture,
                                             "/adapt_architecture",
                                             callback_group=self.parallel_cb_group)

        self._action_server = ActionServer(
            self,
            AdaptArchitectureConcurrent,
            '/adapt_concurrent',
            execute_callback=self.execute_callback,
            callback_group=self.parallel_cb_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        self.create_subscription(SymbolicState, "symbolic_state",
                                 self.state_callback,
                                 10, callback_group=self.parallel_cb_group)
        self.create_subscription(RelaxedRequirements, "/requirements",
                                 self.requirements_callback,
                                 10, callback_group=self.parallel_cb_group)

        self.current_feature = "HI_SPEED"
        self.need_to_register_reward = False
        self.current_state = None
        self.load_all_models()

        if not self.check_all_nodes_running():
            if self.get_parameter(RUNNING_PARAM_NAME).value:
                self.get_logger().error("Not all ROS nodes are running. Aborting.")
                return
            self.get_logger().warn(
                "Not all ROS nodes are running. \
                Continuing anyway due to require_nodes_running=False"
            )

        if self.get_parameter("verify_parameters").value:
            if not self.check_all_node_parameters():
                self.get_logger().error("Parameter verification failed. Aborting.")
                sys.exit(1)

        self.create_generators()

        self.get_logger().info("Adaptation Engine initialized.")

    def print_q_table(self):
        q_df = pd.DataFrame({rule: self.q_table[:, i] for i, rule in enumerate(self.all_rules)},
                            index=[str(state.TQLName) for state in self.generator.allStates])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            self.get_logger().info(f"\n{str(q_df.transpose())}")

    def initial_queries(self, initial_insert_query):
        typedb_client = self.create_client(Query, f"/{self.kb_node}/query",
                                           callback_group=self.parallel_cb_group)

        self.wait_for_typedb(typedb_client)

        update_running_client = self.create_client(UpdateRunning,
                                                   f"/{self.kb_node}/update_running",
                                                   callback_group=self.parallel_cb_group)
        try:
            initial_insert_res = client_request_init(self, typedb_client,
                                                     Query.Request(
                                                         query=initial_insert_query,
                                                         query_type=Query.Request.INSERT),
                                                     "Initial Model Load", 120)

            if initial_insert_res is None or not initial_insert_res.success:
                self.get_logger().error("Failed to insert initial model. Aborting")
                sys.exit(1)

            start_time_res = client_request_init(
                self, update_running_client,
                UpdateRunning.Request(node_names=[str(node.getFrom().getName())
                                                  for node in self.rosNodesInRossystems]),
                "Update Running")
            if start_time_res is None or not start_time_res.success:
                self.get_logger().error("Failed to insert node start times. Aborting")
                sys.exit(1)

            # self.request_measures() TODO: return
        except RuntimeError as e:
            self.get_logger().error(f"RuntimeError during initial model load: {str(e)}")
            sys.exit(1)

        interfaces = ["navigate_to_pose", "detect_object_srv"]
        # ^ this should be derived from some awarenss of the BT
        # ideally it being generated from a DSL
        self.get_logger().info("Fetch applicable now?")

        fetch_applicable_client = self.create_client(FetchApplicable,
                                                     f"/{self.kb_node}/fetch_applicable_tactics",
                                                     callback_group=self.parallel_cb_group)
        fetch_response = client_request_init(self,
                                             fetch_applicable_client,
                                             FetchApplicable.Request(
                                                                    resolution_model=self.sys_name,
                                                                    interface_name=interfaces[0]
                                             ),
                                             "Fetch Applicable Tactics")
        self.rule_dict = {}
        self.rule_dict[interfaces[0]] = fetch_response.tactics
        self.rule_to_actions = {}
        self.get_logger().info(f"?: {str(fetch_response.tactics)}")
        fetch_adaptations_client = self.create_client(FetchAdaptations,
                                                      f"/{self.kb_node}/fetch_tactic_actions",
                                                      callback_group=self.parallel_cb_group)
        for tactic in fetch_response.tactics:
            actions_res = client_request_init(
                    self, fetch_adaptations_client,
                    FetchAdaptations.Request(tactic_name=tactic),
                    "Fetch Adaptations")
            self.rule_to_actions[tactic] = actions_res.adaptations
            self.get_logger().info(f"adaptations fetched during initial model load for \
                                   tactic_name {tactic}: \n len {len(actions_res.adaptations)} : \
                                    { str([str(adap) for adap in actions_res.adaptations])}")

        self.get_logger().info("Initial Queries complete")

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on activate")

        self.get_logger().info("activate success")
        return TransitionCallbackReturn.SUCCESS

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
        try:
            resolution_model = self.generator.getResolutionModel(self.tacticsModel,
                                                                 self.tacticsModelResource).get(0)
            self.systemName = resolution_model.getName()
            self.sys_name = str(resolution_model.getName())
            initial_insert_query = ""
            initial_insert_query += str(self.generate_architecture_query())
            initial_insert_query += str(self.generator.generateTQL(self.tacticsModel,
                                                                   self.tacticsModelResource,
                                                                   True))
            initial_insert_query += str(self.req_generator.generateInsertQuery(
                self.requirementsModel,
                self.sys_name, False))

            self.initial_insert_query = initial_insert_query
            # self.initial_queries(initial_insert_query)

            self.resulting_feature = {}
            self.rules_with_effect = []
            for rule in self.tacticsModel.getAdaptationRules():
                atom_rule = (AtomicRule) @ rule
                rule_body = atom_rule.getRuleBody()
                rule_name = str(atom_rule.getName())
                if rule_body.getEffect() is not None:
                    self.rules_with_effect.append(str(atom_rule.getName()))
                self.resulting_feature[rule_name] = str(rule_body.getAtomicAction().getActions()
                                                        .get((Integer) @ 0).getVariant()
                                                        .getFeature().getName())

            self.previous_rule = None

            self.previous_state = ""
            all_states = self.generator.allStates
            escapes = self.generator.escapes
            self.selected_rule = None

            self.reward_registry = {}
            self.adaptation_policy = {}
            self.get_logger().info(f"Num states: {str(all_states.size())}")

            for state_java in all_states:
                state_name = str(state_java.TQLName)
                self.get_logger().info(f"State found: {state_name}")
                self.adaptation_policy[state_name] = [str(rule_name_java)
                                                      for rule_name_java in
                                                      state_java.associatedRulesTQLNames]

                if (len(self.adaptation_policy[state_name]) > 0):
                    # escapes = str(escapes.get(Pair(state_name, 'RULE_WITH_EFFECT')))
                    # self.get_logger().info(f"State {state_name} has the escape \
                    #                         {escapes} for the first rule")
                    pass
                for action in self.resulting_feature:
                    self.reward_registry[(state_name, action)] = []

            self.get_logger().info(f"Keyset size: {escapes.keySet().size()}")
            self.get_logger().info(f"Escapes itself: {str(escapes)}")
            for x in escapes.keySet():
                self.get_logger().info(f"Key: {x}")
                self.get_logger().info(f"Value: {escapes.get(x)}")

            # self.get_logger().info(f"Reward registry \n{self.reward_registry}")
            # self.get_logger().info(f"Adaptation Policy \n{self.adaptation_policy}")

            if (self.get_parameter("policy_learner").value == "q-learning"):
                self.get_logger().info("Q learning selected")
                self.all_rules = [str(ruleName) for ruleName in self.generator.ruleNames]
                self.q_table = np.full((int(all_states.size()), len(self.all_rules)), -np.inf)

                self.learning_rate = 0.8
                self.discount_factor = 0.95
                self.state_to_index = {}
                [self.state_to_index.update({str(state_name.TQLName): i})
                 for i, state_name in enumerate(all_states)]

                for state in self.adaptation_policy.keys():
                    s_i = self.state_to_index[state]
                    for rule in self.adaptation_policy[state]:
                        a_i = self.all_rules.index(rule)
                        self.q_table[s_i, a_i] = 0

            self.print_q_table()

            self._start_database_init()

            self.get_logger().info("Configure Success")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Configure error {e}")

            return TransitionCallbackReturn.ERROR

    def _start_database_init(self):
        """Start the async database initialization chain."""
        typedb_client = self.create_client(Query, f"/{self.kb_node}/query",
                                           callback_group=self.parallel_cb_group)
        self.wait_for_typedb(typedb_client)

        future = typedb_client.call_async(
            Query.Request(query=self.initial_insert_query, query_type=Query.Request.INSERT)
        )
        future.add_done_callback(self._on_model_inserted)

    def _on_model_inserted(self, future):
        """Callback after model insert - proceed to next step."""
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().error("Failed to insert initial model")
                return

            self.get_logger().info("Initial model inserted successfully")

            # Next step: update running
            update_running_client = self.create_client(UpdateRunning,
                                                       f"/{self.kb_node}/update_running",
                                                       callback_group=self.parallel_cb_group)
            future = update_running_client.call_async(
                UpdateRunning.Request(node_names=[str(node.getFrom().getName())
                                                  for node in self.rosNodesInRossystems])
            )
            future.add_done_callback(self._on_running_updated)
        except Exception as e:
            self.get_logger().error(f"Error in model insert callback: {e}")

    def _on_running_updated(self, future):
        """Callback after running update - proceed to fetch tactics."""
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().error("Failed to update running nodes")
                return

            self.get_logger().info("Running nodes updated successfully")
            self._fetch_applicable_tactics_async()
        except Exception as e:
            self.get_logger().error(f"Error in running update callback: {e}")

    def _fetch_applicable_tactics_async(self):
        """Fetch applicable tactics asynchronously."""
        fetch_applicable_client = self.create_client(FetchApplicable,
                                                     f"/{self.kb_node}/fetch_applicable_tactics",
                                                     callback_group=self.parallel_cb_group)
        future = fetch_applicable_client.call_async(
            FetchApplicable.Request(resolution_model=self.sys_name,
                                    interface_name="navigate_to_pose")
        )
        future.add_done_callback(self._on_tactics_fetched)

    def _on_tactics_fetched(self, future):
        """Callback after tactics fetched."""
        try:
            result = future.result()
            self.rule_dict = {"navigate_to_pose": result.tactics}
            self.rule_to_actions = {}
            self.get_logger().info(f"Fetched tactics: {result.tactics}")

            # Recursively fetch adaptations for each tactic
            self._fetch_adaptations_for_tactics(list(result.tactics), 0)
        except Exception as e:
            self.get_logger().error(f"Error fetching tactics: {e}")

    def _fetch_adaptations_for_tactics(self, tactics_list, index):
        """Recursively fetch adaptations for each tactic."""
        if index >= len(tactics_list):
            self.get_logger().info("Database initialization complete")
            return

        tactic = tactics_list[index]
        fetch_adaptations_client = self.create_client(FetchAdaptations,
                                                      f"/{self.kb_node}/fetch_tactic_actions",
                                                      callback_group=self.parallel_cb_group)
        future = fetch_adaptations_client.call_async(
            FetchAdaptations.Request(tactic_name=tactic)
        )

        # Capture variables for closure
        def callback(f, tactic_name=tactic, tactics=tactics_list, idx=index):
            try:
                result = f.result()
                self.rule_to_actions[tactic_name] = result.adaptations
                self.get_logger().info(f"Fetched {len(result.adaptations)} adaptations for {tactic_name}")
                self._fetch_adaptations_for_tactics(tactics, idx + 1)
            except Exception as e:
                self.get_logger().error(f"Error fetching adaptations for {tactic_name}: {e}")

        future.add_done_callback(callback)

    def goal_callback(self, goal_request):
        # When accepting a goal, we execute an initial adaptation akin to the non-policy version.
        self.get_logger().info("Goal callback")

        if (self.current_state is None):
            self.get_logger().info("Initial symbolic state not provided yet")
            return GoalResponse.REJECT

        self.goal_requirements = goal_request.requirements
        self.goal_child_description = goal_request.child_description
        self.feedback = AdaptArchitectureConcurrent.Feedback()
        self.interface_name = self.goal_child_description.interface_name
        self.get_logger().info("Received goal")
        try:
            state = self.current_state  # freeze it for this callback
            self.get_logger().info("01")
            new_rule = self.egreedy(state, True)
            self.get_logger().info("02")
            if (new_rule is None):
                return GoalResponse.ACCEPT
            
            self.selected_rule = self.all_rules[new_rule]
            selected_actions = self.rule_to_actions[self.selected_rule]
            self.feedback.adaptations_executed.extend(selected_actions)
            self.get_logger().info("1")
            self.request_adaptation(selected_actions)
            self.get_logger().info("2")

            return GoalResponse.ACCEPT

        except RuntimeError as e:
            self.get_logger().error(f"RuntimeError during tactic adaptation: {str(e)}")
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute a goal."""
        self.get_logger().info('Executing goal...')

        # After the initial adaptation in the goal_callback
        # We move into reacting to any states changes
        # The reaction consists of registering a reward for the previous action
        # and choosing a new action.

        # Start executing the action

        goal_handle.publish_feedback(self.feedback)  # initial adaptations

        while not goal_handle.is_cancel_requested:
            if (self.state_changed):
                try:
                    self.get_logger().info("State changed, choosing new adaptation")
                    self.state_changed = False
                    new_state = self.current_state  # freeze it for this callback
                    prev_state = self.previous_state
                    prev_rule = self.selected_rule
                    reward_window = self.reward_window.copy()
                    self.reward_window.clear()

                    # this can happen if there's no actions associated for the state
                    if (prev_rule is not None):
                        reward = self.mean_rewards_between_states(reward_window,
                                                                  prev_state, new_state)
                        self.update_q_table(prev_state.state_name,
                                            prev_rule, new_state.state_name, reward)

                    new_rule = self.egreedy(new_state, True)

                    if (new_rule is not None):
                        self.selected_rule = self.all_rules[new_rule]
                        selected_actions = self.rule_to_actions[self.selected_rule]
                        self.feedback.adaptations_executed.extend(selected_actions)
                        self.request_adaptation(selected_actions)

                    goal_handle.publish_feedback(self.feedback)

                except RuntimeError as e:
                    self.get_logger().error(f"RuntimeError during tactic adaptation: {str(e)}")
                    sys.exit(1)

        goal_handle.canceled()
        self.get_logger().info('Goal canceled')
        return AdaptArchitectureConcurrent.Result()

        # goal_handle.succeed()

        # # Populate result message
        # result = AdaptArchitectureConcurrent.Result()

        # self.get_logger().info('Returning result')

        # return result

    def client_request(self, client, request, identifier, timeout_sec=10):
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

        # self.get_logger().info(f"Reward registry \n{self.reward_registry}")

    async def _update_param_value_async(self):
        """Async method to update parameter value without blocking the executor."""
        try:
            _ = self.client_request(
                    self.update_has_param_cli,
                    UpdateHasParamValue.Request(feature_name=self.current_feature),
                    "Update Has Param Value")

            # self.get_logger().info(f"Reward registry \n{self.reward_registry}")
        except Exception as e:
            self.get_logger().error(f"Error updating parameter value: {str(e)}")

    def state_callback(self, state_msg):
        self.get_logger().info("State callback")

        try:
            self.previous_state = self.current_state
            self.current_state = state_msg
            self.state_changed = True
        except RuntimeError as e:
            self.get_logger().error(f"RuntimeError during context update: {str(e)}")

    def requirements_callback(self, reqs_msg: RelaxedRequirements):
        self.get_logger().info("Requirements callback")

        self.most_recent_fulfillment = reqs_msg.header.stamp
        mean_ff = mean_fulfillment(reqs_msg, self.systemName)
        self.reward = mean_ff
        self.reward_window.append((Time.from_msg(reqs_msg.header.stamp), mean_ff))
        self.get_logger().info(f"Registered requirement {mean_ff}")

    def egreedy(self, symbolic_state: str, q_learning=False) -> int:
        # e-greedy
        state = symbolic_state.state_name
        choice = np.random.random()
        self.get_logger().info(f"Some choice made {state} {str(self.adaptation_policy)}")
        current_actions = self.adaptation_policy.get(state, None)
        self.get_logger().info(f"current actions {str(current_actions)}")

        if current_actions is None:
            raise RuntimeError("state " + state + " did not retrieve actions from the policy")
        if current_actions == []:
            return None  # TODO: for now we don't adapt, but maybe we should dream instead maybe?
        if choice < self.epsilon:
            # explore
            self.get_logger().info("explore")
            next_action = str(np.random.choice(current_actions))
        else:
            self.get_logger().info("exploit")

            if (q_learning):
                self.get_logger().info("qq")
                self.print_q_table()
                self.get_logger().info(f"exploit2.1 {str(state)}")
                self.get_logger().info(f"exploit2.2 {str(self.state_to_index)}")
                self.get_logger().info(f"exploit3 {self.q_table[self.state_to_index[state]]}")
                action_index = np.argmax(self.q_table[self.state_to_index[state]])
                if(not (self.all_rules[action_index] in current_actions)):
                    raise RuntimeError(" did not retrieve actions from the policy")
                return np.argmax(self.q_table[self.state_to_index[state]])
            else:
                # just max
                known_mean_reward = [mean(self.reward_registry.get((state, possible_action), [0]))
                                     for possible_action in current_actions]
                next_action = current_actions[known_mean_reward.index(max(known_mean_reward))]

        return self.all_rules.index(next_action)

    def qlearning(self, state):
        return np.argmax(self.q_table[self.state_to_index[state]])

    def mean_rewards_between_states(self, rewards_list: tuple[Time, float],
                                    first_state: SymbolicState,
                                    last_state: SymbolicState) -> float:
        earliest = Time.from_msg(first_state.header.stamp)
        latest = Time.from_msg(last_state.header.stamp)
        self.get_logger().info(f"Earliest {first_state.state_name} : {str(earliest)}")
        self.get_logger().info(f"Latest {last_state.state_name} : {str(latest)}")

        for rqs in rewards_list:
            self.get_logger().info(f"reward time {rqs[1]} : {str(rqs[0])}")

        to_be_averaged = [rqs[1] for rqs in rewards_list if rqs[0] > earliest and rqs[0] < latest]

        try:
            return mean(to_be_averaged)
        except StatisticsError:
            self.get_logger().warn("reward list was empty")
            return 0

    def update_q_table(self, rewarded_state: str, rule, new_state: str, reward: float):
        s_index = self.state_to_index[rewarded_state]
        s_prime_index = self.state_to_index[new_state]
        action_index = self.all_rules.index(rule)
        Q_current = self.q_table[s_index, action_index]  # Q(S,A)
        Q_next = self.q_table[s_prime_index]  # Q(S')
        discounted_optimal_next_Q = self.discount_factor * \
            np.max(Q_next)  # γ * max_a Q(S',a)

        # Q(S,A) <- Q(S,A) + 𝛂[R + γmax_a Q(S',a) - Q(s,a)]
        self.q_table[s_index, action_index] = Q_current + self.learning_rate * \
            (reward + discounted_optimal_next_Q - Q_current)

    def wait_for_typedb(self, client):
        while not client.wait_for_service(timeout_sec=30.0):
            self.get_logger().info("Waiting for TypeDB to be ready")

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

    def check_all_node_parameters(self):
        checks = []
        for ros_node in self.rosNodesInRossystems:
            parameterNames = [ros_node.getFrom().getParameter().get((Integer) @ i).getName()
                              for i in range(ros_node.getFrom().getParameter().size())]
            nodeName = ros_node.getName()
            if len(parameterNames) == 0:
                self.get_logger().warn(f"Node {nodeName} has no parameters defined.")
                checks.append(True)
                continue

            parameter_client = self.create_client(GetParameters,
                                                  "/" + str(nodeName) + "/get_parameters")

            while not parameter_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info(f"get_parameters service for {nodeName} not available, \
                                       waiting again...")

            request = GetParameters.Request()
            request.names = parameterNames
            response = parameter_client.call(request)
            returned_parameters = response.values
            if len(returned_parameters) == 0:
                self.get_logger().warn(f"Node {nodeName} returned no parameters, \
                                       one of the requested parameters may not exist.")
                checks.append(False)
                continue

            # TODO: Check the values as well.

        return all(checks)

    def check_all_nodes_running(self):
        self.rosNodesInRossystems = RosToolingSupport.allRossystemNodes(self.rosSystemResources)
        discovered_nodes = set(self.get_node_names())
        known_nodes = set()
        for ros_node in self.rosNodesInRossystems:
            node_name = ros_node.getName()
            if (node_name.charAt(0) == '/'):
                node_name = node_name.substring(1)
            known_nodes.add(node_name)

        if known_nodes.issubset(discovered_nodes):
            self.get_logger().info("All ROS nodes are running.")
            return True
        else:
            missing_nodes = known_nodes - discovered_nodes
            missing_nodes = [str(node) for node in missing_nodes]
            self.get_logger().warn(
                f"The following ROS nodes are not running: {', '.join(missing_nodes)}"
            )
            return False

    def generate_architecture_query(self):
        self.get_logger().info("Generating architecture TQL query...")
        architectureTQL = String("")
        res_models = self.generator.getResolutionModel(self.tacticsModel,
                                                       self.tacticsModelResource)

        context_models = self.generator.getContextModel(self.tacticsModel,
                                                        self.tacticsModelResource)

        rossystems = HashSet()
        for resModel in res_models:
            rossystems.add(resModel.getRossystem())
        for contextModel in context_models:
            for c_rossys in self.generator.rossystemfromContextModel(contextModel, self.tacticsModelResource):
                rossystems.add(c_rossys)

        architectureTQL += self.generator.generateArchitectureInsertQuery(rossystems)
        return architectureTQL

    def generate_node_start_times_query(self):
        current_time = datetime.now().isoformat(timespec='milliseconds')
        return self.generator.generateInsertStartTime(self.rosNodesInRossystems, current_time)

    def create_generators(self):
        namer = NameProvider()
        namer.register(RosNameProvider())
        namer.register(TacticsNameProvider())
        namer.register(ContextNameProvider(namer))
        namer.register(ResolutionNameProvider())
        self.generator = TQLGenerator(namer)
        self.req_generator = RequirementsModelGenerator(namer)

    def request_measures(self):
        create_measures_client = self.create_client(Trigger, f"/{self.kb_node}/create_measures",
                                                    callback_group=self.parallel_cb_group)
        if (create_measures_client.wait_for_service()):
            # java.lang.System.out.println("createMeasures Service is available");
            future = create_measures_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future)
            response = future.result()

            if response.success is False:
                raise RuntimeError("Create Measures Failed")

            self.destroy_client(create_measures_client)
            return response
        else:
            raise RuntimeError("Service is not available")

    def load_all_models(self):
        RosToolingSupport.loadModel(
            RosStandaloneSetupGenerated(),
            self.resSet,
            "ros",
            self.get_parameter("ros_path").value,
        )

        self.ros2Resources = RosToolingSupport.loadModel(
            Ros2StandaloneSetupGenerated(),
            self.resSet,
            "ros2",
            self.get_parameter("ros2_path").value,
        )

        self.rosSystemResources = RosToolingSupport.loadModel(
            RosSystemStandaloneSetupGenerated(),
            self.resSet,
            "rossystem",
            self.get_parameter("rossystem_path").value,
        )

        RosToolingSupport.loadModel(
            FeatureModelStandaloneSetupGenerated(),
            self.resSet,
            "var",
            self.get_parameter("variability_path").value,
        )

        RosToolingSupport.loadModel(
            ResolutionModelStandaloneSetupGenerated(),
            self.resSet,
            "resolution",
            self.get_parameter("resolution_path").value,
        )

        RosToolingSupport.loadModel(
            ContextModelStandaloneSetupGenerated(),
            self.resSet,
            "context",
            self.get_parameter("context_path").value,
        )

        requirementsResources = RosToolingSupport.loadModel(
            RequirementsModelStandaloneSetupGenerated(),
            self.resSet,
            "reqs",
            self.get_parameter("requirements_path").value,
        )

        tacticsResources = RosToolingSupport.loadModel(
            TacticsStandaloneSetupGenerated(),
            self.resSet,
            "tactics",
            self.get_parameter("tactics_path").value,
        )

        self.tacticsModelResource = tacticsResources.get(0)
        self.tacticsModel = (TacticsModel) @ self.tacticsModelResource.getContents().get(0)

        self.requirementsModelResource = requirementsResources.get(0)
        self.requirementsModel = (RequirementsModel) @ \
            self.requirementsModelResource.getContents().get(0)


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
