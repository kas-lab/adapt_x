#!/usr/bin/env python3
import os
import sys
import rclpy
import json
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.srv import GetParameters
from ros_typedb_msgs.srv import Query
from ros_typedb_msgs.msg import QueryResult, IndexList, ResultTree
from std_srvs.srv import Trigger
from std_msgs.msg import String as std_String
from aal_msgs.srv import AdaptArchitectureTactical, AdaptArchitecture
from adapt_x_msgs.srv import FetchApplicable, UpdateRunning, FetchAdaptations, SatisfiedTactic, UpdateHasParamValue, HypotheticalReqFulfillment, RequirementsFulfillment, HypotheticalReqsFulfillment
from aal_msgs.msg import Adaptation
from typing import List
from datetime import datetime
from rclpy.parameter import parameter_value_to_python, Parameter
import pandas as pd
import functools
import time
import jpype
import jpype.imports
from rclpy.executors import MultiThreadedExecutor
from statistics import mean
from rclpy.lifecycle import TransitionCallbackReturn
import numpy as np

jars_path = os.path.join(get_package_share_directory("adapt_x"), "jars")

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

from java.lang import String, Integer

RUNNING_PARAM_NAME = "require_nodes_running"


def some_adap(adap):
    if(adap.adaptation_target == Adaptation.ROSPARAMETER):
        return "ParameterAdaptation"
    if(adap.adaptation_target == Adaptation.STATETRANSITION):
        return "LifecycleAdaptation"
Adaptation.__str__ = some_adap

def extract_parameter_values(index_lists: List[IndexList],
                             result_tree: ResultTree, INDEX_VAR: str) -> List:
    parameterValues = []

    for i in range(len(index_lists)):
        parameterValues.append(None)

    for index_list in index_lists:
        parameterIndex = -1
        parameterValue = None
        for index in index_list.index:
            q_r = result_tree.results[index]
            if q_r.type != QueryResult.THING:
                continue

            thing = q_r.thing
            if (thing.variable_name == INDEX_VAR):
                for attr in thing.attributes:
                    parameterIndex = attr.value.integer_array_value[0]

            if (thing.variable_name == "value"):
                for attr in thing.attributes:
                    attribute_value = attr.value  # this is a ParameterValue type.
                    valueList = parameter_value_to_python(attribute_value)
                    parameterValue = valueList[0]

            if (parameterIndex >= 0):
                parameterValues[parameterIndex] = parameterValue

    return parameterValues


def extract_string_attribute(result_tree: ResultTree,
                             thing_variable_name: str, att_variable_name: str):
    for q_result in result_tree.results:
        if (q_result.type != QueryResult.THING):
            continue

        thing = q_result.thing
        if (not thing.variable_name == thing_variable_name):
            continue

        for attr in thing.attributes:
            if (attr.variable_name == att_variable_name):
                return attr.value.string_array_value[0]
    return ""  # TODO: throw exception?


def process_set_parameter(result_tree, index_lists):
    node_name = ""

    parameterValues = []

    parameterValues = extract_parameter_values(index_lists,
                                               result_tree, str(TQLGenerator.INDEX_VAR))

    node_name = extract_string_attribute(result_tree, str(TQLGenerator.NODE_VAR),
                                         "node_name")

    param_name = extract_string_attribute(result_tree, str(TQLGenerator.PARAM_VAR),
                                          "parameter_name")

    adaptation = Adaptation()
    adaptation.adaptation_target = Adaptation.ROSPARAMETER

    param = Parameter(param_name, value=parameterValues)

    adaptation.node_name = node_name

    adaptation.parameter_adaptation = param.to_parameter_msg()

    return adaptation


def find_subquery(result_tree_list: List[ResultTree], subquery_name):
    for i in range(len(result_tree_list)):
        result_tree = result_tree_list[i]
        for query_result in result_tree.results:
            if (query_result.type == QueryResult.SUB_QUERY) and \
               (query_result.sub_query_name == subquery_name):
                return query_result.children_index, i

def client_request_init(node, client, request, identifier, timeout_sec=10):
    if (client.wait_for_service(timeout_sec=15.0)):
        node.get_logger().info(f"Sending {identifier} query to Typedb...")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        if future.result() is not None:
            node.get_logger().info(f"Received response for {identifier} query.")
            result = future.result()

            if hasattr(result, 'success') and not result.success:
                node.get_logger().error(f"Typedb {identifier} query failed.")
                raise Exception(f"Typedb {identifier} query had success false.")
            return result
        else:
            node.get_logger().error(f"Service call failed for {identifier} query. Potential timeout")
            raise Exception(f"Typedb query service call for {identifier} failed.")
    else:
        node.get_logger().error(f"Typedb query service not available for {identifier} query.")
        raise Exception(f"Typedb query service not available for {identifier}")


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
                 policy_mode: bool = True):
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
        self.declare_parameter("policy_mode", policy_mode)
        self.reward = None
        self.resSet = XtextResourceSet()
        self.previous_requirements = None
        RosToolingSupport.registerEPackages()
        self.kb_node = "tactical_retreat_kb_node"
        self.exclusive_cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()

        self.parallel_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()
        
        typedb_client = self.create_client(Query, f"/{self.kb_node}/query")
        
        self.satisfied_tactic_client = self.create_client(SatisfiedTactic, f"/{self.kb_node}/satisfied_tactic", callback_group=self.parallel_cb_group)
        self.dream_client_two = self.create_client(HypotheticalReqsFulfillment,f"/{self.kb_node}/fetch_hyp_requirements", callback_group=self.parallel_cb_group)
        
        self.update_has_param_value_client = self.create_client(UpdateHasParamValue, f"/{self.kb_node}/update_has_param_value", callback_group=self.parallel_cb_group)
        self.dream_client = self.create_client(HypotheticalReqFulfillment,f"/{self.kb_node}/fetch_hyp_requirement")

        self.aal_client = self.create_client(AdaptArchitecture, "/adapt_architecture",  callback_group=self.parallel_cb_group)
        update_running_client = self.create_client(UpdateRunning, f"/{self.kb_node}/update_running")
        fetch_adaptations_client = self.create_client(FetchAdaptations, f"/{self.kb_node}/fetch_tactic_actions")

        if self.get_parameter("policy_mode").value:
            self.create_subscription(std_String, "symbolic_state",
                                     self.state_callback,
                                     10, callback_group=self.parallel_cb_group)
            self.create_subscription(std_String, "/requirements",
                                     self.requirements_callback,
                                     10, callback_group=self.parallel_cb_group)

        self.current_feature = "HI_SPEED"
        self.need_to_register_reward = False
        self.current_state = ""
        self.load_all_models()

        if not self.check_all_nodes_running():
            if (self.get_parameter(RUNNING_PARAM_NAME).value):
                self.get_logger().error("Not all ROS nodes are running. Aborting.")
                return
            else:
                self.get_logger().warn(
                    "Not all ROS nodes are running. \
                    Continuing anyway due to require_nodes_running=False"
                )

        if (self.get_parameter("verify_parameters").value):
            if not self.check_all_node_parameters():
                self.get_logger().error("Parameter verification failed. Aborting.")
                sys.exit(1)

        self.create_generators()
        resolution_model = self.generator.getResolutionModel(self.tacticsModel,
                                                             self.tacticsModelResource).get(0)
        self.systemName = resolution_model.getName()
        initial_insert_query = ""
        initial_insert_query += str(self.generate_architecture_query())
        initial_insert_query += str(self.generator.generateTQL(self.tacticsModel,
                                                               self.tacticsModelResource, self.get_parameter("policy_mode").value))
        initial_insert_query += str(self.req_generator.generateInsertQuery(self.
                                                                           requirementsModel,
                                                                           self.systemName, False))

        self.wait_for_typedb(typedb_client)
        
        try:
            initial_insert_response = client_request_init(self, typedb_client,
                                                    Query.Request(query=initial_insert_query, query_type=Query.Request.INSERT),
                                                    "Initial Model Load")
                                                               

            if initial_insert_response is None or not initial_insert_response.success:
                self.get_logger().error("Failed to insert initial model. Aborting")
                sys.exit(1)

            # start_time_res = self.request_typedb_query(str(self.generate_node_start_times_query()),
            #                                            Query.Request.INSERT,
            #                                            "Node Start Times")
            start_time_res = client_request_init(
                self, update_running_client,
                UpdateRunning.Request(node_names=[str(node.getFrom().getName()) for node in self.rosNodesInRossystems]),
                "Update Running")
            if start_time_res is None or not start_time_res.success:
                self.get_logger().error("Failed to insert node start times. Aborting")
                sys.exit(1)

            self.request_measures()
        except Exception as e:
            self.get_logger().error(f"Exception during initial model load: {str(e)}")
            sys.exit(1)

        interfaces_involved = ["navigate_to_pose", "detect_object_srv"] 
        # ^ this should be derived from some awarenss of the BT, ideally it being generated from a DSL

        fetch_applicable_client = self.create_client(FetchApplicable, f"/{self.kb_node}/fetch_applicable_tactics")
        fetch_response = client_request_init(self, fetch_applicable_client,
            FetchApplicable.Request(
                resolution_model=str(self.systemName),
                interface_name=interfaces_involved[0]
            ),                "Fetch Applicable Tactics"
        )
        self.rule_dict = {}
        self.rule_dict[interfaces_involved[0]] = fetch_response.tactics
        self.rule_to_actions = {}
        self.get_logger().info(f"?: {str(fetch_response.tactics)}")
        for tactic in fetch_response.tactics:
            actions_res = client_request_init(
                    self, fetch_adaptations_client,
                    FetchAdaptations.Request(tactic_name=tactic),
                    "Fetch Adaptations")
            self.rule_to_actions[tactic] = actions_res.adaptations

        self.resulting_feature = {}
        self.rules_with_effect = []
        for rule in self.tacticsModel.getAdaptationRules():
            atom_rule = (AtomicRule) @ rule
            rule_body = atom_rule.getRuleBody()
            rule_name = str(atom_rule.getName())
            if(rule_body.getEffect() != None): self.rules_with_effect.append(str(atom_rule.getName()))
            self.resulting_feature[rule_name] = str(rule_body.getAtomicAction().getActions().get((Integer) @ 0).getVariant().getFeature().getName())

        self.previous_rule = None
        if(self.get_parameter("policy_mode").value):
            self.previous_state = ""
            state_action_pairs = self.generator.stateActionPairs

            self.reward_registry = {}
            self.adaptation_policy = {}
            self.get_logger().info(f"Num states: {str(state_action_pairs.keySet().size())}")
            for state_java_string in state_action_pairs.keySet():
                state = str(state_java_string)
                self.get_logger().info(f"State found: {state}")
                self.adaptation_policy[state] = [str(rule_name_java) for rule_name_java in state_action_pairs.get(state_java_string)]
                for action in self.resulting_feature:
                    self.reward_registry[(state,action)] = []


            # for rule in self.rules_with_effect:
            #     fetch_response = client_request_init(self, self.dream_client,
            #         HypotheticalReqFulfillment.Request(
            #             rule_name=rule,
            #             system_name=str(self.systemName),
            #             requirement_name="MoveSafely"
            #         ),                "Fetch Hypothetical Flflment"
            #     )




            self.get_logger().info(f"Reward registry \n{self.reward_registry}")
            self.get_logger().info(f"Adaptation Policy \n{self.adaptation_policy}")



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

        if(self.get_parameter("policy_mode").value):
            self.create_service(AdaptArchitectureTactical,
                    '/adapt_architecture_tactical', self.policy_tactic_adaptation_callback)
        else:
            self.create_service(AdaptArchitectureTactical,
                                '/adapt_architecture_tactical', self.tactic_adaptation_callback)

        return TransitionCallbackReturn.SUCCESS

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
                self.get_logger().error(f"Service call failed for {identifier} query. Potential timeout")
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
            raise Exception("AAL adaptation service not available")

    def postprocess(self, future):
        # self.get_logger().info("Postprocessing adaptation results.")
        result = future.result()

        if(result is None or not result.success):
            self.get_logger().error("Adaptation failed, cannot postprocess.")
            return

        # update_query = self.generator.generateUpdateHasParamValue(self.current_feature)

        # self.request_typedb_query(str(update_query), Query.Request.UPDATE, "Update has_param_value")

        # req_ful_res = client_request(
        #         self, self.req_ful_client,
        #         RequirementsFulfillment.Request(system_name=str(self.systemName), in_effect=True),
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
            update_hasparam_res = self.client_request(
                    self.update_has_param_value_client,
                    UpdateHasParamValue.Request(feature_name=self.current_feature),
                    "Update Has Param Value")

            # self.get_logger().info(f"Reward registry \n{self.reward_registry}")
        except Exception as e:
            self.get_logger().error(f"Error updating parameter value: {str(e)}")

    def state_callback(self, state_msg):
        try:
            self.current_state = state_msg.data
        except Exception as e:
            self.get_logger().error(f"Exception during context update: {str(e)}")

    def requirements_callback(self, req_msg):
        try:
            req_dict_list = json.loads(req_msg.data)
            self.get_logger().info(f"Req dict list {str(req_dict_list)}")
            self.reward = mean(float(req_dict["fulfillment"]) for req_dict in req_dict_list if req_dict["system"] == str(self.systemName))
            # self.get_logger().info(f"Received req fulfillment")
        except Exception as e:
            self.get_logger().error(f"Exception during context update: {str(e)}")

    def tactic_adaptation_callback(self, request, response):
        try:
            interface_name = request.child_description.interface_name


            rules_for_interface = self.rule_dict.get(interface_name, [])

            if(not rules_for_interface):
                self.get_logger().info(f"No applicable tactics found for interface {interface_name}. No adaptation will be applied.")
                response.success = True
                return response

            sat_res = self.client_request(
                self.satisfied_tactic_client,
                SatisfiedTactic.Request(tactics=rules_for_interface),
                "Satisfied Tactic", 1.1)

            satisfied = sat_res.satisfied
            if(not any(satisfied)):
                self.get_logger().info(f"No applicable tactics found for interface {interface_name} that satisfy their guard.")
                response.success = True
                return response

            #satisfied is a list of bools, rules_for interface, we want the first rule satisfied

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

        except Exception as e:
            self.get_logger().error(f"Exception during tactic adaptation: {str(e)}")
            response.success = False
            return response


    def pi(self, state, epsilon=0.5):
        #e-greedy
        choice = np.random.random()
        current_actions = self.adaptation_policy.get(state,None)
        if(current_actions == None):
            raise Exception("state " + state + " did not retrieve actions from the policy")
        if(current_actions == []):
            return None #TODO: for now we don't adapt, but maybe we should dream instead or something?
        if choice < epsilon:
            #explore

            next_action = str(np.random.choice(current_actions))
        else:
            #exploit
            known_mean_reward = [mean(self.reward_registry.get((state,possible_action),[0])) \
                                  for possible_action in current_actions]
            next_action = current_actions[known_mean_reward.index(max(known_mean_reward))]
        
        return next_action

    def policy_tactic_adaptation_callback(self, request, response):
        try:
            state = self.current_state #freeze it for this callback
            start_time = time.perf_counter()
            
            #Simple, based on the state we're in we pick a rule that is available according to the policy, according to explore/exploit.
            interface_name = request.child_description.interface_name

            rules_for_interface = self.rule_dict.get(interface_name, [])

            # self.get_logger().info(f"These are the rules for interface {interface_name}: {rules_for_interface}")
            if(not rules_for_interface):
                self.get_logger().info(f"No applicable tactics found for interface {interface_name}. No adaptation will be applied.")
                response.success = True
                return response

            if(self.need_to_register_reward and self.reward):
                x = self.reward_registry.get((self.previous_state,self.previous_rule),["NERP"])
                if x == ["NERP"]: self.get_logger().info("DICT MISSED")
                else:
                    self.get_logger().info("DICT HIT")
                    x.append(self.reward)
                self.need_to_register_reward = False

            
            #satisfied is a list of bools, rules_for interface, we want the first rule satisfied

            selected_rule = None
            selected_rule = self.pi(state)
            if not selected_rule:
                response.success = True
                return response
            feature_name = self.resulting_feature[selected_rule]
            selected_actions = self.rule_to_actions[selected_rule]
            non_selected_actions = rules_for_interface.copy()
            non_selected_actions.remove(selected_rule)

            # TODO: dreaming
            # sat_res = self.client_request(
            #     self.dream_client_two,
            #     HypotheticalReqsFulfillment.Request(rule_names=non_selected_actions, system_name=str(self.systemName)),
            #     "Satisfied Tactic", 1.1)

            self.current_feature = feature_name

            self.request_adaptation(selected_actions)

            self.previous_rule = selected_rule
            self.previous_state = state
            self.need_to_register_reward = True

            elapsed_time = time.perf_counter() - start_time
            self.get_logger().info(f'Policy adaptation done in {elapsed_time:.4f} seconds.')
            response.success = True
            return response
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            self.get_logger().info(f'Satisfied not done in {elapsed_time:.4f} seconds.')
            self.get_logger().error(f"Exception during tactic adaptation: {str(e)}")
            response.success = False
            return response

    def wait_for_typedb(self, client):
        while not client.wait_for_service(timeout_sec=30.0):
            self.get_logger().info("Waiting for TypeDB to be ready")

    def _schedule_param_update(self):
        """Schedule parameter update as a non-blocking operation."""
        try:
            # Create the request but don't wait for it synchronously
            update_request = UpdateHasParamValue.Request(feature_name=self.current_feature)
            
            # Call async and add a callback to handle the result
            future = self.update_has_param_value_client.call_async(update_request)
            future.add_done_callback(self._on_param_update_complete)
            
            self.get_logger().info(f"Scheduled parameter update for feature: {self.current_feature}")
        except Exception as e:
            self.get_logger().error(f"Error scheduling parameter update: {str(e)}")

    def _on_param_update_complete(self, future):
        """Callback when parameter update completes."""
        try:
            result = future.result()
            if result and hasattr(result, 'success') and result.success:
                self.get_logger().info(f"Parameter update completed successfully")
            else:
                self.get_logger().error("Parameter update failed")
        except Exception as e:
            self.get_logger().error(f"Error in parameter update callback: {str(e)}")

    def check_all_node_parameters(self):
        checks = []
        for ros_node in self.rosNodesInRossystems:
            parameterNames = [ros_node.getFrom().getParameter().get((Integer) @ i).getName()
                              for i in range(ros_node.getFrom().getParameter().size())]
            nodeName = ros_node.getName()
            if (len(parameterNames) == 0):
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
            if (len(returned_parameters) == 0):
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
        for resModel in res_models:
            architectureTQL += self.generator.generateArchitectureInsertQuery(resModel)
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
        create_measures_client = self.create_client(Trigger, f"/{self.kb_node}/create_measures")
        if (create_measures_client.wait_for_service()):
            # java.lang.System.out.println("createMeasures Service is available");
            future = create_measures_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future)
            response = future.result()

            if response.success is False:
                raise Exception("Create Measures Failed")

            self.destroy_client(create_measures_client)
            return response
        else:
            raise Exception("Service is not available")

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

    # def request_typedb_query(self, query: str, query_type: bytes, identifier: str):
    #     request = Query.Request()
    #     request.query = query
    #     request.query_type = query_type

    #     if (self.typedb_client.wait_for_service(timeout_sec=15.0)):
    #         self.get_logger().info(f"Sending {identifier} query to Typedb...")
    #         future = self.typedb_client.call_async(request)
    #         rclpy.spin_until_future_complete(self, future)
    #         if future.result() is not None:
    #             self.get_logger().info(f"Received response for {identifier} query.")
    #             result = future.result()

    #             if not result.success:
    #                 self.get_logger().error(f"Typedb {identifier} query failed.")
    #                 raise Exception(f"Typedb {identifier} query had success false.")
    #             return result
    #         else:
    #             self.get_logger().error(f"Service call failed for {identifier} query.")
    #             raise Exception(f"Typedb query service call for {identifier} failed.")
    #     else:
    #         self.get_logger().error(f"Typedb query service not available for {identifier} query.")
    #         raise Exception(f"Typedb query service not available for {identifier}")


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
