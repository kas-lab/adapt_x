#!/usr/bin/env python3
import os
import sys
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.srv import GetParameters
from ros_typedb_msgs.srv import Query
from ros_typedb_msgs.msg import QueryResult, IndexList, ResultTree
from std_srvs.srv import Trigger
from aal_msgs.srv import AdaptArchitectureTactical, AdaptArchitecture
from aal_msgs.msg import Adaptation
from typing import List
from datetime import datetime
from rclpy.parameter import parameter_value_to_python, Parameter

import jpype
import jpype.imports

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

from java.lang import String, Integer
got_response = False


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
                 verify_parameters: bool = False):
        super().__init__("adaptation_engine")
        self.declare_parameter("tactics_path", tactics_path)
        self.declare_parameter("resolution_path", resolution_path)
        self.declare_parameter("variability_path", variability_path)
        self.declare_parameter("context_path", context_path)
        self.declare_parameter("requirements_path", requirements_path)
        self.declare_parameter("ros_path", ros_path)
        self.declare_parameter("ros2_path", ros2_path)
        self.declare_parameter("rossystem_path", rossystem_path)
        self.declare_parameter("require_nodes_running", require_nodes_running)
        self.declare_parameter("verify_parameters", verify_parameters)
        self.resSet = XtextResourceSet()
        self.previous_requirements = None
        RosToolingSupport.registerEPackages()
        kb_node = "tactical_retreat_kb_node"
        self.typedb_client = self.create_client(Query, f"/{kb_node}/query")
        self.create_measures_client = self.create_client(Trigger, f"/{kb_node}/create_measures")
        self.aal_client = self.create_client(AdaptArchitecture, "/adapt_architecture")

        self.load_all_models()

        if not self.check_all_nodes_running():
            if (self.get_parameter("require_nodes_running").value):
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
                return

        self.create_generators()

        initial_insert_query = ""
        initial_insert_query += str(self.generate_architecture_query())
        initial_insert_query += str(self.generator.generateTQL(self.tacticsModel,
                                                               self.tacticsModelResource))
        initial_insert_query += str(self.req_generator.generateInsertQuery(self.
                                                                           requirementsModel,
                                                                           "rebetmc", False))

        self.wait_for_typedb()

        try:
            initial_insert_response = self.request_typedb_query(initial_insert_query,
                                                                Query.Request.INSERT,
                                                                "Initial Model Load")

            if initial_insert_response is None or not initial_insert_response.success:
                self.get_logger().error("Failed to insert initial model. Aborting")
                sys.exit(1)
                return

            start_time_res = self.request_typedb_query(str(self.generate_node_start_times_query()),
                                                       Query.Request.INSERT,
                                                       "Node Start Times")
            if start_time_res is None or not start_time_res.success:
                self.get_logger().error("Failed to insert node start times. Aborting")
                sys.exit(1)
                return

            self.request_measures()
        except Exception as e:
            self.get_logger().error(f"Exception during initial model load: {str(e)}")
            sys.exit(1)
            return

        self.create_service(AdaptArchitectureTactical,
                            '/adapt_architecture_tactical', self.tactic_adaptation_callback)

        self.get_logger().info("Adaptation Engine initialized and ready.")

    def request_adaptation(self, adaptations: List[Adaptation]):
        request = AdaptArchitecture.Request(adaptations=adaptations)

        if self.aal_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info("")
            future = self.aal_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                self.get_logger().info("Received response for AAL adaptation request")
                result = future.result()

                if not result.success:
                    self.get_logger().error("AAL adaptation request failed.")
                    raise Exception("AAL adaptation request had success false.")
                return result
            else:
                self.get_logger().error("Service call failed for AAL adaptation request.")
                raise Exception("AAL adaptation service call failed.")

    def tactic_adaptation_callback(self, request, response):
        try:
            interface_name = request.child_description.interface_name
            requirements = request.requirements

            need_to_update_req = True
            need_to_update_req = (requirements is not None) and \
                                 (not len(requirements) == 0) and \
                                 (self.previous_requirements != requirements)

            if (need_to_update_req):
                update_query = self.req_generator.generateUpdateInEffect(requirements)
                self.request_typedb_query(str(update_query), Query.Request.UPDATE,
                                          "Update Requirements in effect")
            # TODO: consider multiple resolution models
            resolution_model = self.generator.getResolutionModel(self.tacticsModel,
                                                                 self.tacticsModelResource).get(0)
            fetch_q = self.generator.generateFetchApplicableTactics(resolution_model.getName(),
                                                                    interface_name)
            fetch_response = self.request_typedb_query(str(fetch_q),
                                                       Query.Request.FETCH,
                                                       "Fetch Applicable Tactics")

            result_tree_list = fetch_response.results

            if len(result_tree_list) == 0:
                self.get_logger().info("No applicable tactics found.")
                response.success = True
                return response

            self.get_logger().info(f"Found {len(result_tree_list)} applicable tactics.")

            param_values_sq = find_subquery(result_tree_list,
                                            str(TQLGenerator.PARAM_VALUES_SUBQUERY_NAME))
            if param_values_sq is None:
                raise Exception("No parameter values subquery found in results. \
                                        Aborting adaptation.")

            index_lists, result_tree_index = param_values_sq

            adaptation = process_set_parameter(result_tree_list[result_tree_index], index_lists)

            self.request_adaptation([adaptation])

            feature_name = extract_string_attribute(result_tree_list[result_tree_index],
                                                    str(TQLGenerator.FEATURE_VAR), "feature_name")

            update_query = self.generator.generateUpdateHasParamValue(feature_name)

            self.request_typedb_query(str(update_query), Query.Request.UPDATE, "Update has_param_value")
            response.success = True
            return response
        except Exception as e:
            self.get_logger().error(f"Exception during tactic adaptation: {str(e)}")
            response.success = False
            return response

    def wait_for_typedb(self):
        while not self.typedb_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().info("Waiting for TypeDB to be ready")

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
        if (self.create_measures_client.wait_for_service()):
            # java.lang.System.out.println("createMeasures Service is available");
            future = self.create_measures_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future)
            response = future.result()

            if response.success is False:
                raise Exception("Create Measures Failed")

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

    def request_typedb_query(self, query: str, query_type: bytes, identifier: str):
        request = Query.Request()
        request.query = query
        request.query_type = query_type

        if (self.typedb_client.wait_for_service(timeout_sec=15.0)):
            self.get_logger().info(f"Sending {identifier} query to Typedb...")
            future = self.typedb_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                self.get_logger().info(f"Received response for {identifier} query.")
                result = future.result()

                if not result.success:
                    self.get_logger().error(f"Typedb {identifier} query failed.")
                    raise Exception(f"Typedb {identifier} query had success false.")
                return result
            else:
                self.get_logger().error(f"Service call failed for {identifier} query.")
                raise Exception(f"Typedb query service call for {identifier} failed.")
        else:
            self.get_logger().error(f"Typedb query service not available for {identifier} query.")
            raise Exception(f"Typedb query service not available for {identifier}")


def main():
    rclpy.init()

    action_client = AdaptationEngine()

    while not got_response:
        rclpy.spin_once(action_client)

    action_client.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
