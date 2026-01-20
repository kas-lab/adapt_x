#!/usr/bin/env python3
import os
import sys
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.srv import GetParameters
from ros_typedb_msgs.srv import Query
from std_srvs.srv import Trigger
from aal_msgs.srv import AdaptArchitectureTactical
import jpype
from datetime import datetime
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


class AdaptationEngine(Node):

    def __init__(self):
        super().__init__("adaptation_engine")
        self.declare_parameter("tactics_path", "")
        self.declare_parameter("resolution_path", "")
        self.declare_parameter("variability_path", "")
        self.declare_parameter("context_path", "")
        self.declare_parameter("requirements_path", "")
        self.declare_parameter("ros_path", [""])
        self.declare_parameter("ros2_path", [""])
        self.declare_parameter("rossystem_path", [""])
        self.declare_parameter("require_nodes_running", False)
        self.declare_parameter("verify_parameters", False)
        self.resSet = XtextResourceSet()
        self.previous_requirements = None
        RosToolingSupport.registerEPackages()
        self.typedb_client = self.create_client(Query, "/tactical_retreat_kb_node/query")
        self.create_measures_client = self.create_client(Trigger, "/tactical_retreat_kb_node/create_measures")

        self.load_all_models()

        if not self.check_all_nodes_running():
            if (self.get_parameter("require_nodes_running").value):
                self.get_logger().error("Not all ROS nodes are running. Aborting.")
                return
            else:
                self.get_logger().warn(
                    "Not all ROS nodes are running. Continuing anyway due to require_nodes_running=False"
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
        initial_insert_response = self.request_typedb_query(initial_insert_query,
                                                            Query.Request.INSERT,
                                                            "Initial Model Load")

        if initial_insert_response is None or not initial_insert_response.success:
            self.get_logger().error("Failed to insert initial model. Aborting")
            sys.exit(1)
            return

        start_times_res = self.request_typedb_query(str(self.generate_node_start_times_query()),
                                                    Query.Request.INSERT,
                                                    "Node Start Times")
        if start_times_res is None or not start_times_res.success:
            self.get_logger().error("Failed to insert node start times. Aborting")
            sys.exit(1)
            return

        self.create_service(AdaptArchitectureTactical, '/adapt_architecture_tactical', self.tactic_adaptation_callback)

    def tactic_adaptation_callback(self, request, response):
        action_des = request.child_description
        requirements = request.requirements

        need_to_update_req = True
        need_to_update_req = (requirements is not None) and \
                             (not len(requirements) == 0) and \
                             (self.previous_requirements != requirements)
        update_query = self.req_generator.generateUpdateInEffect(requirements)

    def wait_for_typedb(self):
        while not self.typedb_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().info("Waiting for TypeDB to be ready")

    def check_all_node_parameters(self):
        checks = []
        for ros_node in self.rosNodesInRossystems:
            parameterNames = [ros_node.getFrom().getParameter().get((Integer) @ i).getName() for i in range(ros_node.getFrom().getParameter().size())]
            nodeName = ros_node.getName()
            if (len(parameterNames) == 0):
                self.get_logger().warn(f"Node {nodeName} has no parameters defined.")
                checks.append(True)
                continue

            parameter_client = self.create_client(GetParameters,
                "/" + str(nodeName) + "/get_parameters")

            while not parameter_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info(f"get_parameters service for {nodeName} not available, waiting again...")

            request = GetParameters.Request()
            request.names = parameterNames
            response = parameter_client.call(request)
            returned_parameters = response.values
            if(len(returned_parameters) == 0):
                self.get_logger().warn(f"Node {nodeName} returned no parameters, one of the requested parameters may not exist.")
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
        self.requirementsModel = (RequirementsModel) @ self.requirementsModelResource.getContents().get(0)

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
                return future.result()
            else:
                self.get_logger().error(f"Service call failed for {identifier} query.")
                return None
        else:
            self.get_logger().error(f"Typedb query service not available for {identifier} query.")
            return None

def main():
    rclpy.init()

    action_client = AdaptationEngine()

    while not got_response:
        rclpy.spin_once(action_client)

    action_client.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
