import sys
import os
import rclpy
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from ros_typedb_msgs.srv import Query
from std_srvs.srv import Trigger
from rclpy.executors import MultiThreadedExecutor
from adapt_x_msgs.srv import UpdateRunning
from ament_index_python.packages import get_package_share_directory
from typing import List
import jpype
import jpype.imports

jars_path = os.path.join(get_package_share_directory("adapt_x"), "jars", "rem")

jpype.startJVM(
    classpath=[jars_path + "/*"]
)

from rosToolingSupport import RosToolingSupport
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
from uma.caosd.rhea.AdvancedConstraintsFMmetamodel import AdvancedConstraintsFMmetamodelPackage

from java.lang import String, Integer
from java.util import HashSet


YELLOW_BOLD = "\033[33;1m"
RESET = "\033[0m"


class SetupKB(Node):

    def __init__(self, resolution_path: str = "",
                 var_path: str = "",
                 context_path: str = "",
                 reqs_path: str = "",
                 ros_path: List[str] = [""],
                 ros2_path: List[str] = [""],
                 rossystem_path: List[str] = [""],
                 tactics_path: str = "",
                 require_nodes_running: bool = False,
                 verify_parameters: bool = False,
                 generate_state: bool = True,
                 consider_features_in_state: bool = False
                 ):
        super().__init__('kb_setup_node')
        self.declare_parameter("tactics_path", tactics_path)
        self.declare_parameter("resolution_path", resolution_path)
        self.declare_parameter("var_path", var_path)
        self.declare_parameter("context_path", context_path)
        self.declare_parameter("reqs_path", reqs_path)
        self.declare_parameter("ros_path", ros_path)
        self.declare_parameter("ros2_path", ros2_path)
        self.declare_parameter("rossystem_path", rossystem_path)
        self.declare_parameter("require_nodes_running", require_nodes_running)
        self.declare_parameter("verify_parameters", verify_parameters)
        self.declare_parameter("generate_state", generate_state)
        self.declare_parameter("consider_features_in_state", consider_features_in_state)

        self.do_states = self.get_parameter("generate_state").value
        self.consider_features = self.get_parameter("consider_features_in_state").value
        self.resSet = XtextResourceSet()
        self.parallel_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self.kb_node = "tactical_retreat_kb_node"

        self.load_all_models()

        if not self.check_all_nodes_running():
            if self.get_parameter("require_nodes_running").value:
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

    def progress_message(self, msg: str):
        self.get_logger().info(f"{YELLOW_BOLD}{msg}{RESET}")

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
            for c_rossys in self.generator.rossystemfromContextModel(contextModel,
                                                                     self.tacticsModelResource):
                rossystems.add(c_rossys)

        architectureTQL += self.generator.generateArchitectureInsertQuery(rossystems)
        return architectureTQL

    def generate_initial_query(self):
        resolution_model = self.generator.getResolutionModel(self.tacticsModel,
                                                             self.tacticsModelResource).get(0)
        self.systemName = resolution_model.getName()
        self.sys_name = str(resolution_model.getName())
        initial_insert_query = ""
        initial_insert_query += str(self.generate_architecture_query())

        initial_insert_query += str(self.generator.generateTQL(self.tacticsModel,
                                                               self.tacticsModelResource,
                                                               self.do_states,
                                                               self.consider_features).get(0))
        initial_insert_query += str(self.req_generator.generateInsertQuery(
            self.requirementsModel,
            self.sys_name, False))

        self.initial_insert_query = initial_insert_query

    def create_generators(self):
        namer = NameProvider()
        namer.register(RosNameProvider())
        namer.register(TacticsNameProvider())
        namer.register(ContextNameProvider(namer))
        namer.register(ResolutionNameProvider())
        self.generator = TQLGenerator(namer)
        self.req_generator = RequirementsModelGenerator(namer)

    def load_model(self, ext, setup_obj):
        resources = RosToolingSupport.loadModel(
            setup_obj,
            self.resSet,
            ext,
            self.get_parameter(f"{ext}_path").value,
        )

        self.progress_message(f"Loaded {int(resources.size())}.{ext} files")
        return resources

    def load_all_models(self):
        RosToolingSupport.registerEPackages()
        AdvancedConstraintsFMmetamodelPackage.eINSTANCE.eClass()

        self.load_model("ros", RosStandaloneSetupGenerated())
        self.ros2Resources = self.load_model("ros2", Ros2StandaloneSetupGenerated())
        self.rosSystemResources = self.load_model("rossystem", RosSystemStandaloneSetupGenerated())
        self.load_model("var", FeatureModelStandaloneSetupGenerated())
        self.load_model("resolution", ResolutionModelStandaloneSetupGenerated())
        self.load_model("context", ContextModelStandaloneSetupGenerated())
        requirementsResources = self.load_model("reqs",
                                                RequirementsModelStandaloneSetupGenerated())
        tacticsResources = self.load_model("tactics", TacticsStandaloneSetupGenerated())

        self.tacticsModelResource = tacticsResources.get(0)
        self.tacticsModel = (TacticsModel) @ self.tacticsModelResource.getContents().get(0)

        self.requirementsModelResource = requirementsResources.get(0)
        self.requirementsModel = (RequirementsModel) @ \
            self.requirementsModelResource.getContents().get(0)

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on activate")
        try:
            self._start_database_init()
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Activate error {e}")

            return TransitionCallbackReturn.ERROR

    def on_configure(self, state: State):
        try:
            self.generate_initial_query()
            self.get_logger().info("Configure Success ")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Configure error {e}")

            return TransitionCallbackReturn.ERROR

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        # Log, only for demo purposes
        self.get_logger().info("on_deactivate() is called.")
        # Same reasong here that for on_activate().
        # These are the two only cases where you need to call the parent method.
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """
        Cleanup the node, after a cleaning-up transition is requested.

        on_cleanup callback is being called when the lifecycle node
        enters the "cleaning up" state.

        :return: The state machine either invokes a transition to the "unconfigured" state or stays
        in "inactive" depending on the return value.
        TransitionCallbackReturn.SUCCESS transitions to "unconfigured".
        TransitionCallbackReturn.FAILURE transitions to "inactive".
        TransitionCallbackReturn.ERROR or any uncaught exceptions to "errorprocessing"
        """
        self.destroy_timer(self._timer)
        self.destroy_publisher(self._pub)

        self.get_logger().info('on_cleanup() is called.')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """
        Shutdown the node, after a shutting-down transition is requested.

        on_shutdown callback is being called when the lifecycle node
        enters the "shutting down" state.

        :return: The state machine either invokes a transition to the "finalized" state or stays
        in the current state depending on the return value.
        TransitionCallbackReturn.SUCCESS transitions to "unconfigured".
        TransitionCallbackReturn.FAILURE transitions to "inactive".
        TransitionCallbackReturn.ERROR or any uncaught exceptions to "errorprocessing"
        """
        self.destroy_client(self.typedb_client)
        self.destroy_client(self.update_running_client)
        self.destroy_client(self.create_measures_client)

        self.get_logger().info('on_shutdown() is called.')
        return TransitionCallbackReturn.SUCCESS

    def wait_for_typedb(self, client):
        while not client.wait_for_service(timeout_sec=30.0):
            self.get_logger().info("Waiting for TypeDB to be ready")

    def _start_database_init(self):
        """Start the async database initialization chain."""
        self.typedb_client = self.create_client(Query, f"/{self.kb_node}/query",
                                                callback_group=self.parallel_cb_group)

        self.wait_for_typedb(self.typedb_client)
        # self.get_logger().info(self.initial_insert_query)
        future = self.typedb_client.call_async(
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
            self.update_running_client = self.create_client(UpdateRunning,
                                                            f"/{self.kb_node}/update_running",
                                                            callback_group=self.parallel_cb_group)
            future = self.update_running_client.call_async(
                UpdateRunning.Request(node_names=[str(node.getFrom().getName())
                                                  for node in self.rosNodesInRossystems])
            )
            future.add_done_callback(self._on_running_updated)
        except Exception as e:
            self.get_logger().error(f"Error in model insert callback: {e}")

    def _on_running_updated(self, future):
        """Callback after running update - proceed to create measures."""
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().error("Failed to update running nodes")
                return

            self.get_logger().info("Running nodes updated successfully")

            # Last step: create measures
            self.create_measures_client = self.create_client(Trigger, f"/{self.kb_node}/create_measures",
                                                             callback_group=self.parallel_cb_group)

            future = self.create_measures_client.call_async(Trigger.Request())

            future.add_done_callback(self._on_measures_created)

        except Exception as e:
            self.get_logger().error(f"Error in running update callback: {e}")

    def _on_measures_created(self, future):
        """Callback after creating measures - proceed to shutdown node measures."""
        response = future.result()

        if response.success is False:
            self.get_logger().error("Failed to create measures")

        self.trigger_shutdown()

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


def main(args=None):
    rclpy.init()

    setup_node = SetupKB()
    mt_executor = MultiThreadedExecutor()
    mt_executor.add_node(setup_node)
    setup_node.executor = mt_executor
    mt_executor.spin()

    setup_node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
