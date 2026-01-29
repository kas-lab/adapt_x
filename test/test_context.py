import os
import unittest
import launch
import launch_ros
import launch_testing.actions
import rclpy
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import AnyLaunchDescriptionSource
from typedb_tactics.typedb_tactical_retreat_interface import TacticalRetreatInterface
from adapt_x.adaptation_engine import AdaptationEngine


#Use every possible combination of the AdaptDecorator any to check if it is working properly by verifying how many adapt requests AAL receives.


def generate_test_description():

    rfrog_launch_files = os.path.join(get_package_share_directory("rebet_frog"), "launch")
    # ros2 launch rebet_frog spawn_tb3.launch.py gui:=$GUI_VALUE myseed:=$MYSEED_VALUE
    return (
        launch.LaunchDescription(
            [
                SetEnvironmentVariable('REBET_TERMINAL_PREFIX', ''),
                IncludeLaunchDescription(
                    AnyLaunchDescriptionSource(
                        os.path.join(rfrog_launch_files, "spawn_tb3.launch.py")
                    ),
                    launch_arguments={
                        "gui": "false",
                        "myseed": "1",
                    }.items(),
                ),
                launch.actions.TimerAction(
                    period=15, actions=[]
                ),
                IncludeLaunchDescription(
                    AnyLaunchDescriptionSource(
                        os.path.join(rfrog_launch_files, "adaptation_engine_launch.py")
                    ),
                ),
                # Launch tests 0.5 s later
                launch.actions.TimerAction(
                    period=5, actions=[launch_testing.actions.ReadyToTest()]
                ),
            ]
        ),
        {},
    )


# Active tests
class TestAdaptDecorator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("test_rebet")

        # self.kb_interface = TacticalRetreatInterface(
        #                     'localhost:1729',
        #                     'test_tactics_model',
        #                     ['../typedb_tactics/schemas/data_structure/data_structure.tql',
        #                         '../typedb_tactics/schemas/context_model/context_model.tql',
        #                         '../typedb_tactics/schemas/logical_expressions/logical_expression.tql',
        #                         '../typedb_tactics/schemas/ros_model/ros_model.tql',
        #                         '../typedb_tactics/schemas/feature_model/feature_model.tql',
        #                         '../typedb_tactics/schemas/tactics_model/tactics_model.tql',
        #                         '../typedb_tactics/schemas/discover_tactics_model/discover_tactics_model.tql'],
        #                     ['../typedb_tactics/schemas/ros_model/ros_model_example.tql',
        #                         '../typedb_tactics/schemas/tactics_model/tactics_example.tql'],
        #                     force_database=True,
        #                     force_data=True,
        #                     infer=True)
        # xtext_dir = os.path.join(get_package_share_directory("rebet_frog"), "xtext")
        # self.adaptation_engine = AdaptationEngine(
        #     resolution_path=os.path.join(xtext_dir, "rebetfrog.resolution"),
        #     variability_path=os.path.join(xtext_dir, "partial_nav2.var"),
        #     context_path=os.path.join(xtext_dir, "frog.context"),
        #     requirements_path=os.path.join(xtext_dir, "frog.reqs"),
        #     ros_path=[os.path.join(xtext_dir, "builtin_interfaces.ros"),
        #               os.path.join(xtext_dir, "rcl_interfaces.ros"),
        #               os.path.join(xtext_dir, "ros_core.ros"),
        #               os.path.join(xtext_dir, "bond.ros"),
        #               os.path.join(xtext_dir, "theora_image_transport.ros"),
        #               os.path.join(xtext_dir, "common_msgs.ros"),
        #               os.path.join(xtext_dir, "lifecycle_msgs.ros"),
        #               os.path.join(xtext_dir, "map_msgs.ros"),
        #               os.path.join(xtext_dir, "nav2_msgs.ros"),
        #               os.path.join(xtext_dir, "tf2_msgs.ros"),
        #               os.path.join(xtext_dir, "nav_2d_msgs.ros"),
        #               os.path.join(xtext_dir, "dwb_msgs.ros"),
        #               os.path.join(xtext_dir, "rosgraph_msgs.ros"),
        #               os.path.join(xtext_dir, "rebet_msgs.ros")],
        #     ros2_path=[os.path.join(xtext_dir, "velocity_smoother.ros2"),
        #                os.path.join(xtext_dir, "amcl.ros2"),
        #                os.path.join(xtext_dir, "behavior_server.ros2"),
        #                os.path.join(xtext_dir, "bt_navigator.ros2"),
        #                os.path.join(xtext_dir, "controller_server.ros2"),
        #                os.path.join(xtext_dir, "global_costmap.ros2"),
        #                os.path.join(xtext_dir, "lifecycle_manager_localization.ros2"),
        #                os.path.join(xtext_dir, "lifecycle_manager_navigation.ros2"),
        #                os.path.join(xtext_dir, "local_costmap.ros2"),
        #                os.path.join(xtext_dir, "map_server.ros2"),
        #                os.path.join(xtext_dir, "planner_server.ros2"),
        #                os.path.join(xtext_dir, "smoother_server.ros2"),
        #                os.path.join(xtext_dir, "waypoint_follower.ros2"),
        #                os.path.join(xtext_dir, "bt_navigator_navigate_to_pose_rclcpp_node.ros2"),
        #                os.path.join(xtext_dir, "bt_navigator_navigate_through_poses_rclcpp_node.ros2"),
        #                os.path.join(xtext_dir, "camera_controller.ros2"),
        #                os.path.join(xtext_dir, "camera_driver.ros2"),
        #                os.path.join(xtext_dir, "detect_object.ros2"),
        #                os.path.join(xtext_dir, "turtlebot3_diff_drive.ros2"),
        #                os.path.join(xtext_dir, "turtlebot3_imu.ros2"),
        #                os.path.join(xtext_dir, "turtlebot3_joint_state.ros2"),
        #                os.path.join(xtext_dir, "turtlebot3_laserscan.ros2")],
        #     rossystem_path=[os.path.join(xtext_dir, "navigation.rossystem"),
        #                     os.path.join(xtext_dir, "localization.rossystem"),
        #                     os.path.join(xtext_dir, "yolo.rossystem"),
        #                     os.path.join(xtext_dir, "turtlebot3.rossystem"),
        #                     os.path.join(xtext_dir, "rebetfrog.rossystem")],
        #     tactics_path=os.path.join(xtext_dir, "frog.tactics"),
        #     require_nodes_running=False,
        #     verify_parameters=False
        # )

    def tearDown(self):
        self.node.destroy_node()
        # self.kb_interface.delete_database()

    def test_01_context(self, proc_output):
        tactics = self.kb_interface.get_applicable_tactics('rebetmc', '/navigate_to_pose')
        assert len(tactics) == 3

        from eu.coresense.adaptation.generator import TQLGenerator
        from eu.coresense.common import NameProvider

        generator = TQLGenerator(NameProvider())
        query = generator.generateFetchApplicableTactics("rebetmc", "/navigate_to_pose")

        dict_result = kb_interface.fetch_database(str(query))
        response = fetch_query_result_to_ros_msg(dict_result)
        index_lists, index = find_subquery(response.results,
                                        str(TQLGenerator.PARAM_VALUES_SUBQUERY_NAME))
        result_tree = response.results[index]
        adaptation = process_set_parameter(result_tree, index_lists)
        print(str(adaptation))
        assert adaptation.adaptation_target == Adaptation.ROSPARAMETER
        assert adaptation.node_name == "velocity_smoother"
        assert adaptation.parameter_adaptation.name == "max_velocity"