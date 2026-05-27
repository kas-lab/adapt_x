import rclpy
import importlib
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.executors import ExternalShutdownException
from nav.action import AdaptArchitectureConcurrent
import time
from nav2_msgs.action import FollowPath
from ros_monitors_functions.factory import MonitorFactory
from adapt_x_msgs.msg import RelaxedRequirements
from adapt_x_msgs.msg import SymbolicState
from adapt_x.adaptation_engine_common import mean_fulfillment
from rclpy.time import Time


def get_ros_msg_type_from_string(msg_type_string: str, msg_type='msg'):
    msg_type_list = [part for part in msg_type_string.split('/') if part]

    if len(msg_type_list) == 2:
        package_name, message_name = msg_type_list
    elif len(msg_type_list) == 3:
        package_name = msg_type_list[0]
        msg_type = msg_type_list[1]
        message_name = msg_type_list[2]
    else:
        raise ValueError(
            f'Unsupported message type format: {msg_type_string}')

    msg_module = importlib.import_module(f'{package_name}.{msg_type}')
    return getattr(msg_module, message_name)


class ProxyActionServer(Node):
    def __init__(self):
        super().__init__('proxy_action_server')
        interface_name = "follow_path"
        interface_type = "FollowPath"
        self.context_vars = ["path_length"]
        self.create_subscription(SymbolicState, "symbolic_state",
                                 self.global_state_callback,
                                 10, callback_group=self.parallel_cb_group)
        self.create_subscription(RelaxedRequirements, "/requirements",
                                 self.requirements_callback,
                                 10, callback_group=self.parallel_cb_group)
        self._action_server = ActionServer(
            self,
            get_ros_msg_type_from_string(interface_type),
            '/proxy_' + interface_name,
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        self.monitor = MonitorFactory(self.context_vars)  # TODO: make implicit registration possible.

        self._action_client = ActionClient(
            self,
            get_ros_msg_type_from_string(interface_type), interface_name)

    def global_state_callback(self, state_msg):
        self.get_logger().info("State callback")

        try:
            self.previous_global_state = self.current_global_state
            self.current_global_state = state_msg
            if (self.previous_global_state is not None):
                self.state_changed = self.different_state(state_msg)
        except RuntimeError as e:
            self.get_logger().error(f"RuntimeError during context update: {str(e)}")

    def requirements_callback(self, reqs_msg: RelaxedRequirements):
        self.get_logger().info("Requirements callback")

        self.most_recent_fulfillment = reqs_msg.header.stamp
        mean_ff = mean_fulfillment(reqs_msg, self.systemName)
        self.reward = mean_ff
        self.reward_window.append((Time.from_msg(reqs_msg.header.stamp), mean_ff))
        self.get_logger().info(f"Registered requirement {mean_ff}")

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info('Received goal request')
        task_context = {context_var: self.monitor[context_var](goal_request)
                        for context_var in self.context_vars}

        current_task_state = self.determine_state(task_context)

        next_action = self.learner(current_task_state)

        self.execute_action(next_action)

        send_goal_future = self._action_client.send_goal_async(goal_request)

        goal_handle = send_goal_future.result()

        if (goal_handle.accepted):
            return GoalResponse.ACCEPT

        return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        self.cancel_future = goal_handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute a goal."""

        # Concurrent adaptation


def main(args=None):
    rclpy.init(args=args)

    minimal_action_server = ProxyActionServer()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(minimal_action_server, executor=executor)

    minimal_action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
