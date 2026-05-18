import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.executors import ExternalShutdownException
from aal_msgs.action import AdaptArchitectureConcurrent
import time


class TestActionServer(Node):
    def __init__(self):
        super().__init__('test_action_server')

        self._action_server = ActionServer(
            self,
            AdaptArchitectureConcurrent,
            '/adapt_concurrent',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute a goal."""
        self.get_logger().info('Executing goal...')

        # # Append the seeds for the Fibonacci sequence
        # feedback_msg = Fibonacci.Feedback()
        # feedback_msg.sequence = [0, 1]

        # Start executing the action
        while not goal_handle.is_cancel_requested:
            # Sleep for demonstration purposes
            time.sleep(1)

        goal_handle.canceled()
        self.get_logger().info('Goal canceled')
        return AdaptArchitectureConcurrent.Result()

        # Populate result message
        result = AdaptArchitectureConcurrent.Result()

        self.get_logger().info('Returning result')

        return result


def main(args=None):
    rclpy.init(args=args)

    minimal_action_server = TestActionServer()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(minimal_action_server, executor=executor)

    minimal_action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
