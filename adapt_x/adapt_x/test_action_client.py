import rclpy
from rclpy.action import ActionClient, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
import sys
from aal_msgs.action import AdaptArchitectureConcurrent
import time
from adapt_x_msgs.msg import RelaxedRequirement
from adapt_x_msgs.msg import RelaxedRequirements
from adapt_x_msgs.msg import SymbolicState
from std_msgs.msg import Header


class TestActionClient(Node):

    def __init__(self):
        super().__init__('test_action_client')
        self._action_client = ActionClient(self, AdaptArchitectureConcurrent, '/adapt_concurrent')

    def send_goal(self):
        goal_msg = AdaptArchitectureConcurrent.Goal()

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        time.sleep(5)
        self.get_logger().info('Trying to cancel')

        self.cancel_future = goal_handle.cancel_goal_async()

        self.cancel_future.add_done_callback(self.cancel_callback)

    def cancel_callback(self, future):
        self.get_logger().info('Cancel callback invoked!')
        result = future.result()
        rclpy.shutdown()


class MockActionClient(Node):

    def __init__(self):
        super().__init__('mock_action_client')
        self._action_client = ActionClient(self, AdaptArchitectureConcurrent, '/adapt_concurrent')

        self.reqs_publisher = self.create_publisher(
            RelaxedRequirements, 'requirements', 10)

        self.state_publisher = self.create_publisher(
            SymbolicState, 'symbolic_state', 10)

        print("?")
        self.publish_state(1)
        time.sleep(1)
        self.publish_req(0.6)

    def create_header(self):
        return Header(stamp=self.get_clock().now().to_msg())

    def publish_req(self, ff):
        self.get_logger().info(f"Publishing req ff {ff}")

        reqs_msgs = [RelaxedRequirement(name='MockRequirement',
                                        fulfillment=ff,
                                        system_name="rem_tiago")]

        reqs_msg = RelaxedRequirements(header=self.create_header(), requirements=reqs_msgs)
        self.reqs_publisher.publish(reqs_msg)

    def publish_state(self, num):
        self.get_logger().info(f"Publishing state state_expr__{num}")
        self.state_publisher.publish(
                SymbolicState(state_name=f"state_expr__{num}",
                              header=self.create_header()))

    def feedback_callback(self, feedback):
        printed = 'Received feedback: {0}'.format(str(feedback.feedback.adaptations_executed))
        self.get_logger().info(printed)

    def send_goal(self):
        self.get_logger().info("Sending goal")

        goal_msg = AdaptArchitectureConcurrent.Goal()

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        time.sleep(2)
        self.publish_state(4)
        time.sleep(1)
        self.publish_req(0.5)
        time.sleep(2)
        self.publish_state(3)
        time.sleep(1)
        self.publish_req(0.3)
        time.sleep(1)
        self.publish_req(0.5)
        time.sleep(6)
        self.get_logger().info('Trying to cancel')

        self.cancel_future = goal_handle.cancel_goal_async()

        self.cancel_future.add_done_callback(self.cancel_callback)

    def cancel_callback(self, future):
        self.get_logger().info('Cancel callback invoked!')
        result = future.result()
        rclpy.shutdown()



def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 2:
        print("Missing argument!")
        sys.exit(1)

    print(sys.argv[1])
    print(str(sys.argv[1] == "test"))
    if (str(sys.argv[1]) == "test"):
        print("I'm here??")
        action_client = TestActionClient()

        action_client.send_goal()

        rclpy.spin(action_client)

    if (str(sys.argv[1]) == "mock"):
        action_client = MockActionClient()

        action_client.send_goal()

        rclpy.spin(action_client)

if __name__ == '__main__':
    main()