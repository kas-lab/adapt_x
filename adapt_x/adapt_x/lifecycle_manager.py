from typing import Optional

import rclpy
import time

# Node, State and Publisher are aliases for LifecycleNode, LifecycleState and LifecyclePublisher
# respectively.
# In case of ambiguity, the more explicit names can be imported.

from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from std_msgs.msg import Empty

class LifecycleManager(Node):

    def __init__(self, node_name, **kwargs):
        """Construct the node."""
        super().__init__(node_name, **kwargs)

        self.declare_parameter("node_names", [""])
        self.node_names = self.get_parameter("node_names").value
        self.state_changer = {}
        self.ready_map = {}
        self.parallel_group = ReentrantCallbackGroup()
        for node_name in self.node_names:
            self.state_changer[node_name] = self.create_client(ChangeState,
                                                               f"{node_name}/change_state",
                                                               callback_group=self.parallel_group)
            self.ready_map[node_name] = self.create_subscription(Empty, f"{node_name}/ready",
                                                                 self._ready_cb, 10,
                                                                 callback_group=self.parallel_group)

    def _state_changed_cb(self, future):
        self.get_logger().info("state changed")
        result = future.result()
        self.configured.append(result.success)

    def _ready_cb(self, _):
        self.get_logger().info("somebody was ready")
        self.ready.append(True)

    def on_configure(self, state: State) -> TransitionCallbackReturn:
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

        configure_tr = Transition(id=Transition.TRANSITION_CONFIGURE)
        self.configured = []
        self.ready = []

        for node_name in self.node_names:
            fut = self.state_changer[node_name].call_async(ChangeState.Request(transition=configure_tr))
            fut.add_done_callback(self._state_changed_cb)

        while len(self.configured) < len(self.node_names):
            self.get_logger().info(f"Waiting for {str(self.node_names)} nodes to be configured",
                                   throttle_duration_sec=1)

        while len(self.ready) < len(self.node_names):
            self.get_logger().info("Waiting nodes to be ready",
                                   throttle_duration_sec=1)
        self.get_logger().info("on_configure() is called.")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        # Differently to rclcpp, a lifecycle publisher transitions automatically between the inactive and
        # enabled state and viceversa.
        # For that reason, we only need to write an on_configure() and on_cleanup() callbacks, and we don't
        # need to write on_activate()/on_deactivate() callbacks.

        # Log, only for demo purposes
        self.get_logger().info("on_activate() is called.")

        # The default LifecycleNode callback is the one transitioning
        # LifecyclePublisher entities from inactive to enabled.
        # If you override on_activate(), don't forget to call the parent class method as well!!
        return super().on_activate(state)

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

        self.get_logger().info("on_cleanup() is called.")
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

        self.get_logger().info("on_shutdown() is called.")
        return TransitionCallbackReturn.SUCCESS


# A lifecycle node has the same node API
# as a regular node. This means we can spawn a
# node, give it a name and add it to the executor.


def main():
    rclpy.init()

    lc_node = LifecycleManager("lc_manager")

    executor = rclpy.executors.MultiThreadedExecutor()

    rclpy.spin(lc_node, executor=executor)

    lc_node.destroy()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
