import rclpy

# Node, State and Publisher are aliases for LifecycleNode,
# LifecycleState and LifecyclePublisher respectively.
# In case of ambiguity, the more explicit names can be imported.
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.parameter import Parameter
from std_msgs.msg import Empty


class LifecycleManager(Node):

    def __init__(self, node_name, **kwargs):
        """Construct the node."""
        super().__init__(node_name, **kwargs)
        nodes_param = self.declare_parameter(
            'nodes',
            Parameter.Type.STRING_ARRAY
        )

        if nodes_param.type_ == Parameter.Type.NOT_SET:
            raise ValueError("The 'nodes' parameter must be specified")

        self.node_names = nodes_param.value

        if not self.node_names:
            raise ValueError("The 'nodes' parameter must not be empty")

        self.state_changer = {}
        self.ready_map = {}
        self.ready_nodes = set()
        self.parallel_group = ReentrantCallbackGroup()
        for node_name in self.node_names:
            self.state_changer[node_name] = self.create_client(
                ChangeState,
                f"{node_name}/change_state",
                callback_group=self.parallel_group,
            )
            self.ready_map[node_name] = self.create_subscription(
                Empty,
                f"{node_name}/ready",
                lambda _, ready_node=node_name: self._ready_cb(ready_node),
                10,
                callback_group=self.parallel_group,
            )

    def _state_changed_cb(self, future, node_name, results):
        try:
            result = future.result()
            results[node_name] = result.success
        except Exception as exc:
            self.get_logger().error(
                f'Failed to change lifecycle state of {node_name}: {exc}'
            )
            results[node_name] = False

    def _ready_cb(self, node_name):
        self.ready_nodes.add(node_name)

    def _request_transition(self, node_name, transition_id, results):
        client = self.state_changer[node_name]
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f'Waiting for lifecycle state service of {node_name}',
                throttle_duration_sec=1,
            )

        transition = Transition(id=transition_id)
        future = client.call_async(ChangeState.Request(transition=transition))
        future.add_done_callback(
            lambda done_future: self._state_changed_cb(
                done_future,
                node_name,
                results,
            )
        )

    def _wait_transition_results(self, results, action):
        while len(results) < len(self.node_names):
            self.get_logger().info(
                f'Waiting nodes to be {action}',
                throttle_duration_sec=1,
            )

        failed_nodes = [
            node_name for node_name, success in results.items()
            if not success
        ]
        if failed_nodes:
            self.get_logger().error(
                f"Failed to {action}: {', '.join(failed_nodes)}"
            )
            return False

        return True

    def _wait_ready_nodes(self):
        while len(self.ready_nodes) < len(self.node_names):
            missing_nodes = sorted(set(self.node_names) - self.ready_nodes)
            self.get_logger().info(
                f"Waiting nodes to be ready: {', '.join(missing_nodes)}",
                throttle_duration_sec=1,
            )

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """
        Configure the node, after a configuring transition is requested.

        on_configure callback is being called when the lifecycle node
        enters the "configuring" state.

        :return: The state machine either invokes a transition to the
          "inactive" state or stays in "unconfigured" depending on the return
          value.
          TransitionCallbackReturn.SUCCESS transitions to "inactive".
          TransitionCallbackReturn.FAILURE transitions to "unconfigured".
          TransitionCallbackReturn.ERROR or any uncaught exceptions to
          "errorprocessing".
        """
        self.configured = {}
        self.ready_nodes = set()

        if not self.node_names:
            return TransitionCallbackReturn.FAILURE

        for node_name in self.node_names:
            self._request_transition(
                node_name,
                Transition.TRANSITION_CONFIGURE,
                self.configured,
            )

        if not self._wait_transition_results(self.configured, 'configured'):
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info('on_configure() is called.')
        return super().on_configure(state)

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_activate() is called.')
        self._wait_ready_nodes()

        # The default LifecycleNode callback is the one transitioning
        # LifecyclePublisher entities from inactive to enabled.
        # If you override on_activate(), don't forget to call the parent class
        # method as well.
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        # Log, only for demo purposes
        # self.get_logger().info("on_deactivate() is called.")
        # Same reason here that for on_activate().
        # These are the two only cases where you need to call the parent
        # method.
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """
        Cleanup the node, after a cleaning-up transition is requested.

        on_cleanup callback is being called when the lifecycle node
        enters the "cleaning up" state.

        :return: The state machine either invokes a transition to the
          "unconfigured" state or stays in "inactive" depending on the return
          value.
          TransitionCallbackReturn.SUCCESS transitions to "unconfigured".
          TransitionCallbackReturn.FAILURE transitions to "inactive".
          TransitionCallbackReturn.ERROR or any uncaught exceptions to
          "errorprocessing".
        """

        self.get_logger().info("on_cleanup() is called.")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """
        Shutdown the node, after a shutting-down transition is requested.

        on_shutdown callback is being called when the lifecycle node
        enters the "shutting down" state.

        :return: The state machine either invokes a transition to the
          "finalized" state or stays in the current state depending on the
          return value.
          TransitionCallbackReturn.SUCCESS transitions to "unconfigured".
          TransitionCallbackReturn.FAILURE transitions to "inactive".
          TransitionCallbackReturn.ERROR or any uncaught exceptions to
          "errorprocessing".
        """

        self.get_logger().info("on_shutdown() is called.")
        return TransitionCallbackReturn.SUCCESS


def main():
    rclpy.init()

    lc_node = LifecycleManager("lc_manager")

    executor = rclpy.executors.MultiThreadedExecutor()

    rclpy.spin(lc_node, executor=executor)

    lc_node.destroy()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
