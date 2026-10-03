import rclpy
import sys
from threading import Event
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import Node
from rclpy.executors import MultiThreadedExecutor
from ros_monitors_functions.factory import MonitorFactory
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from adapt_x_msgs.srv import FetchResolutions
from adapt_x_msgs.srv import FetchPlan
from adapt_x_msgs.srv import UpdateHasParamValue
from aal_msgs.srv import AdaptArchitecture
from aal_msgs.msg import Adaptation
from typing import List
import traceback
from std_msgs.msg import Empty
from adapt_x.adaptation_strategies import create_strategy

GREEN_BOLD = "\033[32;1m"
RESET = "\033[0m"
sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST,
                     depth=3)


class ConfigurationManager(Node):
    def __init__(self,
                 kb_node_name: str = "tactical_retreat_kb",
                 system_name: str = "",
                 adaptation_strategy: str = "tactic"):
        super().__init__('follow_path_adaptation')
        self.declare_parameter("kb_node_name", kb_node_name)
        self.declare_parameter("system_name", system_name)
        self.declare_parameter("adaptation_strategy", adaptation_strategy)
        self.declare_parameter("adaptation_timeout", 120.0)

        self.parallel_cb_group = ReentrantCallbackGroup()
        self.kb_node = self.get_parameter("kb_node_name").value
        self.system_name = self.get_parameter("system_name").value
        self.start_grace = None
        self.result_received = False
        self.result_status = None
        self.adaptation_complete = False
        self.goal_received = False
        self.plan_fetched = False
        self.monitored = False
        self.adaptation_plan: List[Adaptation] = []
        self.goal_canceled = False
        self.cancel_due_to_violation = False
        self.cancel_complete = False
        self.ignore_result_on_cancel = False
        self.feedback = None

        self.aal_client = self.create_client(AdaptArchitecture,
                                             "/adapt_architecture",
                                             callback_group=self.parallel_cb_group)

        self.adaptation_policy = {}
        self.monitor_future = None

        self.ongoing_monitor = False
        self.ongoing_adaptation = False

        self.monitor = MonitorFactory()

        self.update_has_param_cli = self.create_client(UpdateHasParamValue,
                                                       f"/{self.kb_node}/update_has_param_value",
                                                       callback_group=self.parallel_cb_group)

        self.ready_publisher = self.create_publisher(Empty, f"{self.get_name()}/ready", 10)
        strategy_name = self.get_parameter("adaptation_strategy").value
        self.strategy = create_strategy(strategy_name,
                                        kb_node_name=self.kb_node,
                                        system_name=self.system_name,
                                        context_vars=[],
                                        interface_name="",
                                        interface_type="",
                                        parent_node=self)
        self.message(f"Using {strategy_name} strategy")
        self.to_adapt = False
        self.deactivation_complete = False
        self.executing = False
        self.message("Created")

    def message(self, msg: str):
        self.get_logger().info(f"{GREEN_BOLD}{msg}{RESET}")

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
        self.message("Configurating Configuration Manager")

        serv_type, serv_name = self.strategy.get_monitor_service()

        self._monitor_client = self.create_client(serv_type, serv_name,
                                                            callback_group=self.parallel_cb_group)

        self.initialization_count = 0

        self.fetch_plan_client = self.create_client(FetchPlan,
                                                    f"/{self.kb_node}/fetch_plan",
                                                    callback_group=self.parallel_cb_group)

        self._timer_ = self.create_timer(2.0, self.monitor_adapt,
                                         callback_group=self.parallel_cb_group)
        self._start_initialization()
        self.message("Reached end of on_configure()")
        return TransitionCallbackReturn.SUCCESS

    def monitor_adapt(self):
        if self.ongoing_adaptation:
            return

        self.send_request()

        if self.to_adapt:
            self.ongoing_adaptation = True
            try:
                self._start_adaptation()
            except Exception:
                self.get_logger().error(
                    f'Global adaptation failed; monitoring stopped:\n{traceback.format_exc()}')
                self._timer_.cancel()
            finally:
                self.ongoing_adaptation = False
                self.to_adapt = False

    def _start_initialization(self):
        """Start the async database initialization chain."""

        self.message("Starting initialization")
        initialization_service = self.strategy.get_initialization_service()

        srv_type, srv_name = initialization_service
        initialization_client = self.create_client(srv_type, srv_name,
                                                   callback_group=self.parallel_cb_group)
        attempts = 0
        self.message(".")
        while not initialization_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info(f'init service with signature {srv_name} not available, waiting again...{attempts}/10')
            attempts += 1
            if (attempts > 10):
                self.get_logger().error("Call failed for initialization. Potential timeout")
                raise RuntimeError("Service call failed for get state. Potential timeout")

        self.message(".")
        try:
            req = self.strategy.get_initialization_request()
        except Exception as e:
            self.get_logger().error(f"Error during initialization: {e}\n{traceback.format_exc()}")

        future = initialization_client.call_async(
            req
        )
        self.message(".")
        timeout_handle = self.create_timer(30.0,
                                           lambda: self._on_timeout(future, "init"))
        future._timeout_handle = timeout_handle
        self.get_logger().info(f"Called initialization service for configuration manager")
        future.add_done_callback(self._on_initialization_complete)

    def _on_timeout(self, future, description):
        """Handle initialization timeout."""
        if not future.done():
            self.get_logger().error(f"{description} service call timed out")
            self.destroy_timer(future._timeout_handle)
            self.trigger_shutdown()
        else:
            # Future completed in time, cancel the timeout
            self.destroy_timer(future._timeout_handle)

    def _on_initialization_complete(self, future):
        """Callback after model insert - proceed to next step."""

        if hasattr(future, '_timeout_handle'):
            self.destroy_timer(future._timeout_handle)

        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().error("Failed to initialize")
                self.trigger_shutdown()
                return

            self.message("Initialization complete, now fetching resolutions.")
            self.strategy.on_initialization(result)

            fetch_adap_client = self.create_client(FetchResolutions,
                                                   f"/{self.kb_node}/fetch_resolutions",
                                                   callback_group=self.parallel_cb_group)
            future = fetch_adap_client.call_async(
                FetchResolutions.Request()
            )

            timeout_handle = self.create_timer(30.0,
                                               lambda: self._on_timeout(future,
                                                                        "fetch_resolutions"))
            future._timeout_handle = timeout_handle
            future.add_done_callback(self._on_adaptations_fetched)
        except Exception as e:
            self.get_logger().error(f"Error during initialization: {e}\n{traceback.format_exc()}")

    def _on_adaptations_fetched(self, future):
        """Callback after running update - proceed to create measures."""
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().error("Failed to fetch resolutions")
                return

            self.resolutions = {}
            for resolution in result.resolutions:
                self.resolutions[resolution.resolution_action_name] = resolution.adaptations

            self.message(f"{self.get_name()} is ready.")
            self.ready_publisher.publish(Empty())

        except Exception as e:
            self.get_logger().error(f"Error in running update callback: {e}")

    def send_request(self):
        attempts = 0
        if (self.monitor_future is None):
            if (not self.ongoing_monitor):
                # self.get_logger().info(f"Req triggered?{str(self._monitor_client.is_activated)}")

                while not self._monitor_client.wait_for_service(timeout_sec=3.0):
                    self.get_logger().info(f'service not available, waiting again...{attempts}/10')
                    attempts += 1
                    if (attempts > 10):
                        self.get_logger().error("Call failed for monitor. Potential timeout")
                        raise RuntimeError("Service call failed for get state. Potential timeout")
                req = self.strategy.get_monitor_request()
                self.ongoing_monitor = True
                self.monitor_future = self._monitor_client.call_async(req)
                # self.get_logger().info("Req sent")
        else:
            if (self.monitor_future.done()):
                response = self.monitor_future.result()
                if response is not None:
                    # self.get_logger().info(f"Response of Monitoring {str(response)}")
                    try:
                        on_mon = self.strategy.on_monitored(response)

                        # self.get_logger().info(f"on mon result {on_mon}")

                        if on_mon is False:
                            self.message("Cancelling due to monitor")
                            self.to_adapt = True

                        self.monitored = True
                        self.ongoing_monitor = False

                    except RuntimeError as e:
                        self.get_logger().error(f"RuntimeError during monitor req: {str(e)}")
                self.monitor_future = None

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_activate() is called.")
        return super().on_activate(state)

    def _start_adaptation(self):
        """Start the async adaptation."""

        self.message("Performing global adaptation")
        action = self.strategy.on_adaptation_loop()
        if action is not None:
            self.message("Adapting")
            self.execute_action(action)
        else:
            self.message("Nothing to adapt, apparently.")

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        # Log, only for demo purposes
        self.get_logger().info("on_deactivate() is called.")

        # Same reasong here that for on_activate().
        # These are the two only cases where you need to call the parent method.
        self.get_logger().info("on_deactivate() complete.")
        self.deactivation_complete = True

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
        self.get_logger().info('on_shutdown() is called.')
        self.destroy_timer(self._timer_)
        sys.exit(1)
        return TransitionCallbackReturn.SUCCESS


    def execute_action(self, action):
        if action is not None:
            self.fetch_plan(action)
            while self.plan_fetched is not True:
                self.get_logger().info("Waiting for plan to be fetched",
                                       throttle_duration_sec=5)

            # self.get_logger().info(f"Adaptation Plan {str(self.adaptation_plan)}")
            self.request_adaptation(self.adaptation_plan)
            self.plan_fetched = False

            while self.adaptation_complete is not True:
                self.get_logger().info("Waiting for adaptation to complete",
                                       throttle_duration_sec=1)

            # self.req_sub = self.create_subscription(RelaxedRequirements, "/requirements",
            #                                         self.requirements_callback,
            #                                         qos_profile=sub_qos,
            #                                         callback_group=self.parallel_cb_group)            
            self.adaptation_plan = []
            self.adaptation_complete = False

    def _on_param_update_complete(self, future):
        """Callback when parameter update completes."""
        try:
            result = future.result()
            if result and hasattr(result, 'success') and result.success:
                self.get_logger().info("Parameter update completed successfully")
            else:
                self.get_logger().error("Parameter update failed")
        except RuntimeError as e:
            self.get_logger().error(f"Error in parameter update callback: {str(e)}")
        finally:
            self.adaptation_complete = True

    def _on_cancel_goal_complete(self, future):
        """Callback when cancelling the goal completes."""
        self.cancel_complete = True

    def _schedule_param_update(self):
        """Schedule parameter update as a non-blocking operation."""
        try:
            # Create the request but don't wait for it synchronously
            update_request = UpdateHasParamValue.Request(feature_name=self.current_feature)
            # Call async and add a callback to handle the result
            if self.update_has_param_cli.wait_for_service(timeout_sec=10.0):
                future = self.update_has_param_cli.call_async(update_request)
                future.add_done_callback(self._on_param_update_complete)
            else:
                raise RuntimeError("UpdateHasParam service not available")

            self.get_logger().info(f"Scheduled parameter update for: {self.current_feature}")
        except RuntimeError as e:
            self.get_logger().error(f"Error scheduling parameter update: {str(e)}")

    def postprocess(self, future):
        self.get_logger().info("Postprocessing adaptation results.")
        self.adaptation_success = False
        try:
            result = future.result()
            self.adaptation_success = result is not None and result.success
        except Exception:
            self.get_logger().error(f'AAL service failed:\n{traceback.format_exc()}')
        finally:
            self.adaptation_complete = True

    def fetch_plan(self, feature_adaptation: str):
        # feature_name = ""
        # if ("desel" in feature_adaptation):
        #     activation = False
        #     feature_name = feature_adaptation.replace("deselect_", "")
        # elif ("select_" in feature_adaptation):
        #     activation = True
        #     feature_name = feature_adaptation.replace("select_", "")
        # self.current_feature = feature_name
        while self.ongoing_monitor is True:
            self.get_logger().info("Waiting for monitoring to finish",
                                   throttle_duration_sec=1)

        request = self.strategy.get_plan_request(feature_adaptation, "")

        self.get_logger().info(f"FetchPlan req {str(request)}")
        if self.fetch_plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info("")
            future = self.fetch_plan_client.call_async(request)

            timeout_handle = self.create_timer(120.0,
                                               lambda: self._on_timeout(future, "fetch_plan"))
            future._timeout_handle = timeout_handle
            future.add_done_callback(self._on_plan_fetched)

        else:
            self.get_logger().error("FetchPlan service not available")
            raise RuntimeError("FetchPlan service not available")

    def _on_plan_fetched(self, future):
        if hasattr(future, '_timeout_handle'):
            self.destroy_timer(future._timeout_handle)
        result = future.result()

        if (result is None):
            self.get_logger().error("FetchPlan failed, cannot postprocess.")
            self.trigger_shutdown()

        self.get_logger().info(f"Fetch plan resolution actions {str(result.resolution_actions)}")
        things = []
        for resolution_action in result.resolution_actions:
            things.extend(self.resolutions[resolution_action])

        self.adaptation_plan = things
        self.plan_fetched = True

    def request_adaptation(self, adaptations: List[Adaptation]):
        self.adaptation_complete = False
        self.adaptation_success = False
        try:
            if not adaptations:
                self.adaptation_success = True
                return
            timeout = float(self.get_parameter('adaptation_timeout').value)
            if timeout <= 0:
                raise ValueError('adaptation_timeout must be positive')
            if not self.aal_client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError('AAL adaptation service not available')
            future = self.aal_client.call_async(
                AdaptArchitecture.Request(adaptations=adaptations))
            completed = Event()
            future.add_done_callback(lambda _: completed.set())
            if not completed.wait(timeout):
                future.cancel()
                raise RuntimeError('AAL response timed out; adaptation outcome unknown')
            self.postprocess(future)
            if not self.adaptation_success:
                raise RuntimeError('AAL adaptation failed; configuration may be partially applied')
        finally:
            self.adaptation_complete = True



def main(args=None):
    rclpy.init(args=args)

    minimal_action_server = ConfigurationManager()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(minimal_action_server, executor=executor)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
