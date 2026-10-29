from typing import Optional
import rclpy
import sys
from threading import Event
import importlib
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from action_msgs.msg import GoalStatus
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import Node
from adapt_x.lifecycle_client import LifecycleClient
from rclpy.executors import MultiThreadedExecutor
from ros_monitors_functions.factory import MonitorFactory
from adapt_x_msgs.msg import RelaxedRequirements
from rclpy.type_support import check_is_valid_srv_type
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_services_default
from rclpy.client import Client
from rclpy.callback_groups import CallbackGroup
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from adapt_x_msgs.srv import FetchResolutions
from adapt_x_msgs.srv import FetchPlan
from adapt_x_msgs.srv import UpdateHasParamValue
from aal_msgs.srv import AdaptArchitecture
from aal_msgs.msg import Adaptation
from typing import List
import traceback
from std_msgs.msg import Empty
import time
from adapt_x.adaptation_strategies import create_strategy
from adapt_x.handle_goal_resend import construct_repeat_goal


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


GREEN_BOLD = "\033[32;1m"
RESET = "\033[0m"
sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST,
                     depth=3)


class ProxyActionServer(Node):
    def __init__(self, action_cls: type[object] | None = None,
                 interface_name: str = "",
                 kb_node_name: str = "tactical_retreat_kb",
                 system_name: str = "",
                 task_context_vars: list[str] = [""],
                 continuous_adaptation: bool = True,
                 adaptation_strategy: str = ""):
        super().__init__('follow_path_adaptation')
        self.declare_parameter("interface_name", interface_name)
        self.declare_parameter("kb_node_name", kb_node_name)
        self.declare_parameter("system_name", system_name)
        self.declare_parameter("context_variables", task_context_vars)
        self.declare_parameter("interface_type", "")
        self.declare_parameter("goal_attribute", "")
        self.declare_parameter("continuous_adaptation", continuous_adaptation)
        self.declare_parameter("adaptation_strategy", adaptation_strategy)
        self.declare_parameter("adaptation_timeout", 120.0)
        self.declare_parameter("grace_period", 10)

        self.parallel_cb_group = ReentrantCallbackGroup()
        self.kb_node = self.get_parameter("kb_node_name").value
        self.system_name = self.get_parameter("system_name").value
        self.grace_period = self.get_parameter("grace_period").value
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
        self.req_sub = None
        self._monitor_client = None
        self._action_client = None
        self.fetch_plan_client = None
        self._timer = None

        self.continuous_adaptation = self.get_parameter("continuous_adaptation").value
        self.aal_client = self.create_client(AdaptArchitecture,
                                             "/adapt_architecture",
                                             callback_group=self.parallel_cb_group)

        i_type = str(self.get_parameter("interface_type").value)

        self.adaptation_policy = {}
        self.monitor_future = None

        self.interface_name = self.get_parameter("interface_name").value
        self.interface_type = get_ros_msg_type_from_string(i_type, "action") \
            if i_type != "" else action_cls
        self.goal_attribute = self.get_parameter("goal_attribute").value

        c_vars_p_val = self.get_parameter("context_variables").get_parameter_value()
        self.context_vars = c_vars_p_val.string_array_value
        self.ongoing_monitor = False
        self.monitor = MonitorFactory()
        self.total_initializations = 0
        self.finished_initializations = 0



        self.action_name = i_type.rsplit("/", maxsplit=1)[-1]

        self.update_has_param_cli = self.create_client(UpdateHasParamValue,
                                                       f"/{self.kb_node}/update_has_param_value",
                                                       callback_group=self.parallel_cb_group)

        self.ready_publisher = self.create_publisher(Empty, f"{self.get_name()}/ready", 10)
        strategy_name = self.get_parameter("adaptation_strategy").value
        self.strategy = create_strategy(strategy_name,
                                        kb_node_name=self.kb_node,
                                        system_name=self.system_name,
                                        context_vars=self.context_vars,
                                        interface_name=self.interface_name,
                                        interface_type=self.action_name,
                                        parent_node=self)
        self.message(f"Using {strategy_name} strategy")
        self.report_pub = None
        if hasattr(self.strategy, "get_report_topic"):
            rep_topic_type, rep_topic_name = self.strategy.get_report_topic()
            self.report_pub = self.create_publisher(rep_topic_type, rep_topic_name, 10)
        self.deactivation_complete = False
        self.executing = False
        self.message("Created")

    def message(self, msg: str):
        self.get_logger().info(f"{GREEN_BOLD}{msg}{RESET}")

    def create_lifecycle_client(
        self,
        srv_type,
        srv_name: str,
        *,
        qos_profile: QoSProfile = qos_profile_services_default,
        callback_group: Optional[CallbackGroup] = None
    ) -> Client:
        """
        Create a new service client.

        :param srv_type: The service type.
        :param srv_name: The name of the service.
        :param qos_profile: The quality of service profile to apply the service client.
        :param callback_group: The callback group for the service client. If ``None``, then the
            nodes default callback group is used.
        """
        if callback_group is None:
            callback_group = self.default_callback_group
        check_is_valid_srv_type(srv_type)
        failed = False
        try:
            with self.handle:
                client_impl = _rclpy.Client(
                    self.handle,
                    srv_type,
                    srv_name,
                    qos_profile.get_c_qos_profile())
        except ValueError:
            failed = True
        if failed:
            self._validate_topic_or_service_name(srv_name, is_service=True)

        client = LifecycleClient(
            self.context,
            client_impl, srv_type, srv_name, qos_profile,
            callback_group)
        callback_group.add_entity(client)
        self._clients.append(client)
        self._wake_executor()
        self.add_managed_entity(client)
        return client

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
        self.message(str(self.interface_type))
        self.req_sub = self.create_subscription(RelaxedRequirements, "/requirements",
                                                self.requirements_callback, qos_profile=sub_qos,
                                                callback_group=self.parallel_cb_group)

        serv_type, serv_name = self.strategy.get_monitor_service()

        self._monitor_client = self.create_lifecycle_client(serv_type, serv_name,
                                                            callback_group=self.parallel_cb_group)

        self._action_client = ActionClient(
            self,
            self.interface_type, self.interface_name)


        self.fetch_plan_client = self.create_client(FetchPlan,
                                                    f"/{self.kb_node}/fetch_plan",
                                                    callback_group=self.parallel_cb_group)

        self._timer_ = self.create_timer(2.0, self.send_request,
                                         callback_group=self.parallel_cb_group)
        self._start_initialization()
        self.message("Reached end of on_configure()")
        return TransitionCallbackReturn.SUCCESS

    def _start_initialization(self):
        """Start the async database initialization chain."""

        self.message("Starting initialization")
        initialization_services = self.strategy.get_initialization_service()
        initialization_requests = self.strategy.get_initialization_request()
        self.total_initializations = len(initialization_services)
        for i, initialization_service in enumerate(initialization_services):
            self.message(f"Initialization {i+1}/{self.total_initializations}")
            srv_type, srv_name = initialization_service
            initialization_client = self.create_client(srv_type, srv_name,
                                                    callback_group=self.parallel_cb_group)
            attempts = 0
            self.message(".")
            while not initialization_client.wait_for_service(timeout_sec=3.0):
                self.get_logger().info(f'init service with signature {srv_name} not available, waiting again...{attempts}/10')
                attempts += 1
                if attempts > 10:
                    self.get_logger().error("Call failed for initialization. Potential timeout")
                    raise RuntimeError("Service call failed for get state. Potential timeout")

            self.message(".")
            try:
                req = initialization_requests[i]
            except Exception as e:
                self.get_logger().error(f"Error during initialization: {e}\n{traceback.format_exc()}")

            future = initialization_client.call_async(
                req
            )
            self.message(".")
            timeout_handle = self.create_timer(30.0,
                                            lambda: self._on_timeout(future, "init"))
            future._timeout_handle = timeout_handle
            self.get_logger().info(f"Called initialization service for {self.interface_name}")
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

            self.strategy.on_initialization(result)
            self.finished_initializations+=1

            if self.finished_initializations >= self.total_initializations:
                self.message("Initialization complete, now fetching resolutions.")

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

            self._action_server = ActionServer(
                self,
                self.interface_type,
                f"/proxy_{self.interface_name}",
                callback_group=self.parallel_cb_group,
                execute_callback=self.execute_callback,
                goal_callback=self.goal_callback,
                cancel_callback=self.cancel_callback)

            self.message(f"{self.get_name()} is ready.")
            self.ready_publisher.publish(Empty())

        except Exception as e:
            self.get_logger().error(f"Error in running update callback: {e}")

    def send_request(self):
        attempts = 0
        if (self.monitor_future is None):
            if (self._monitor_client.is_activated and not self.ongoing_monitor):
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

                        # # self.get_logger().info(f"on mon result {on_mon}")

                        # if on_mon is False and not self.in_grace_period():
                        #     self.message("Cancelling due to monitor")
                        #     self.cancel_due_to_violation = True

                        self.monitored = True
                        self.ongoing_monitor = False

                    except RuntimeError as e:
                        self.get_logger().error(f"RuntimeError during monitor req: {str(e)}")
                self.monitor_future = None

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_activate() is called.")

        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        # Log, only for demo purposes
        self.get_logger().info("on_deactivate() is called.")
        try:
            self.strategy.on_task_complete()
        except Exception as e:
            self.get_logger().error(f"Error during deactivation: {e}\n{traceback.format_exc()}")

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

    def in_grace_period(self) -> bool:
        if self.start_grace is None:
            return False
        from rclpy.constants import S_TO_NS
        elapsed = self.get_clock().now() - self.start_grace
        self.start_grace = None
        within_period = (elapsed.nanoseconds / S_TO_NS) < self.grace_period
        if not within_period:
            self.start_grace = None
        return within_period

    def requirements_callback(self, reqs_msg: RelaxedRequirements):
        # self.get_logger().info(f"Requirements callbackkk {self._monitor_client.is_activated} {self.executing}")
        if (self._monitor_client.is_activated and self.executing):  # TODO: lifecycle subscriber ;)
            on_req = None

            on_req = self.strategy.on_requirements(reqs_msg)

            # self.get_logger().info(f"on req result {on_req}")

            if on_req is False and not self.in_grace_period():
                self.message("Cancelling due to requirement")

                self.cancel_due_to_violation = True

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info('Received goal request')
        self.goal_canceled = False
        self.task_context = {}

        if (self.context_vars != ['']):
            self.task_context = {context_var:
                                 self.monitor[context_var](self, getattr(goal_request,
                                                                         self.goal_attribute))
                                 for context_var in self.context_vars}

            self.get_logger().info(f'Task context {str(self.task_context)}')
        self.goal_request = goal_request

        try:
            trigger_return = self.trigger_activate()

            if trigger_return != TransitionCallbackReturn.SUCCESS:
                self.get_logger().error('Failed to activate node')
                return GoalResponse.REJECT
        except RuntimeError as e:
            # Node is already active or in an incompatible state
            self.get_logger().info(f'Activation skipped (already active or transitioning): {str(e)}')
            return GoalResponse.REJECT

        self.strategy.on_receive_task()
        return GoalResponse.ACCEPT

    def _on_goal_sent(self, future):
        self.get_logger().info("Goal was sent callback activated")
        self.client_goal_handle = future.result()
        # self.get_logger().info(f"With this future result {self.client_goal_handle}")

        if (self.client_goal_handle is not None):
            self.goal_received = True
            self._get_result_future = self.client_goal_handle.get_result_async()

            self._get_result_future.add_done_callback(self.get_result_callback)
        else:
            raise RuntimeError("Something went wrong sending goal")

    def feedback_callback(self, feedback_msg):
        self.feedback = feedback_msg.feedback
        # self.get_logger().info('Received feedback')

    def get_result_callback(self, future):
        self.get_logger().info('Result received')

        wrapped_result = future.result()
        self.result = wrapped_result.result
        self.result_status = wrapped_result.status

        # If we requested cancellation due to a violation, ignore this
        # result so the execute loop continues. Clear the flag and
        # return without setting `result_received`.
        if self.ignore_result_on_cancel:
            self.get_logger().info('Result ignored due to local cancellation request')
            self.ignore_result_on_cancel = False
            return

        self.result_received = True

    def cancel_goal(self):
        if (not self.goal_canceled):
            cancel_future = self.client_goal_handle.cancel_goal_async() # TODO: add callback?
            return cancel_future
        else:
            self.get_logger().warn("Tried to cancel already cancelled goal")

    def cancel_callback(self, server_goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        self.cancel_goal()
        return CancelResponse.ACCEPT

    def execute_action(self, action):
        if (action is not None):
            self.fetch_plan(action)
            while (self.plan_fetched is not True):
                self.get_logger().info("Waiting for plan to be fetched",
                                       throttle_duration_sec=5)

            # self.get_logger().info(f"Adaptation Plan {str(self.adaptation_plan)}")
            self.request_adaptation(self.adaptation_plan)
            self.plan_fetched = False

            while (self.adaptation_complete is not True):
                self.get_logger().info("Waiting for adaptation to complete",
                                       throttle_duration_sec=1)

            # self.req_sub = self.create_subscription(RelaxedRequirements, "/requirements",
            #                                         self.requirements_callback,
            #                                         qos_profile=sub_qos,
            #                                         callback_group=self.parallel_cb_group)
            self.adaptation_plan = []
            self._monitor_client._enabled = True
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
        self._monitor_client._enabled = False
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

        request = self.strategy.get_plan_request(feature_adaptation, self.action_name)

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

    async def execute_callback(self, server_goal_handle):
        try:
            return await self._execute_callback(server_goal_handle)
        except Exception:
            self.get_logger().error(f'Task aborted after failure:\n{traceback.format_exc()}')
            self.executing = False
            self._monitor_client._enabled = False
            self.adaptation_plan = []
            self.plan_fetched = False
            self.cancel_due_to_violation = False
            if server_goal_handle.is_active:
                server_goal_handle.abort()
            self.trigger_deactivate()
            return self.interface_type.Result()

    async def _execute_callback(self, server_goal_handle):
        self.get_logger().info('executing')

        """Execute a goal."""

        while self.monitored is False:
            self.get_logger().info("Waiting for first time monitoring",
                                   throttle_duration_sec=5)

        self.strategy.on_task_context(self.task_context)

        next_action = self.strategy.on_adaptation_loop()

        self.execute_action(next_action)

        send_goal_future = self._action_client.send_goal_async(
            self.goal_request,
            feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self._on_goal_sent)
        while (self.goal_received is not True):
            self.get_logger().info("Waiting for response to goal sent",
                                   throttle_duration_sec=1)

        self.get_logger().info("Goal Received")
        self.goal_received = False
        self.executing = True

        if (not self.client_goal_handle.accepted):
            server_goal_handle.canceled()

        while (not self.result_received):
            # Might be better to use cont. adaptation flag in determining state change.
            try:
                if (self.cancel_due_to_violation):
                    self.cancel_due_to_violation = False
                    self.message("Cancellation due to violation")
                    # Mark that we should ignore the next action result
                    # since we initiated the cancel locally due to a violation.
                    self.ignore_result_on_cancel = True
                    cancel_future = self.cancel_goal()
                    self.cancel_complete = False
                    if (cancel_future is not None):
                        self.message("A")
                        cancel_future.add_done_callback(self._on_cancel_goal_complete)

                        while self.cancel_complete is not True:
                            self.get_logger().info("Waiting for goal cancellation",
                                                   throttle_duration_sec=1)

                        self.get_logger().info("Pretending to think")

                        action = self.strategy.on_adaptation_loop()

                        if (action is not None):
                            self.message("Adapting")
                            self.execute_action(action)
                        else:
                            self.message("Nothing to adapt, apparently.")

                        time.sleep(2)  # Not sure why this is needed, but Nav2 pretends to accept a goal but doesn't actually.
                        while not self._action_client.wait_for_server(timeout_sec=3.0):
                            self.get_logger().info('action server not avaialble.')
                        resent_goal = construct_repeat_goal(self.goal_request, self.get_logger(), self.feedback)
                        self.get_logger().info(f'{str(self.feedback)}')
                        send_goal_future = self._action_client.send_goal_async(
                            resent_goal,
                            feedback_callback=self.feedback_callback)
                        send_goal_future.add_done_callback(self._on_goal_sent)
                        while (self.goal_received is not True):
                            self.get_logger().info("Waiting for response to goal sent",
                                                   throttle_duration_sec=1)

                        self.get_logger().info("Goal Received")
                        self.start_grace = self.get_clock().now()
                        self.goal_received = False
                        self.executing = True

                # Let's say we never adapt during ongoing behavior.
                # action = self.strategy.on_adaptation_loop()

                # if (action is not None):
                #     self.execute_action(action)

                if (self.report_pub is not None):
                    self.report_pub.publish(self.strategy.get_report_msg())

            except RuntimeError as e:
                self.get_logger().error(f"RuntimeError during tactic adaptation: {str(e)}")
                raise

        self.message("Out of execution loop")
        self.result_received = False

        # If I was the one to cancel it due to a violation, we can repeat the goal after reconf?

        self.executing = False

        self.get_logger().info(f"Passing along result {self.result_status}")

        self.strategy.process_goal_result(int(self.result_status))
        # Propagate the status from the real action server
        if self.result_status == GoalStatus.STATUS_SUCCEEDED:
            server_goal_handle.succeed()
        elif self.result_status == GoalStatus.STATUS_ABORTED:
            server_goal_handle.abort()
        elif self.result_status == GoalStatus.STATUS_CANCELED:
            server_goal_handle.canceled()

        self.trigger_deactivate()

        while self.deactivation_complete is not True:
            self.get_logger().info("Waiting for deactivation to complete",
                                   throttle_duration_sec=1)

        self.deactivation_complete = False
        return self.result
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
