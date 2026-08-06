from typing import Optional
import rclpy
import sys
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
from adapt_x.adaptation_strategies import create_strategy


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
        self.parallel_cb_group = ReentrantCallbackGroup()
        self.kb_node = self.get_parameter("kb_node_name").value
        self.system_name = self.get_parameter("system_name").value
        self.result_received = False
        self.result_status = None
        self.adaptation_complete = False
        self.goal_received = False
        self.plan_fetched = False
        self.monitored = False
        self.adaptation_plan: List[Adaptation] = []

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

        self.action_name = i_type.split("/")[-1]

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
        self.report_pub = None
        if (hasattr(self.strategy, "get_report_topic")):
            rep_topic_type, rep_topic_name = self.strategy.get_report_topic()
            self.report_pub = self.create_publisher(rep_topic_type, rep_topic_name, 10)
        self.deactivation_complete = False
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
        self.get_logger().info(f"he {str(client)}")
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
        self.message(self.interface_type)
        self.req_sub = self.create_subscription(RelaxedRequirements, "/requirements",
                                                self.requirements_callback, qos_profile=sub_qos,
                                                callback_group=self.parallel_cb_group)

        serv_type, serv_name = self.strategy.get_monitor_service()

        self._monitor_client = self.create_lifecycle_client(serv_type, serv_name,
                                                            callback_group=self.parallel_cb_group)

        self._action_client = ActionClient(
            self,
            self.interface_type, self.interface_name)

        self.initialization_count = 0

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
                # self.get_logger().info(f"Req triggered? {str(self._monitor_client.is_activated)}")

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
                        self.strategy.on_monitored(response)
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

    def requirements_callback(self, reqs_msg: RelaxedRequirements):
        self.get_logger().info("Requirements callback", throttle_duration_sec=10)
        if (self._monitor_client.is_activated):  # TODO: lifecycle subscriber ;)
            self.strategy.on_requirements(reqs_msg)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info('Received goal request')
        self.task_context = {context_var: self.monitor[context_var](self,
                                                                    getattr(goal_request,
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
        self.client_goal_handle = future.result()
        if (self.client_goal_handle is not None):
            self.goal_received = True
            self._get_result_future = self.client_goal_handle.get_result_async()

            self._get_result_future.add_done_callback(self.get_result_callback)
        else:
            raise RuntimeError("Something went wrong sending goal")

    def get_result_callback(self, future):
        self.get_logger().info('Result received')

        wrapped_result = future.result()
        self.result = wrapped_result.result
        self.result_status = wrapped_result.status
        self.result_received = True

    def cancel_callback(self, server_goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        self.cancel_future = self.client_goal_handle.cancel_goal_async()
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

            self.req_sub = self.create_subscription(RelaxedRequirements, "/requirements",
                                                    self.requirements_callback,
                                                    qos_profile=sub_qos,
                                                    callback_group=self.parallel_cb_group)            
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
        result = future.result()

        if (result is None or not result.success):
            self.get_logger().error("Adaptation failed, cannot postprocess.")
            return

        # self._schedule_param_update()
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
        if (len(adaptations) > 0):
            request = AdaptArchitecture.Request(adaptations=adaptations)

            if self.aal_client.wait_for_service(timeout_sec=10.0):
                self.get_logger().info("")
                future = self.aal_client.call_async(request)
                future.add_done_callback(self.postprocess)

            else:
                self.get_logger().error("AAL adaptation service not available")
                raise RuntimeError("AAL adaptation service not available")
        else:
            self.get_logger().info("No planned adaptations")
            self.adaptation_complete = True

    async def execute_callback(self, server_goal_handle):
        self.get_logger().info('executing')

        """Execute a goal."""

        while self.monitored is False:
            self.get_logger().info("Waiting for first time monitoring",
                                   throttle_duration_sec=5)

        self.strategy.on_task_context(self.task_context)

        next_action = self.strategy.on_adaptation_loop()
        self.destroy_subscription(self.req_sub)
        self.execute_action(next_action)

        send_goal_future = self._action_client.send_goal_async(self.goal_request)
        send_goal_future.add_done_callback(self._on_goal_sent)
        while (self.goal_received is not True):
            self.get_logger().info("Waiting for response to goal sent",
                                   throttle_duration_sec=1)

        self.get_logger().info("Goal Received")
        self.goal_received = False

        if (not self.client_goal_handle.accepted):
            server_goal_handle.canceled()

        while (not self.result_received):
            # Might be better to use cont. adaptation flag in determining state change.
            try:
                action = self.strategy.on_adaptation_loop()

                if (action is not None):
                    self.execute_action(action)

                if (self.report_pub is not None):
                    self.report_pub.publish(self.strategy.get_report_msg())

            except RuntimeError as e:
                self.get_logger().error(f"RuntimeError during tactic adaptation: {str(e)}")
                self.trigger_shutdown()

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
        self.result_received = False
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
