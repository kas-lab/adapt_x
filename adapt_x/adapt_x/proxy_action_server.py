import importlib
import threading
import traceback
from typing import List
from typing import Optional

from aal_msgs.srv import AdaptArchitecture
from action_msgs.msg import GoalStatus
from adapt_x.adaptation_strategies import create_strategy
from adapt_x.lifecycle_client import LifecycleClient
from adapt_x_msgs.msg import DesiredFeatures
from adapt_x_msgs.msg import RelaxedRequirements
from adapt_x_msgs.srv import FetchPlanFromDesiredFeatures
from adapt_x_msgs.srv import UpdateFeaturesSelected
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import CallbackGroup
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.executors import MultiThreadedExecutor
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_services_default
from rclpy.task import Future
from rclpy.type_support import check_is_valid_srv_type
from ros_monitors_functions.factory import MonitorFactory
from std_msgs.msg import Empty


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


GREEN_BOLD = '\033[32;1m'
RESET = '\033[0m'

MODE_GOAL_BOUND = 'goal_bound'
MODE_AUTONOMOUS = 'autonomous'


def validate_operation_mode(operation_mode: str,
                            adaptation_strategy: str) -> str:
    """Validate the startup operation mode against the chosen strategy."""
    if operation_mode not in (MODE_GOAL_BOUND, MODE_AUTONOMOUS):
        raise ValueError(f'Unsupported operation_mode: {operation_mode}')
    if (operation_mode == MODE_AUTONOMOUS and
            adaptation_strategy != 'tactic'):
        raise ValueError(
            'autonomous operation_mode requires the tactic '
            'adaptation strategy')
    return operation_mode


def normalize_actions(actions: Optional[str | List[str]]) -> List[str]:
    """Return strategy output as a list of feature adaptations."""
    if actions is None:
        return []
    if isinstance(actions, str):
        return [actions]
    return list(actions)


def parse_feature_action(action: str) -> tuple[str, bool]:
    """Parse a select/deselect adaptation into feature name and activation."""
    if action.startswith('deselect_'):
        return action.removeprefix('deselect_'), False
    if action.startswith('select_'):
        return action.removeprefix('select_'), True
    raise ValueError(f'Unsupported feature adaptation: {action}')


def desired_features_from_actions(
        raw_actions: Optional[str | List[str]]) -> DesiredFeatures:
    """Convert one validated feature-action batch into desired features."""
    requested_states = {}
    active_features = []
    inactive_features = []
    for action in normalize_actions(raw_actions):
        feature_name, activation = parse_feature_action(action)
        if not feature_name:
            raise ValueError('Feature adaptation has an empty feature name')

        previous_state = requested_states.get(feature_name)
        if previous_state is not None:
            if previous_state != activation:
                raise ValueError(
                    'Conflicting adaptations requested for feature '
                    f'{feature_name}')
            continue

        requested_states[feature_name] = activation
        if activation:
            active_features.append(feature_name)
        else:
            inactive_features.append(feature_name)

    return DesiredFeatures(
        active_features=active_features,
        inactive_features=inactive_features)


class ProxyActionServer(Node):
    def __init__(self, node_name, action_cls: type[object] | None = None,
                 interface_name: str = '',
                 kb_node_name: str = 'tactical_retreat_kb_node',
                 system_name: str = '',
                 task_context_vars: list[str] = [''],
                 continuous_adaptation: bool = True,
                 adaptation_period: float = 0.5,
                 adaptation_strategy: str = '',
                 operation_mode: str = MODE_GOAL_BOUND, **kwargs):
        super().__init__(node_name, **kwargs)
        self.declare_parameter('interface_name', interface_name)
        self.declare_parameter('kb_node_name', kb_node_name)
        self.declare_parameter('system_name', system_name)
        self.declare_parameter('context_variables', task_context_vars)
        self.declare_parameter('interface_type', '')
        self.declare_parameter('goal_attribute', '')
        self.declare_parameter('continuous_adaptation', continuous_adaptation)
        self.declare_parameter('adaptation_period', adaptation_period)
        self.declare_parameter('adaptation_strategy', adaptation_strategy)
        self.declare_parameter('operation_mode', operation_mode)
        # Startup configuration only: runtime parameter changes have no
        # effect on the operation mode.
        self.operation_mode = validate_operation_mode(
            self.get_parameter('operation_mode').value,
            self.get_parameter('adaptation_strategy').value)
        self._autonomous_task = None
        self._autonomous_stop = False
        self._autonomous_loop_done = threading.Event()
        self._autonomous_loop_done.set()
        self.parallel_cb_group = ReentrantCallbackGroup()
        self.kb_node = self.get_parameter('kb_node_name').value
        self.system_name = self.get_parameter('system_name').value
        self.result_received = False
        self.result_status = None
        self.goal_received = False
        self.monitored = False

        self.continuous_adaptation = self.get_parameter(
            'continuous_adaptation').value
        self.aal_client = self.create_client(AdaptArchitecture,
                                             '/adapt_architecture',
                                             callback_group=self.parallel_cb_group)

        i_type = str(self.get_parameter('interface_type').value)

        self.adaptation_policy = {}
        self.monitor_future = None

        self.interface_name = self.get_parameter('interface_name').value
        self.interface_type = get_ros_msg_type_from_string(i_type, 'action') \
            if i_type != '' else action_cls
        self.goal_attribute = self.get_parameter('goal_attribute').value

        c_vars_p_val = self.get_parameter(
            'context_variables').get_parameter_value()
        self.context_vars = c_vars_p_val.string_array_value

        self.monitor = MonitorFactory()

        self.update_features_selected_client = self.create_client(
            UpdateFeaturesSelected,
            f'/{self.kb_node}/update_features_selected',
            callback_group=self.parallel_cb_group
        )

        self.ready_publisher = self.create_publisher(
            Empty, f'{self.get_name()}/ready', 10)

        self.strategy = create_strategy(self.get_parameter('adaptation_strategy').value,
                                        kb_node=self.kb_node,
                                        system_name=self.system_name,
                                        context_vars=self.context_vars,
                                        interface_name=self.interface_name,
                                        logger=self.get_logger())

    def message(self, msg: str):
        self.get_logger().info(f'{GREEN_BOLD}{msg}{RESET}')

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
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST,
                             depth=3)
        self.req_sub = self.create_subscription(RelaxedRequirements, '/requirements',
                                                self.requirements_callback, qos_profile=sub_qos,
                                                callback_group=self.parallel_cb_group)

        serv_type, serv_name = self.strategy.get_monitor_service()

        self._monitor_client = self.create_lifecycle_client(
            serv_type, serv_name, callback_group=self.parallel_cb_group)

        if self.operation_mode == MODE_GOAL_BOUND:
            self._action_client = ActionClient(
                self,
                self.interface_type, self.interface_name)

        self.fetch_plan_client = self.create_client(
            FetchPlanFromDesiredFeatures,
            f'/{self.kb_node}/fetch_plan_from_desired_features',
            callback_group=self.parallel_cb_group)

        self._timer_ = self.create_timer(
            1.0, self.send_request, callback_group=self.parallel_cb_group)
        self._start_initialization()
        return TransitionCallbackReturn.SUCCESS

    def _start_initialization(self):
        """Start the async database initialization chain."""
        initialization_service = self.strategy.get_initialization_service()

        srv_type, srv_name = initialization_service
        initialization_client = self.create_client(
            srv_type,
            srv_name,
            callback_group=self.parallel_cb_group
        )
        # TODO: wait for service
        request = self.strategy.get_initialization_request()
        future = initialization_client.call_async(request)
        future.add_done_callback(self._on_initialization_complete)

    def _on_initialization_complete(self, future):
        """Handle tactic initialization, then expose the configured mode."""
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().error('Failed to initialize')
                self.trigger_shutdown()
                return

            self.message(f'Tactic rules {result.tactic_rules}')
            self.strategy.on_initialization(result)
            self._expose_adaptation_interface()
        except Exception as e:
            self.get_logger().error(
                f'Error during initialization: {e}\n{traceback.format_exc()}')

    def _expose_adaptation_interface(self):
        """Expose the initialized proxy according to its operation mode."""
        if self.operation_mode == MODE_AUTONOMOUS:
            self.ready_publisher.publish(Empty())
            # Nothing else activates an autonomous proxy: no action goal
            # ever arrives, so the proxy activates itself once the tactic
            # rules are loaded.
            self.trigger_activate()
            return

        self._action_server = ActionServer(
            self,
            self.interface_type,
            f'/proxy_{self.interface_name}',
            callback_group=self.parallel_cb_group,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.ready_publisher.publish(Empty())

    def send_request(self):
        attempts = 0
        if (self._monitor_client.is_activated):
            if (self.monitor_future is None):
                while not self._monitor_client.wait_for_service(timeout_sec=3.0):
                    self.get_logger().info(
                        f'service not available, waiting again...{attempts}/10')
                    attempts += 1
                    if (attempts > 10):
                        self.get_logger().error('Call failed for monitor. Potential timeout')
                        raise RuntimeError(
                            'Service call failed for get state. Potential timeout')
                req = self.strategy.get_monitor_request()
                self.monitor_future = self._monitor_client.call_async(req)
            else:
                if (self.monitor_future.done()):
                    response = self.monitor_future.result()
                    if response is not None:
                        try:
                            self.strategy.on_monitored(response)
                            self.monitored = True

                        except RuntimeError as e:
                            self.get_logger().error(
                                f'RuntimeError during monitor req: {str(e)}')
                    self.monitor_future = None

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_activate() is called.')
        if self.operation_mode == MODE_AUTONOMOUS:
            self._start_autonomous_loop()

        return super().on_activate(state)

    def _start_autonomous_loop(self):
        """Start exactly one autonomous adaptation coroutine."""
        if self._autonomous_task is not None:
            return
        self._autonomous_stop = False
        self._autonomous_loop_done.clear()
        self._autonomous_task = self.executor.create_task(
            self._autonomous_loop)

    def _stop_autonomous_loop(self):
        """Stop scheduling iterations and wait for current work to finish."""
        self._autonomous_stop = True
        if (self.operation_mode == MODE_AUTONOMOUS and
                self._autonomous_task is not None):
            self._autonomous_loop_done.wait()

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        # Log, only for demo purposes
        self.get_logger().info('on_deactivate() is called.')
        self._stop_autonomous_loop()
        self.strategy.on_task_complete()

        # Same reasong here that for on_activate().
        # These are the two only cases where you need to call the parent method.
        self.get_logger().info('on_deactivate() complete.')

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
        self._stop_autonomous_loop()
        self.destroy_timer(self._timer_)

        self.get_logger().info('on_shutdown() is called.')
        return TransitionCallbackReturn.SUCCESS

    def requirements_callback(self, reqs_msg: RelaxedRequirements):
        self.get_logger().info('Requirements callback', throttle_duration_sec=10)
        if (self._monitor_client.is_activated):  # TODO: lifecycle subscriber ;)
            self.strategy.on_requirements(reqs_msg)

    def destroy(self):
        self._autonomous_stop = True
        if getattr(self, '_action_server', None) is not None:
            self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info('Received goal request')
        self.task_context = {
            context_var: self.monitor[context_var](
                self, getattr(goal_request, self.goal_attribute))
            for context_var in self.context_vars
            if context_var != '' and
            hasattr(goal_request, self.goal_attribute) and
            context_var in self.monitor
        }

        self.get_logger().info(f'Task context {str(self.task_context)}')
        self.goal_request = goal_request

        try:
            trigger_return = self.trigger_activate()
            self.get_logger().info(
                f'Triggered Activation res {str(trigger_return)}')

            if trigger_return != TransitionCallbackReturn.SUCCESS:
                self.get_logger().error('Failed to activate node')
                return GoalResponse.REJECT
        except RuntimeError as e:
            # Node is already active or in an incompatible state
            self.get_logger().info(
                f'Activation skipped (already active or transitioning): {str(e)}')
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
            raise RuntimeError('Something went wrong sending goal')

    def get_result_callback(self, future):
        self.get_logger().info('Result received')

        wrapped_result = future.result()
        self.result = wrapped_result.result
        self.result_status = wrapped_result.status
        self.trigger_deactivate()
        self.result_received = True

    def cancel_callback(self, server_goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        self.cancel_future = self.client_goal_handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    async def execute_actions(
            self, raw_actions: Optional[str | List[str]]) -> bool:
        """Fetch and execute one complete batch of feature adaptations."""
        desired_features = desired_features_from_actions(raw_actions)
        if (not desired_features.active_features and
                not desired_features.inactive_features):
            return True

        self._monitor_client._enabled = False
        try:
            if not self.fetch_plan_client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(
                    'FetchPlanFromDesiredFeatures service not available')

            result = await self.fetch_plan_client.call_async(
                FetchPlanFromDesiredFeatures.Request(
                    desired_features=desired_features,
                    resolution_model=self.system_name))
            if result is None or not result.success:
                raise RuntimeError('Failed to fetch adaptation plan')

            feature_changes = result.feature_changes
            if (not result.adaptation_plan and
                    not feature_changes.active_features and
                    not feature_changes.inactive_features):
                return True

            if not self.aal_client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError('AAL adaptation service not available')
            adaptation_result = await self.aal_client.call_async(
                AdaptArchitecture.Request(
                    adaptations=result.adaptation_plan))
            if adaptation_result is None or not adaptation_result.success:
                raise RuntimeError('Adaptation failed')

            if not self.update_features_selected_client.wait_for_service(
                    timeout_sec=10.0):
                raise RuntimeError(
                    'UpdateFeaturesSelected service not available')
            update_result = await (
                self.update_features_selected_client.call_async(
                    UpdateFeaturesSelected.Request(
                        feature_changes=feature_changes)))
            if update_result is None or not update_result.success:
                raise RuntimeError('Failed to update feature selected state')
            return True
        finally:
            self._monitor_client._enabled = True

    async def _abort_for_adaptation_error(
            self, server_goal_handle, phase: str, exc: Exception):
        """Cancel forwarded work, shut down, and abort the proxy goal."""
        self.get_logger().error(f'{phase} adaptation failed: {exc}')

        client_goal_handle = getattr(self, 'client_goal_handle', None)
        if (client_goal_handle is not None and
                getattr(client_goal_handle, 'accepted', False)):
            try:
                await client_goal_handle.cancel_goal_async()
            except Exception as cancel_exc:
                self.get_logger().warning(
                    'Failed to cancel forwarded action goal: '
                    f'{cancel_exc}')

        try:
            self.trigger_shutdown()
        except Exception as shutdown_exc:
            self.get_logger().warning(
                f'Failed to shut down adaptation proxy: {shutdown_exc}')

        server_goal_handle.abort()
        return self.interface_type.Result()

    async def _sleep(self, seconds: float):
        """Yield execution until a ROS timer completes."""
        future = Future(executor=self.executor)

        def wake_up():
            if not future.done():
                future.set_result(None)

        timer = self.create_timer(
            seconds, wake_up, callback_group=self.parallel_cb_group)
        try:
            await future
        finally:
            self.destroy_timer(timer)

    async def _run_adaptation_iteration(self):
        """Evaluate the strategy once and execute any resulting batch."""
        actions = self.strategy.on_adaptation_loop()
        if actions:
            await self.execute_actions(actions)

    async def _autonomous_loop(self):
        """Adapt continuously while the node is active."""
        self.message('Autonomous adaptation started')
        terminal_error = None
        try:
            while not self._autonomous_stop:
                if self.monitored:
                    await self._run_adaptation_iteration()
                else:
                    self.get_logger().info(
                        'Waiting for first time monitoring',
                        throttle_duration_sec=5)
                if not self._autonomous_stop:
                    await self._sleep(self._get_adaptation_period())
        except Exception as exc:
            terminal_error = exc
            self.get_logger().error(
                f'Autonomous adaptation terminated: {exc}')
        finally:
            self._autonomous_task = None
            self._autonomous_loop_done.set()
            self.message('Autonomous adaptation stopped')

        if terminal_error is not None:
            try:
                self.trigger_deactivate()
            except Exception as deactivate_exc:
                self.get_logger().warning(
                    'Failed to deactivate autonomous proxy: '
                    f'{deactivate_exc}')

    def _get_adaptation_period(self) -> float:
        """Read and validate the current adaptation period parameter."""
        adaptation_period = self.get_parameter('adaptation_period').value
        if not adaptation_period or adaptation_period <= 0.0:
            raise ValueError('adaptation_period must be greater than zero')
        return adaptation_period

    async def execute_callback(self, server_goal_handle):
        """Execute a goal."""
        self.get_logger().info('executing')
        self.client_goal_handle = None

        while self.monitored is False:
            self.get_logger().info(
                'Waiting for first time monitoring', throttle_duration_sec=5)

        if self.task_context:
            self.strategy.on_task_context(self.task_context)

        try:
            await self._run_adaptation_iteration()
        except (RuntimeError, ValueError) as exc:
            return await self._abort_for_adaptation_error(
                server_goal_handle, 'Initial', exc)

        send_goal_future = self._action_client.send_goal_async(
            self.goal_request)
        send_goal_future.add_done_callback(self._on_goal_sent)
        while (self.goal_received is not True):
            self.get_logger().info(
                'Waiting for response to goal sent', throttle_duration_sec=1)

        self.get_logger().info('Goal Received')
        self.goal_received = False

        if (not self.client_goal_handle.accepted):
            server_goal_handle.canceled()

        while (not self.result_received):
            # Might be better to use cont. adaptation flag in determining state change.
            try:
                await self._run_adaptation_iteration()
                if not self.result_received:
                    await self._sleep(self._get_adaptation_period())
            except (RuntimeError, ValueError) as exc:
                return await self._abort_for_adaptation_error(
                    server_goal_handle, 'Continuous', exc)

        self.get_logger().info('Passing along result')

        # Propagate the status from the real action server
        if self.result_status == GoalStatus.STATUS_SUCCEEDED:
            server_goal_handle.succeed()
        elif self.result_status == GoalStatus.STATUS_ABORTED:
            server_goal_handle.abort()
        elif self.result_status == GoalStatus.STATUS_CANCELED:
            server_goal_handle.canceled()

        self.result_received = False
        return self.result
        # Concurrent adaptation


def main(args=None):
    rclpy.init(args=args)

    minimal_action_server = ProxyActionServer('proxy_action_server')

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(minimal_action_server, executor=executor)

    minimal_action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
