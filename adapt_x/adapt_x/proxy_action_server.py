from typing import Optional
import csv
import rclpy
import importlib
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from action_msgs.msg import GoalStatus
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import Node
from adapt_x.lifecycle_client import LifecycleClient
from adapt_x_msgs.srv import GetState
from rclpy.executors import MultiThreadedExecutor
from ros_monitors_functions.factory import MonitorFactory
from adapt_x_msgs.msg import RelaxedRequirements
from adapt_x.adaptation_engine_common import mean_fulfillment
from rclpy.time import Time
from rclpy.type_support import check_is_valid_srv_type
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_services_default
from rclpy.client import Client
from rclpy.callback_groups import CallbackGroup
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from ros_monitors_functions.path_functions import average_curvature  # TODO: generalize
import numpy as np
import pandas as pd
from adapt_x_msgs.srv import ActionDetails
from adapt_x_msgs.srv import FetchAdaptations
from adapt_x_msgs.srv import FetchPlan
from adapt_x_msgs.srv import UpdateHasParamValue
from adapt_x_msgs.msg import SymbolicState
from aal_msgs.srv import AdaptArchitecture
from aal_msgs.msg import Adaptation
from builtin_interfaces.msg import Time as builtin_time_msg
from statistics import mean, StatisticsError
from typing import List
from adapt_x.q_learning import QLearner
import traceback
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


class ProxyActionServer(Node):
    def __init__(self, action_cls: type[object] | None = None,
                 interface_name: str = "",
                 kb_node_name: str = "tactical_retreat_kb_node",
                 system_name: str = "",
                 task_context_vars: list[str] = [""],
                 continuous_adaptation: bool = True):
        super().__init__('follow_path_adaptation')
        self.declare_parameter("interface_name", interface_name)
        self.declare_parameter("kb_node_name", kb_node_name)
        self.declare_parameter("system_name", system_name)
        self.declare_parameter("context_variables", task_context_vars)
        self.declare_parameter("interface_type", "")
        self.declare_parameter("goal_attribute", "")
        self.declare_parameter("continuous_adaptation", continuous_adaptation)
        self.declare_parameter("policy_learner", "")
        self.declare_parameter("exploration_rate", "")
        self.parallel_cb_group = ReentrantCallbackGroup()

        self.kb_node = self.get_parameter("kb_node_name").value
        self.system_name = self.get_parameter("system_name").value
        self.result_received = False
        self.result_status = None
        self.current_global_state = None
        self.adaptation_complete = False
        self.current_state = None
        self.previous_state = None
        self.goal_received = False
        self.plan_fetched = False
        self.actions_handled = 0
        self.mean_task_context = {}
        self.continuous_adaptation = self.get_parameter("continuous_adaptation").value
        self.epsilon = self.get_parameter("exploration_rate").get_parameter_value().double_value
        self.aal_client = self.create_client(AdaptArchitecture,
                                             "/adapt_architecture",
                                             callback_group=self.parallel_cb_group)

        i_type = str(self.get_parameter("interface_type").value)
        self.reward_window = []
        self.adaptation_policy = {}

        self.interface_name = self.get_parameter("interface_name").value
        self.interface_type = get_ros_msg_type_from_string(i_type, "action") if i_type != "" else action_cls
        self.goal_attribute = self.get_parameter("goal_attribute").value

        self.context_vars = self.get_parameter("context_variables").get_parameter_value().string_array_value

        for c_var in self.context_vars:
            self.mean_task_context[c_var] = [0.0, 0]

        self.monitor = MonitorFactory()  # TODO: make implicit registration possible.
        self.monitor.register_monitor(
            'average_curvature', average_curvature
        )

        self.update_has_param_cli = self.create_client(UpdateHasParamValue,
                                                       f"/{self.kb_node}/update_has_param_value",
                                                       callback_group=self.parallel_cb_group)

        self.get_logger().info(f"name {self.get_name()}")
        self.ready_publisher = self.create_publisher(Empty, f"{self.get_name()}/ready", 10)

        if (self.get_parameter("policy_learner").value == "q-learning"):
            self.get_logger().info("Q learning selected")
            # Actions are -> For each tactic, the alternatives to it.
            # self.all_rules = [str(ruleName) for ruleName in self.generator.ruleNames]
            # self.q_table = np.full((int(all_states.size()), len(self.all_rules)), -np.inf)

            # self.learning_rate = 0.8
            # self.discount_factor = 0.95
            # self.state_to_index = {}
            # [self.state_to_index.update({str(state_name.TQLName): i})
            #     for i, state_name in enumerate(all_states)]

            # for state in self.adaptation_policy.keys():
            #     s_i = self.state_to_index[state]
            #     for rule in self.adaptation_policy[state]:
            #         a_i = self.all_rules.index(rule)
            #         self.q_table[s_i, a_i] = 0

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
        self.get_logger().info("on_configure() called")
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST,
                             depth=3)
        self.req_sub = self.create_subscription(RelaxedRequirements, "/requirements",
                                                self.requirements_callback, qos_profile=sub_qos,
                                                callback_group=self.parallel_cb_group)

        self._get_state_client = self.create_lifecycle_client(GetState, f"{self.kb_node}/get_state",
                                                              callback_group=self.parallel_cb_group)

        self.get_logger().info(f"on_configure() -1 {self.interface_type}")

        self.get_logger().info("on_configure() 0")

        self._action_client = ActionClient(
            self,
            self.interface_type, self.interface_name)

        self.fetch_plan_client = self.create_client(FetchPlan,
                                                    f"/{self.kb_node}/fetch_plan",
                                                    callback_group=self.parallel_cb_group)
        self.requirements_record = []
        self.get_logger().info("on_configure() 1")

        self._timer_ = self.create_timer(1.0, self.send_request,
                                         callback_group=self.parallel_cb_group)
        self._start_creating_learner()
        self.get_logger().info("on_configure() is called.")
        return TransitionCallbackReturn.SUCCESS

    def _start_creating_learner(self):
        """Start the async database initialization chain."""
        self.get_logger().info("_start_creating_learner() called")
        fetch_rules_client = self.create_client(ActionDetails, f"/{self.kb_node}/action_details",
                                                callback_group=self.parallel_cb_group)

        future = fetch_rules_client.call_async(
            ActionDetails.Request(action_name=self.interface_name)
        )
        self.get_logger().info(f"Called ActionDetails service for {self.interface_name}")
        future.add_done_callback(self._on_action_details_retrieved)

    def _on_action_details_retrieved(self, future):
        """Callback after model insert - proceed to next step."""
        self.get_logger().info("_on_action_details_retrieved() callback triggered")
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().error("Failed to get rules")
                self.trigger_shutdown()
                return

            self.get_logger().info("Got the rules")
            self.all_actions = result.reconfigurations
            global_states = result.states  # SymbolicStates

            self.all_states = []

            for global_state in global_states:
                for category in ["below_avg", "above_avg"]:
                    suffix = ""
                    for t_var in self.context_vars:
                        suffix += "_" + t_var + "_" + category
                    new_state_name = global_state.state_name + suffix
                    self.all_states.append(SymbolicState(state_name=new_state_name))

            entries = result.table_entries
            self.learner = QLearner(len(self.all_states), len(self.all_actions),
                                    learning_rate=0.8, discount_factor=0.95)

            self.get_logger().info("Made the learner")
            self.state_to_index = {}
            [self.state_to_index.update({state.state_name: i})  # TODO: consider whether we still need this
                for i, state in enumerate(self.all_states)]

            self.adaptation_policy = {(entry.state.state_name, entry.feature_adaptation): entry.rule_name
                                      for entry in entries}

            self.q_table_entry_to_rule = {}
            for (state, action), rule in self.adaptation_policy.items():
                for x in self.state_to_index.keys():
                    if (state in x):
                        s_i = self.state_to_index[x]
                        a_i = self.all_actions.index(action)
                        self.learner.set_entry((s_i, a_i), 1)  # There is a tactic that suggests this
                        self.q_table_entry_to_rule[(s_i, a_i)] = rule

            # self.learner.print_q_table(self.get_logger(), self.all_actions, self.all_states)

            self.get_logger().info("Policy Table constructed")

            fetch_adap_client = self.create_client(FetchAdaptations,
                                                   f"/{self.kb_node}/fetch_tactic_actions",
                                                   callback_group=self.parallel_cb_group)
            future = fetch_adap_client.call_async(
                FetchAdaptations.Request()
            )
            future.add_done_callback(self._on_adaptations_fetched)
        except Exception as e:
            self.get_logger().error(f"Error in preparing q learning: {e}\n{traceback.format_exc()}")

    def _on_adaptations_fetched(self, future):
        """Callback after running update - proceed to create measures."""
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().error("Failed to fetch adaptations")
                return

            self.resolutions = {}
            for resolution in result.resolutions:
                self.resolutions[resolution.resolution_action_name] = resolution.adaptations

            self.get_logger().info("Adaptations fetched successfully")

            self._action_server = ActionServer(
                self,
                self.interface_type,
                f"/proxy_{self.interface_name}",
                callback_group=self.parallel_cb_group,
                execute_callback=self.execute_callback,
                goal_callback=self.goal_callback,
                cancel_callback=self.cancel_callback)

            self.get_logger().info("Starting bond connection")
            self.ready_publisher.publish(Empty())
            self.get_logger().info("Bond started")

        except Exception as e:
            self.get_logger().error(f"Error in running update callback: {e}")

    def send_request(self):
        if (self._get_state_client.is_activated):
            self.get_logger().info(f"Request triggered? {str(self._get_state_client.is_activated)}")

            response = self._get_state_client.call(GetState.Request(action_name=self.interface_name))
            # self.get_logger().info(f"0 {str(type(future))}")

            # rclpy.spin_until_future_complete(self, future)
            # self.get_logger().info("1")

            if response is not None:
                self.get_logger().info(f"Response of GetState {str(response.current_state)}")
                try:
                    self.previous_global_state = self.current_global_state
                    self.current_global_state = response.current_state
                    if (self.previous_global_state is not None):
                        self.state_changed = True
                except RuntimeError as e:
                    self.get_logger().error(f"RuntimeError during context update: {str(e)}")
            else:
                self.get_logger().error("Service call failed for get state.\
                        Potential timeout")

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_activate() is called.")



        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        # Log, only for demo purposes
        self.get_logger().info("on_deactivate() is called.")
        
        num = len(self.requirements_record)
        x = {"action_num": [self.actions_handled] * num,
             "timestamps": []}
        for req_ful in self.requirements_record:
            x["timestamps"].append(str(req_ful.header.stamp))
            for req in req_ful.requirements:
                if (req.name not in x):
                    x[req.name] = [req.fulfillment]
                else:
                    x[req.name].append(req.fulfillment)

        pd.DataFrame(x).to_csv(f"{self.get_name()}_test.csv")
        self.requirements_record = []
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
        self.destroy_timer(self._timer_)

        self.get_logger().info('on_shutdown() is called.')
        return TransitionCallbackReturn.SUCCESS

    def determine_state(self, task_context):
        self.previous_state = self.current_state
        global_state = self.current_global_state
        g_name = global_state.state_name

        suffix = ""
        for task_context_var in self.context_vars:
            mean_count = self.mean_task_context[task_context_var]
            mean_count[1] += 1
            x = task_context[task_context_var]
            mean_count[0] += (x - mean_count[0]) / mean_count[1]

            if x >= mean_count[0]:
                suffix += "_" + task_context_var + "_" + "above_avg"
            if x < mean_count[0]:
                suffix += "_" + task_context_var + "_" + "below_avg"
        self.task_suffix = suffix
        return SymbolicState(state_name=g_name+suffix, header=global_state.header)

    def requirements_callback(self, reqs_msg: RelaxedRequirements):
        self.get_logger().info("Requirements callback")
        if (self._get_state_client.is_activated):  # TODO: lifecycle subscriber ;) 
            self.requirements_record.append(reqs_msg)
            self.most_recent_fulfillment = reqs_msg.header.stamp
            mean_ff = mean_fulfillment(reqs_msg, self.system_name)
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
        self.task_context = {context_var: self.monitor[context_var](self, getattr(goal_request, self.goal_attribute))
                             for context_var in self.context_vars}

        self.get_logger().info(f'Task context {str(self.task_context)}')
        self.goal_request = goal_request
        # Try to activate, but handle the case where it's already active
        try:
            trigger_return = self.trigger_activate()
            self.get_logger().info(f'Triggered Activation res {str(trigger_return)}')

            if trigger_return != TransitionCallbackReturn.SUCCESS:
                self.get_logger().error('Failed to activate node')
                return GoalResponse.REJECT

            self.actions_handled+=1
            return GoalResponse.ACCEPT
        except RuntimeError as e:
            # Node is already active or in an incompatible state
            self.get_logger().info(f'Activation skipped (already active or transitioning): {str(e)}')

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
        self.trigger_deactivate()
        self.result_received = True

    def cancel_callback(self, server_goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        self.cancel_future = self.client_goal_handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    def mean_rewards_between_states(self, rewards_list: list[tuple[Time, float]],
                                    first_state: SymbolicState,
                                    last_state: SymbolicState) -> float:
        if (first_state is not None):
            earliest = Time.from_msg(first_state.header.stamp)
            first_state_name = first_state.state_name
        else:
            earliest = Time.from_msg(builtin_time_msg(sec=0, nanosec=0))
            first_state_name = "NoState"
        latest = Time.from_msg(last_state.header.stamp)
        self.get_logger().info(f"Earliest {first_state_name} : {str(earliest)}")
        self.get_logger().info(f"Latest {last_state.state_name} : {str(latest)}")

        for rqs in rewards_list:
            self.get_logger().info(f"reward time {rqs[1]} : {str(rqs[0])}")

        to_be_averaged = [rqs[1] for rqs in rewards_list if rqs[0] > earliest and rqs[0] < latest]

        try:
            return mean(to_be_averaged)
        except StatisticsError:
            self.get_logger().warn("reward list was empty")
            return 0

    def egreedy(self, symbolic_state: SymbolicState, q_learning=False) -> int:
        # e-greedy
        state = symbolic_state.state_name
        choice = np.random.random()
        current_actions = self.all_actions

        if choice < self.epsilon:
            # explore
            self.get_logger().info("explore")
            next_action = str(np.random.choice(current_actions))
            return self.all_actions.index(next_action)
        else:
            self.get_logger().info("exploit")

            if (q_learning):
                self.get_logger().info("qq")
                # self.learner.print_q_table(self.get_logger(), self.all_actions, self.all_states)
                return int(self.learner.exploit(self.state_to_index[state]))
            # else:
            #     # just max
            #     known_mean_reward = [mean(self.reward_registry.get((state, possible_action), 
            # [0]))
            #                          for possible_action in current_actions]
            #     next_action = current_actions[known_mean_reward.index(max(known_mean_reward))]
        return -1

    def execute_action(self, action):
        if (action is not None):
            self.selected_rule = self.all_actions[action]
            self.fetch_plan(self.selected_rule)
            while (self.plan_fetched is not True):
                self.get_logger().info("Waiting for plan to be fetched",
                                       throttle_duration_sec=1)

            self.get_logger().info(f"Adaptation Plan {str(self.adaptation_plan)}")
            self.request_adaptation(self.adaptation_plan)
            self.plan_fetched = False

            while (self.adaptation_complete is not True):
                self.get_logger().info("Waiting for adaptation to complete",
                                       throttle_duration_sec=1)

            self.adaptation_plan = None
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
            future = self.update_has_param_cli.call_async(update_request)
            future.add_done_callback(self._on_param_update_complete)

            self.get_logger().info(f"Scheduled parameter update for: {self.current_feature}")
        except RuntimeError as e:
            self.get_logger().error(f"Error scheduling parameter update: {str(e)}")

    def postprocess(self, future):
        self.get_logger().info("Postprocessing adaptation results.")
        result = future.result()

        if (result is None or not result.success):
            self.get_logger().error("Adaptation failed, cannot postprocess.")
            return

        self._schedule_param_update()

    def fetch_plan(self, feature_adaptation):
        self._get_state_client._enabled = False
        if ("desel" in feature_adaptation):
            activation = False
            feature_name = feature_adaptation.replace("deselect_", "")
        elif ("select_" in feature_adaptation):
            activation = True
            feature_name = feature_adaptation.replace("select_", "")
        self.current_feature = feature_name
        request = FetchPlan.Request(feature_name=feature_name, activation=activation)
        self.get_logger().info(f"FetchPlan req {str(request)}")
        if self.fetch_plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info("")
            future = self.fetch_plan_client.call_async(request)
            future.add_done_callback(self._on_plan_fetched)

        else:
            self.get_logger().error("FetchPlan service not available")
            raise RuntimeError("FetchPlan service not available")

    def _on_plan_fetched(self, future):
        result = future.result()

        if (result is None):
            self.get_logger().error("FetchPlan failed, cannot postprocess.")
            self.trigger_shutdown()

        self.get_logger().info(f"Fetch plan resolution actions {str(result.resolution_actions)}")
        things = []
        for resolution_action in result.resolution_actions:
            things.extend(self.resolutions[resolution_action])

        self.adaptation_plan = things
        self._get_state_client._enabled = True
        self.plan_fetched = True

    def request_adaptation(self, adaptations: List[Adaptation]):
        request = AdaptArchitecture.Request(adaptations=adaptations)

        if self.aal_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info("")
            future = self.aal_client.call_async(request)
            future.add_done_callback(self.postprocess)

        else:
            self.get_logger().error("AAL adaptation service not available")
            raise RuntimeError("AAL adaptation service not available")

    def task_state(self, global_state):
        g_name = global_state.state_name
        return SymbolicState(state_name=g_name+self.task_suffix, header=global_state.header)

    async def execute_callback(self, server_goal_handle):
        self.get_logger().info('executing')

        """Execute a goal."""

        while self.current_global_state is None:
            self.get_logger().info("Waiting for first global state",
                                   throttle_duration_sec=1)

        self.current_state = self.determine_state(self.task_context)

        next_action = self.egreedy(self.current_state, True)

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
            if (self.state_changed and self.continuous_adaptation):
                try:
                    self.get_logger().info("State changed, choosing new adaptation")
                    self.state_changed = False
                    new_state = self.task_state(self.current_global_state)
                    prev_state = self.task_state(self.previous_global_state)
                    prev_rule = self.selected_rule
                    reward_window = self.reward_window.copy()
                    self.reward_window.clear()

                    # this can happen if there's no actions associated for the state
                    if (prev_rule is not None):
                        reward = self.mean_rewards_between_states(reward_window,
                                                                  prev_state, new_state)
                        s_index = self.state_to_index[prev_state.state_name]
                        s_prime_index = self.state_to_index[new_state.state_name]
                        action_index = self.all_actions.index(prev_rule)
                        self.learner.update_q_table(s_index,
                                                    action_index, s_prime_index, reward)

                    # action = self.egreedy(new_state, True)

                    # self.execute_action(action)

                except RuntimeError as e:
                    self.get_logger().error(f"RuntimeError during tactic adaptation: {str(e)}")
                    self.trigger_shutdown()

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

    minimal_action_server = ProxyActionServer()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(minimal_action_server, executor=executor)

    minimal_action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
