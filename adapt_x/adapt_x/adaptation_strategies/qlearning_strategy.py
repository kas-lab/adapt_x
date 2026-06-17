import os
from adapt_x.adaptation_strategies.adaptation_strategy import AdaptationStrategy
from adapt_x_msgs.srv import GetState
from adapt_x_msgs.srv._get_state import GetState_Response
from adapt_x_msgs.srv import FetchPolicy
from adapt_x_msgs.msg import SymbolicState
from adapt_x_msgs.msg import RelaxedRequirements
from adapt_x.adaptation_engine_common import mean_fulfillment
from rclpy.time import Time
from builtin_interfaces.msg import Time as builtin_time_msg
from statistics import mean, StatisticsError
from adapt_x.q_learning import QLearner
import numpy as np
import pandas as pd
import time


class QLearningStrategy(AdaptationStrategy):
    def __init__(self, kb_node, system_name, context_vars, interface_name, logger):
        self.kb_node = kb_node
        self.current_global_state: SymbolicState = None
        self.reward_window = []
        self.requirements_record = []
        self.system_name = system_name
        self.current_state = None
        self.first_time = True
        self.context_vars = context_vars
        self.logger = logger
        self.actions_handled = 0
        self.interface_name = interface_name
        self.initial_time = str(int(time.time()))
        self.epsilon = 0.0  # TODO: this
        self.mean_task_context = {}
        for c_var in self.context_vars:
            self.mean_task_context[c_var] = [0.0, 0]

    def on_receive_task(self):
        self.actions_handled += 1

    def get_monitor_service(self):
        return GetState, f"{self.kb_node}/get_state"

    def get_initialization_service(self):
        return FetchPolicy, f"/{self.kb_node}/action_details"

    def get_initialization_request(self):
        FetchPolicy.Request(action_name=self.interface_name)

    def get_monitor_request(self):
        GetState.Request(action_name=self.interface_name)

    def on_monitored(self, response: GetState_Response):
        self.previous_global_state = self.current_global_state
        self.current_global_state = response.current_state
        if (self.previous_global_state is not None):
            self.state_changed = True

    def task_state(self, global_state):
        g_name = global_state.state_name
        return SymbolicState(state_name=g_name+self.task_suffix, header=global_state.header)

    def on_requirements(self, reqs_msg: RelaxedRequirements):
        self.requirements_record.append(reqs_msg)
        self.most_recent_fulfillment = reqs_msg.header.stamp
        mean_ff = mean_fulfillment(reqs_msg, self.system_name)
        self.reward = mean_ff
        self.reward_window.append((Time.from_msg(reqs_msg.header.stamp), mean_ff))

    def on_adaptation_loop(self):
        if (self.first_time):
            self.first_time = False
            self.selected_rule = self.all_actions[self.egreedy(self.current_state, True)]
            return self.selected_rule

        if (self.state_changed):
            # self.get_logger().info("State changed, choosing new adaptation")
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

    def on_task_context(self, task_context: dict):
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
        self.current_state = SymbolicState(state_name=g_name+suffix, header=global_state.header)

    def on_initialization(self, result):
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

        self.logger.info("Made the learner")
        self.state_to_index = {}
        [self.state_to_index.update({state.state_name: i})
            for i, state in enumerate(self.all_states)]
        # TODO: consider whether we still need this ^

        self.adaptation_policy = {(ent.state.state_name, ent.feature_adaptation): ent.rule_name
                                  for ent in entries}

        self.q_table_entry_to_rule = {}
        for (state, action), rule in self.adaptation_policy.items():
            for x in self.state_to_index.keys():
                if (state in x):
                    s_i = self.state_to_index[x]
                    a_i = self.all_actions.index(action)
                    self.learner.set_entry((s_i, a_i), 1)  # There is a tactic that suggests this
                    self.q_table_entry_to_rule[(s_i, a_i)] = rule

        # self.learner.print_q_table(self.get_logger(), self.all_actions, self.all_states)

        self.logger.info("Policy Table constructed")

    def mean_rewards_between_states(self, rewards_list: list[tuple[Time, float]],
                                    first_state: SymbolicState,
                                    last_state: SymbolicState) -> float:
        if (first_state is not None):
            earliest = Time.from_msg(first_state.header.stamp)
            # first_state_name = first_state.state_name
        else:
            earliest = Time.from_msg(builtin_time_msg(sec=0, nanosec=0))
            # first_state_name = "NoState"
        latest = Time.from_msg(last_state.header.stamp)
        # self.get_logger().info(f"Earliest {first_state_name} : {str(earliest)}")
        # self.get_logger().info(f"Latest {last_state.state_name} : {str(latest)}")

        # for rqs in rewards_list:
        #     self.get_logger().info(f"reward time {rqs[1]} : {str(rqs[0])}")

        to_be_averaged = [rqs[1] for rqs in rewards_list if rqs[0] > earliest and rqs[0] < latest]

        try:
            return mean(to_be_averaged)
        except StatisticsError:
            self.logger.warn("reward list was empty")
            return 0

    def on_task_complete(self):
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
        output_path = f"{self.get_name()}_{self.initial_time}.csv"
        pd.DataFrame(x).to_csv(output_path, mode='a', header=not os.path.exists(output_path))
        self.requirements_record = []

    def egreedy(self, symbolic_state: SymbolicState, q_learning=False) -> int:
        # e-greedy
        state = symbolic_state.state_name
        choice = np.random.random()
        current_actions = self.all_actions

        if choice < self.epsilon:
            # explore
            self.logger.info("explore")
            next_action = str(np.random.choice(current_actions))
            return self.all_actions.index(next_action)
        else:
            self.logger.info("exploit")

            if (q_learning):
                # self.get_logger().info("qq")
                # self.learner.print_q_table(self.get_logger(), self.all_actions, self.all_states)
                return int(self.learner.exploit(self.state_to_index[state]))
            # else:
            #     # just max
            #     known_mean_reward = [mean(self.reward_registry.get((state, possible_action), 
            # [0]))
            #                          for possible_action in current_actions]
            #     next_action = current_actions[known_mean_reward.index(max(known_mean_reward))]
        return -1