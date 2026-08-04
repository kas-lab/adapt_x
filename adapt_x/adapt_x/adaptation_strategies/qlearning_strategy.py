import os
from adapt_x.adaptation_strategies.adaptation_strategy import AdaptationStrategy
from adapt_x_msgs.srv import GetState
from adapt_x_msgs.srv._get_state import GetState_Response
from adapt_x_msgs.srv import FetchPolicy
from adapt_x_msgs.msg import SymbolicState
from adapt_x_msgs.msg import RelaxedRequirements
from adapt_x_msgs.msg import QTable
from adapt_x.adaptation_engine_common import mean_fulfillment
from rclpy.time import Time
from builtin_interfaces.msg import Time as builtin_time_msg
from statistics import mean, StatisticsError
from adapt_x.q_learning import QLearner
import numpy as np
import pandas as pd
import time
from typing import Optional
from action_msgs.msg import GoalStatus

PURP_BOLD = "\033[35;1m"
RESET = "\033[0m"


class QLearningStrategy(AdaptationStrategy):
    def __init__(self, kb_node_name, system_name, context_vars,
                 interface_name, interface_type, strategy_name, parent_node):
        self.kb_node_name = kb_node_name
        self.current_global_state: SymbolicState = None
        self.reward_window = []
        self.requirements_record = []
        self.system_name = system_name
        self.current_state = None
        self.previous_state = None
        self.state_changed = False
        self.first_time = True
        self.context_vars = context_vars
        self.logger = parent_node.get_logger()
        self.actions_handled = 0
        self.interface_name = interface_name
        self.interface_type = interface_type
        self.initial_time = str(int(time.time()))

        parent_node.declare_parameter(f"{strategy_name}.discount_factor", -1.0)
        parent_node.declare_parameter(f"{strategy_name}.epsilon", -1.0)
        parent_node.declare_parameter(f"{strategy_name}.learning_rate", -1.0)
        parent_node.declare_parameter(f"{strategy_name}.load_latest_table", False)
        self.epsilon = parent_node.get_parameter(f"{strategy_name}.epsilon").value
        self.mean_task_context = {}
        self.applied_adaptations = []
        self.selected_rule = None
        self.failed_goals = 0
        self.all_actions = []
        self.load_latest_table = parent_node.get_parameter(f"{strategy_name}.load_latest_table").value
        self.df = parent_node.get_parameter(f"{strategy_name}.discount_factor").value
        self.lr = parent_node.get_parameter(f"{strategy_name}.learning_rate").value
        self.rng = np.random.default_rng(seed=1337)
        learner_description = f"_df{self.df}_lf{self.lr}_e{self.epsilon}"
        self.save_dir = f"{self.interface_name}_{learner_description}_{self.initial_time}"
        os.mkdir(self.save_dir)
        for c_var in self.context_vars:
            self.mean_task_context[c_var] = [0.0, 0]

    def message(self, msg: str):
        self.logger.info(f"{PURP_BOLD}{msg}{RESET}")

    def on_receive_task(self):
        self.message("on_receive_task")
        self.actions_handled += 1
        self.first_time = True

    def get_monitor_service(self):
        return GetState, f"{self.kb_node_name}/get_state"

    def get_initialization_service(self):
        return FetchPolicy, f"/{self.kb_node_name}/fetch_policy"

    def get_initialization_request(self):
        return FetchPolicy.Request(action_name=self.interface_type)

    def get_monitor_request(self):
        return GetState.Request(action_name=self.interface_type)

    def get_report_topic(self):
        return QTable, "/q_table"

    def get_report_msg(self):
        q_table = self.learner.q_table
        q_table_msg = QTable()

        q_table_msg.state_labels = [str(state.state_name) for state in self.all_states]
        q_table_msg.action_labels = self.all_actions
        q_table_msg.rows = len(self.all_states)
        q_table_msg.cols = len(self.all_actions)

        q_table_msg.values = q_table.flatten().tolist()
        return q_table_msg

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
        self.applied_adaptations.append(str(self.selected_rule))
        self.most_recent_fulfillment = reqs_msg.header.stamp
        mean_ff = mean_fulfillment(reqs_msg, None)
        self.reward = mean_ff
        self.reward_window.append((Time.from_msg(reqs_msg.header.stamp), mean_ff))

    # The str represents a variant/feature being selected or deselect like select_var_name
    def on_adaptation_loop(self) -> Optional[str]:
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
                x = self.learner.update_q_table(s_index,
                                                action_index, s_prime_index, reward)

                self.logger.info(f"update equation {str(x)}")

    def on_task_context(self, task_context: dict):
        self.previous_state = self.current_state
        global_state = self.current_global_state  # The state of Enviromental Context Vars
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
        self.logger.info(f"All actions {len(self.all_actions)} {str(self.all_actions)}")
        self.logger.info(f"Global States {len(global_states)} {str(global_states)}")

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
                                learning_rate=self.lr, discount_factor=self.df,
                                save_dir=self.save_dir)

        if (self.load_latest_table):
            try:
                self.learner.load("latest_table.npy")
            except FileNotFoundError:
                self.logger.warn("There was no latest table to load.")
                pass

        # self.logger.info(f"Made the learner¸ {len(self.all_states)} states")
        # self.logger.info(f"Made the learner¸ {str(self.all_states)} states")

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

        self.learner.print_q_table(self.logger, self.all_actions, self.all_states)

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
        self.message("on_task_complete")
        num = len(self.requirements_record)
        x = {"action_num": [self.actions_handled] * num,
             "timestamps": [],
             "adaptation": self.applied_adaptations,
             "failed_goals": [self.failed_goals] * num}
        for req_ful in self.requirements_record:
            x["timestamps"].append(str(req_ful.header.stamp))
            for req in req_ful.requirements:
                if (req.name not in x):
                    x[req.name] = [req.fulfillment]
                else:
                    x[req.name].append(req.fulfillment)

        output_path = f"{self.save_dir}{os.sep}_results_qlearning_.csv"

        # for k, v in x.items():
        #     self.logger.info(f"{str(k)} has len {len(v)}")

        # self.logger.info(f"Task complete {str(x)}")

        pd.DataFrame(x).to_csv(output_path, mode='a',
                               header=not os.path.exists(output_path), sep=";")
        self.message("saving learner")

        self.learner.save(str(self.actions_handled))

        self.requirements_record = []
        self.applied_adaptations = []

    def egreedy(self, symbolic_state: SymbolicState, q_learning=False) -> int:
        # e-greedy
        state = symbolic_state.state_name
        choice = self.rng.random()
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

    def process_goal_result(self, goal_status: int):
        if (goal_status == GoalStatus.STATUS_ABORTED):
            self.failed_goals += 1
        new_state = self.task_state(self.current_global_state)
        prev_state = self.task_state(self.previous_global_state)
        s_index = self.state_to_index[prev_state.state_name]
        s_prime_index = self.state_to_index[new_state.state_name]
        action_index = self.all_actions.index(self.selected_rule)
        self.learner.update_q_table(s_index, action_index, s_prime_index, -1.0)
