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
import numpy as np
import pandas as pd
import time
from typing import Optional
from action_msgs.msg import GoalStatus

PURP_BOLD = "\033[35;1m"
RESET = "\033[0m"


class ConstantStrategy(AdaptationStrategy):
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

        self.mean_task_context = {}
        self.applied_adaptations = []
        self.selected_rule = None
        self.failed_goals = 0
        self.all_actions = []

        self.save_dir = f"{self.interface_name}_constant_{self.initial_time}"
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

    def on_monitored(self, response: GetState_Response):
        self.previous_global_state = self.current_global_state
        self.current_global_state = response.current_state

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
            self.selected_rule = self.all_actions[0]
            return self.selected_rule

    def on_initialization(self, result):
        self.all_actions = result.reconfigurations

    def on_task_context(self, task_context):
        return super().on_task_context(task_context)

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

        output_path = f"{self.save_dir}{os.sep}_results_constant_.csv"

        pd.DataFrame(x).to_csv(output_path, mode='a',
                               header=not os.path.exists(output_path), sep=";")

        self.requirements_record = []
        self.applied_adaptations = []

    def process_goal_result(self, goal_status: int):
        if (goal_status == GoalStatus.STATUS_ABORTED):
            self.failed_goals += 1
