from typing import List
from typing import Optional

from adapt_x.adaptation_strategies.adaptation_strategy import AdaptationStrategy
from adapt_x_msgs.srv import FetchRules
from adapt_x_msgs.srv import SatisfiedRule
from adapt_x_msgs.srv._fetch_rules import FetchRules_Response


class TacticStrategy(AdaptationStrategy):

    def __init__(self, kb_node, system_name, interface_name, logger):
        self.kb_node = kb_node
        self.interface_name = interface_name
        self.system_name = system_name
        self.rule_dict = {}
        self.resolutions = {}
        self.logger = logger
        self.satisfied_rules = []
        self.all_rules = []

    def get_monitor_service(self) -> tuple[object, str]:
        return SatisfiedRule, f'/{self.kb_node}/satisfied_tactic'

    def get_monitor_request(self) -> object:
        return SatisfiedRule.Request(tactic_rules=self.all_rules)

    def get_initialization_service(self) -> tuple[object, str]:
        return FetchRules, f'/{self.kb_node}/fetch_rules'

    def get_initialization_request(self) -> object:
        return FetchRules.Request(
            resolution_model=self.system_name,
            interface_name=self.interface_name
        )

    def on_initialization(self, result: FetchRules_Response):
        rules = result.tactic_rules
        for rule in rules:
            self.all_rules.append(rule.name)
            self.rule_dict[rule.name] = rule.reconfigurations

    def on_monitored(self, response):
        satisfied = response.satisfied
        if (not any(satisfied)):
            return

        self.satisfied_rules = []
        for i, is_satisfied in enumerate(satisfied):
            if is_satisfied:
                self.satisfied_rules.append(self.all_rules[i])

    def on_adaptation_loop(self) -> Optional[List[str]]:
        if (len(self.satisfied_rules) > 0):
            # Here you can use some other heuristic, but we just do the first satisfied rule
            chosen_rule = self.satisfied_rules[0]
            self.logger.info(f'Rule with name {chosen_rule} will be applied')
            return self.rule_dict[chosen_rule]

    def on_requirements(self):
        pass

    def on_task_context(self):
        pass

    def on_task_complete(self):
        pass

    def on_receive_task(self):
        pass
