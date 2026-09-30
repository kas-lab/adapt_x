from adapt_x.adaptation_strategies.adaptation_strategy import AdaptationStrategy
from adapt_x_msgs.srv import SatisfiedRule
from adapt_x_msgs.srv import FetchRules
from adapt_x_msgs.srv import FetchPlan
from adapt_x_msgs.srv._fetch_rules import FetchRules_Response
from typing import Optional


class TacticStrategy(AdaptationStrategy):

    def __init__(self, kb_node_name, system_name, interface_name, interface_type, parent_node):
        self.kb_node = kb_node_name
        self.interface_name = interface_name
        self.interface_type = interface_type
        self.system_name = system_name
        self.rule_dict = {}
        self.req_rule_dict = {}
        self.resolutions = {}
        self.logger = parent_node.get_logger()
        self.satisfied_rules = []
        self.all_rules = []
        self.violated_requirement = None

    def get_monitor_service(self) -> tuple[object, str]:
        return SatisfiedRule, f"/{self.kb_node}/satisfied_rule"

    def get_monitor_request(self) -> object:
        return SatisfiedRule.Request(tactic_rules=self.all_rules, action_name=self.interface_type)

    def get_initialization_service(self) -> tuple[object, str]:
        return FetchRules, f"/{self.kb_node}/fetch_rules"

    def get_initialization_request(self) -> object:
        return FetchRules.Request(resolution_model=self.system_name,
                                  interface_name=self.interface_type)

    def get_plan_request(self, feature_adaptation, action_name):
        return FetchPlan.Request(full_configuration=False,
                                 reconfiguration_string=feature_adaptation,
                                 action_name=action_name)

    def on_initialization(self, result: FetchRules_Response):
        rules = result.tactic_rules
        for rule in rules:
            self.all_rules.append(rule.name)
            self.rule_dict[rule.name] = rule.reconfigurations

            if (rule.requirement_name != "None"):
                if (rule.requirement_name in self.req_rule_dict):
                    self.req_rule_dict[rule.requirement_name].append(rule.name)
                else:
                    self.req_rule_dict[rule.requirement_name] = [rule.name]

        self.logger.info(f"all rules {str(self.all_rules)}")

    def on_monitored(self, response):
        satisfied = response.satisfied
        if (not any(satisfied)):
            self.logger.info(f"No applicable rules found for interface {self.interface_name} that satisfy their guard.")
            return True

        self.satisfied_rules = []
        for i, is_satisfied in enumerate(satisfied):
            current_rule = self.all_rules[i]
            if is_satisfied:
                if current_rule in set().union(*self.req_rule_dict.values()):
                    # For a requirement bearing satisfied rule to apply that req must be violated
                    if (self.violated_requirement is not None):
                        if current_rule in self.req_rule_dict.get(self.violated_requirement, []):
                            self.satisfied_rules.append(current_rule)
                else:
                    self.satisfied_rules.append(current_rule)

        return self.satisfied_rules == []

    def on_adaptation_loop(self) -> Optional[str]:
        if (len(self.satisfied_rules) > 0):
            if (self.violated_requirement is not None):
                related_rules = self.req_rule_dict.get(self.violated_requirement, [])
                self.violated_requirement = None
                for rule in related_rules:
                    if rule in self.satisfied_rules:
                        # Here you can use some other heuristic, but we just do the first satisfied rule
                        self.logger.info(f"Rule with name {rule} will be applied")
                        return self.rule_dict[rule]
            else:
                return self.rule_dict[self.satisfied_rules[0]]

    def on_requirements(self, reqs_msg):
        # TODO: what about multiple violations?
        for requirement in reqs_msg.requirements:
            self.logger.info(f"{requirement.name}: {requirement.fulfillment}")
            if (requirement.name in self.req_rule_dict):
                if (requirement.fulfillment == 0.0):
                    self.violated_requirement = requirement.name
                    return False
                    # violation

    def on_task_context(self, context):
        pass

    def on_task_complete(self):
        pass

    def on_receive_task(self):
        pass

    def process_goal_result(self, goal_status):
        pass
