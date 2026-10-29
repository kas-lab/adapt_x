from adapt_x.adaptation_strategies.adaptation_strategy import AdaptationStrategy
from adapt_x_msgs.srv import ValidRankedTactics
from adapt_x_msgs.srv import FetchRules
from adapt_x_msgs.srv import FetchPlan
from adapt_x_msgs.srv import FetchSimilar
from adapt_x_msgs.srv._fetch_rules import FetchRules_Response
from adapt_x_msgs.srv._fetch_similar import FetchSimilar_Response
from typing import Optional
from std_msgs.msg import String


class CBRTacticStrategy(AdaptationStrategy):

    def __init__(self, kb_node_name, system_name, interface_name, interface_type, parent_node):
        self.kb_node = kb_node_name
        self.interface_name = interface_name
        self.interface_type = interface_type
        self.system_name = system_name
        self.rule_dict = {}
        self.req_rule_dict = {}
        self.resolutions = {}
        self.current_tactic_ranking = {}
        self.logger = parent_node.get_logger()
        self.satisfied_rules = []
        self.satisifed_requirement_rules = []
        self.all_rules = []
        self.violated_requirement = None
        self.violation_count = {}
        self.req_sim = {}
        self.req_sim_progress = {}
        self.violated_requirements_being_evaluated = []
        self.cbr_attempt = None
        self.eval_publisher = parent_node.create_publisher(String, '/cbr_evaluation', 10)

    def get_monitor_service(self) -> tuple[object, str]:
        return ValidRankedTactics, f"/{self.kb_node}/satisfied_rule"

    def get_monitor_request(self) -> object:
        return ValidRankedTactics.Request(tactic_rules=self.all_rules,
                                          action_name=self.interface_type)

    def get_initialization_service(self) -> list[tuple[object, str]]:
        return [(FetchRules, f"/{self.kb_node}/fetch_rules"),
                (FetchSimilar, f"/{self.kb_node}/fetch_similar")]

    def get_initialization_request(self) -> list[object]:
        return [FetchRules.Request(resolution_model=self.system_name,
                                  interface_name=self.interface_type),
                FetchSimilar.Request(interface_name=self.interface_type)]

    def get_plan_request(self, feature_adaptation, action_name):
        return FetchPlan.Request(full_configuration=False,
                                 reconfiguration_string=feature_adaptation,
                                 action_name=action_name)

    def _fetch_rules_init(self, result: FetchRules_Response):
        self.logger.info("Processed fetchrules result")

        rules = result.tactic_rules
        for rule in rules:
            self.all_rules.append(rule.name)
            self.rule_dict[rule.name] = rule.reconfigurations

            if (rule.requirement_name != "None"):
                if (rule.requirement_name in self.req_rule_dict):
                    self.req_rule_dict[rule.requirement_name].append(rule.name)
                else:
                    self.violation_count[rule.requirement_name] = 0
                    self.req_rule_dict[rule.requirement_name] = [rule.name]

        self.logger.info(f"all rules {str(self.all_rules)}")

    def _fetch_similar_init(self, result: FetchSimilar_Response):
        sim_pairs = result.similarity_pairs
        for sim_pair in sim_pairs:
            self.req_sim[sim_pair.requirement_name] = sim_pair.similar_items
            self.req_sim_progress[sim_pair.requirement_name] = 0

    def on_initialization(self, result):
        if isinstance(result, FetchRules_Response):
            self._fetch_rules_init(result)
        if isinstance(result, FetchSimilar_Response):
            self._fetch_similar_init(result)

    def on_monitored(self, response):
        satisfied = response.satisfied
        self.satisfied_rules = []
        self.satisifed_requirement_rules = []
        if not any(satisfied):
            self.logger.info(f"No applicable rules found for interface {self.interface_name} that satisfy their guard.")
            return True


        for i, is_satisfied in enumerate(satisfied):
            current_rule = self.all_rules[i]
            if is_satisfied:
                if current_rule in set().union(*self.req_rule_dict.values()):
                    # For a requirement bearing satisfied rule to apply that req must be violated
                    self.satisifed_requirement_rules.append(current_rule)
                else:
                    self.satisfied_rules.append(current_rule)

        ranked_tactics = response.ranked_tactics
        self.current_tactic_ranking = {}
        for sim_pair in ranked_tactics:
            self.current_tactic_ranking[sim_pair.requirement_name] = sim_pair.similar_items

        return self.satisfied_rules == []

    def on_adaptation_loop(self) -> Optional[str]:
        self.logger.info("0")
        all_satisfied_rules = self.satisfied_rules + self.satisifed_requirement_rules


            
        if (len(all_satisfied_rules) > 0):
            self.logger.info("1")
            if (self.violated_requirement is not None and len(self.satisifed_requirement_rules) > 0):
                self.violation_count[self.violated_requirement]+=1
                vio_count = self.violation_count[self.violated_requirement]
                related_rules = self.req_rule_dict.get(self.violated_requirement, [])
                self.logger.info(f"Related rules {str(related_rules)}")
                violated_requirement = self.violated_requirement
                self.violated_requirement = None
                rules_for_violated = [rule for rule in related_rules if rule in self.satisifed_requirement_rules]
                self.logger.info(f"Rules for violated {str(rules_for_violated)}")

                if(vio_count == 1 and len(rules_for_violated) > 0):
                    rule = rules_for_violated[0]
                    # Here you can use some other heuristic, but we just do the first satisfied rule
                    self.logger.info(f"Rule with name {rule} will be applied")
                    return self.rule_dict[rule]
                else:
                    reqs = self.req_sim[violated_requirement]
                    i = self.req_sim_progress[violated_requirement]
                    rules = [rule for req in reqs for rule in self.current_tactic_ranking[req]]
                    rules.extend(self.current_tactic_ranking["none"])
                    rule = rules[i]
                    self.req_sim_progress[violated_requirement]+=1
                    self.violated_requirements_being_evaluated.append(violated_requirement)
                    self.cbr_attempt = rule
                    self.logger.info(f"Rule with name {rule} will be applied as CBR")

                    return self.rule_dict[rule]
            elif len(self.satisfied_rules) > 0:
                return self.rule_dict[self.satisfied_rules[0]]
        else:
            self.logger.info("No Satisfied Rules")
            if self.violated_requirement is not None:
                violated_requirement = self.violated_requirement
                self.violated_requirement = None
                reqs = self.req_sim[violated_requirement]
                i = self.req_sim_progress[violated_requirement]
                rules = [rule for req in reqs for rule in self.current_tactic_ranking[req]]
                rules.extend(self.current_tactic_ranking["none"])
                rule = rules[i]
                self.req_sim_progress[violated_requirement]+=1
                self.violated_requirements_being_evaluated.append(violated_requirement)
                self.cbr_attempt = rule
                self.logger.info(f"Rule with name {rule} will be applied as CBR")

                return self.rule_dict[rule]

    def on_requirements(self, reqs_msg):
        # TODO: what about multiple violations?
        none_violated = True
        for requirement in reqs_msg.requirements:
            self.logger.info(f"{requirement.name}: {requirement.fulfillment}")
            if (requirement.name in self.req_rule_dict):
                if (requirement.fulfillment == 0.0):
                    self.violated_requirement = requirement.name
                    none_violated = False
                else:
                    self.violation_count[requirement.name] = 0
                    # violation
        
        if(none_violated and len(self.violated_requirements_being_evaluated) > 0):
            self.eval_publisher.publish(String(data=f"{str(self.violated_requirements_being_evaluated)} resolved by {self.cbr_attempt}"))
            self.cbr_attempt = None
            self.violated_requirements_being_evaluated = []



        return none_violated

    def on_task_context(self, context):
        pass

    def on_task_complete(self):
        pass

    def on_receive_task(self):
        pass

    def process_goal_result(self, goal_status):
        pass
