#!/usr/bin/env python3
import os
import rclpy
import sys
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from ament_index_python.packages import get_package_share_directory
from aal_msgs.srv import AdaptArchitectureTactical, AdaptArchitecture
from aal_msgs.msg import Adaptation
from typing import List
from rclpy.parameter import Parameter
import pandas as pd
import jpype
import jpype.imports
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
import time
from adapt_x.adaptation_engine_common import client_request_init
from adapt_x.adaptation_engine_common import mean_fulfillment
from adapt_x.adaptation_engine_common import request_measures
from adapt_x_msgs.srv import UpdateInEffect
from adapt_x_msgs.msg import RelaxedRequirements


fm_jars_path = os.path.join(get_package_share_directory("adapt_x"), "jars", "fmlearning")
rem_jars_path = os.path.join(get_package_share_directory("adapt_x"), "jars", "rem")

jpype.startJVM(
    classpath=[rem_jars_path + "/*"] + [fm_jars_path + "/*"]
)

from java.lang import Integer
from java.nio.file import Files
from java.io import FileWriter
from icsoc20 import FMExploration, IExplorationPolicy, QLearning
from java.util import HashSet, HashMap, ArrayList
from de.ovgu.featureide.fm.core.io.manager import FeatureModelManager
from de.ovgu.featureide.fm.core.configuration import ConfigurationAnalyzer
from de.ovgu.featureide.fm.core.analysis.cnf.formula import FeatureModelFormula
from de.ovgu.featureide.fm.core.configuration import Configuration
from de.ovgu.featureide.fm.core.init import LibraryManager
from de.ovgu.featureide.fm.core.init import FMCoreLibrary

from eu.coresense.variability import FeatureModelStandaloneSetupGenerated
from eu.coresense.variability.generator import FeatureModelGenerator
from eu.coresense.resolution import ResolutionModelStandaloneSetupGenerated, Resolver
from eu.coresense.resolution.resolutionModel import ResolutionModel, SetParam, SetLifecycle
from eu.coresense.common import NameProvider, Utilities
from eu.coresense.context import ContextModelStandaloneSetupGenerated
from eu.coresense.context.generator import ContextNameProvider, ContextModelGenerator
from eu.coresense.context.contextModel import ContextModel
from eu.coresense.common import RosNameProvider, ArchitectureTQLGenerator
from eu.coresense.requirements.generator import RequirementsModelGenerator
from eu.coresense.requirements import RequirementsModelStandaloneSetupGenerated
from eu.coresense.requirements.requirementsModel import RequirementsModel

from ros import ParameterBoolean, ParameterInteger, ParameterDouble, ParameterString, ParameterSequence
from system import System as RosSystem
from de.fraunhofer.ipa.ros import RosStandaloneSetupGenerated
from de.fraunhofer.ipa.rossystem import RosSystemStandaloneSetupGenerated
from de.fraunhofer.ipa.ros2 import Ros2StandaloneSetupGenerated
from rosToolingSupport import RosToolingSupport
from org.eclipse.xtext.resource import XtextResourceSet
from uma.caosd.rhea.BasicFMmetamodel import FeatureModel
from uma.caosd.rhea.AdvancedConstraintsFMmetamodel import AdvancedConstraintsFMmetamodelPackage
from uma.caosd.rhea.BasicFMmetamodel import BasicFMmetamodelPackage


def valueOf(param):
    if (isinstance(param, ParameterBoolean)):
        return bool(((ParameterBoolean) @ param).isValue())
    elif (isinstance(param, ParameterInteger)):
        return int(((ParameterInteger) @ param).getValue())
    elif (isinstance(param, ParameterDouble)):
        return float(((ParameterDouble) @ param).getValue())
    elif (isinstance(param, ParameterString)):
        return str(((ParameterString) @ param).getValue())
    elif (isinstance(param, ParameterSequence)):
        return list(((ParameterSequence) @ param).getValue().stream().map(valueOf).toList())


def reconf_to_adapt(reconfiguration):
    # Convert java reconf object to AAL adaptation msg
    node_name = str(reconfiguration.getNode().getName())
    if (isinstance(reconfiguration, SetParam)):
        setparam = (SetParam) @ reconfiguration
        param_value = valueOf(setparam.getValue())
        param = Parameter(str(setparam.getParam().getName()), value=param_value)
        return Adaptation(adaptation_target=Adaptation.ROSPARAMETER,
                          node_name=node_name,
                          parameter_adaptation=param.to_parameter_msg())

    if (isinstance(reconfiguration, SetLifecycle)):
        pass


class AdaptationEngine(Node):

    def __init__(self, resolution_path: str = "",
                 var_path: str = "",
                 context_path: str = "",
                 reqs_path: str = "",
                 ros_path: List[str] = [""],
                 ros2_path: List[str] = [""],
                 rossystem_path: List[str] = [""],
                 num_rounds: int = 0):
        super().__init__("adaptation_engine")

        self.declare_parameter("resolution_path", resolution_path)
        self.declare_parameter("var_path", var_path)
        self.declare_parameter("ros_path", ros_path)
        self.declare_parameter("ros2_path", ros2_path)
        self.declare_parameter("rossystem_path", rossystem_path)
        self.declare_parameter("context_path", context_path)
        self.declare_parameter("reqs_path", reqs_path)
        self.declare_parameter("num_rounds", num_rounds)

        self.num_rounds = self.get_parameter("num_rounds").value
        self.kb_node = "tactical_retreat_kb_node"

        self.parallel_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self.create_subscription(RelaxedRequirements, "/requirements",
                                 self.requirements_callback,
                                 10, callback_group=self.parallel_cb_group)

        self.aal_client = self.create_client(AdaptArchitecture,
                                             "/adapt_architecture",
                                             callback_group=self.parallel_cb_group)
        self.XtextStuff()
        self.typedb_inserts()

        self.results_dict = {
            "QResults": [-1] * self.num_rounds,
            "SResults": [-1] * self.num_rounds,
            "results": [-1] * self.num_rounds}

        LibraryManager.registerLibrary(FMCoreLibrary.getInstance())
        confAction = HashMap()
        actionConf = HashMap()

        path = Files.createTempFile("featuremodel", ".xml")
        fw = FileWriter(path.toFile(), True)
        fw.write(self.fm_xml)
        fw.close()
        featureModel = FeatureModelManager.load(path)

        formula = FeatureModelFormula(featureModel)
        configuration = Configuration(formula)
        configurationAnalyzer = ConfigurationAnalyzer(formula, configuration)
        nbConfigurations = configurationAnalyzer.number(100000)

        allConfigurations = configurationAnalyzer.getSolutions(100000)
        self.all_configurations = []
        for k, featureList in enumerate(allConfigurations):
            self.all_configurations.append([str(feature) for feature in featureList])
            confAction.put(featureList, k)
            actionConf.put(k, featureList)
            # print(x)
        self.get_logger().info(f"num configurations {str(nbConfigurations)}")
        self.maxAction = Integer(nbConfigurations)

        self.exploration = FMExploration(0.0, confAction, actionConf, 2,
                                         self.maxAction, False, path, "")
        self.initialReward = 0  # lowest possible
        self.learningRate = 0.97  # TODO
        self.discountFactor = 0.31  # TODO
        self.epsilonDecayRate = 0.994  # TODO
        self.deltaDecayRate = 0.99
        self.avg = 5  # number of rewards to be averaged (to smooth the curves a little)
        self.setup_learning()
        self.time_step = 0

        current_configuration = set([str(f)
                                    for f in self.get_current_selected(
                                    self.DSLFeatureModel.getFm())])
        for i in range(nbConfigurations):
            other_configuration = set([str(f) for f in allConfigurations.get(i)])

            if current_configuration == other_configuration:
                self.get_logger().info(f"Match! index is:  {str(i)}")
                self.current_configuration = other_configuration
                self.selected_action = i

        self.resolver = Resolver(self.resolution_model)

        self.get_logger().info("Adaptation Engine initialized and ready.")

    def XtextStuff(self):
        RosToolingSupport.registerEPackages()
        AdvancedConstraintsFMmetamodelPackage.eINSTANCE.eClass()
        BasicFMmetamodelPackage.eINSTANCE.eClass()
        self.resSet = XtextResourceSet()

        RosToolingSupport.loadModel(
            RosStandaloneSetupGenerated(),
            self.resSet,
            "ros",
            self.get_parameter("ros_path").value,
        )

        self.ros2Resources = RosToolingSupport.loadModel(
            Ros2StandaloneSetupGenerated(),
            self.resSet,
            "ros2",
            self.get_parameter("ros2_path").value,
        )

        self.rosSystemResources = RosToolingSupport.loadModel(
            RosSystemStandaloneSetupGenerated(),
            self.resSet,
            "rossystem",
            self.get_parameter("rossystem_path").value,
        )

        fmResources = RosToolingSupport.loadModel(
            FeatureModelStandaloneSetupGenerated(),
            self.resSet,
            "var",
            self.get_parameter("var_path").value,
        )

        contextResources = RosToolingSupport.loadModel(
            ContextModelStandaloneSetupGenerated(),
            self.resSet,
            "context",
            self.get_parameter("context_path").value,
        )

        requirementsResources = RosToolingSupport.loadModel(
            RequirementsModelStandaloneSetupGenerated(),
            self.resSet,
            "reqs",
            self.get_parameter("reqs_path").value,
        )

        requirementsModelResource = requirementsResources.get(0)
        self.requirementsModel = (RequirementsModel) @ \
            requirementsModelResource.getContents().get(0)

        self.contextModelResource = contextResources.get(0)
        self.context_model = (ContextModel) @ self.contextModelResource.getContents().get(0)
        # fmResources is usually a java.util.ArrayList of Xtext Resource objects.
        # Get the first resource, resolve proxies, then extract the root FeatureModel.
        if hasattr(fmResources, "get"):
            resource0 = fmResources.get(0)
        else:
            resource0 = fmResources[0]

        try:
            from org.eclipse.emf.ecore.util import EcoreUtil
            EcoreUtil.resolveAll(resource0)
        except Exception:
            pass

        try:
            root_obj = resource0.getContents().get(0)
        except Exception:
            root_obj = resource0.contents().get(0)

        try:
            from org.eclipse.emf.ecore.resource import Resource as EMFResource
            if isinstance(root_obj, EMFResource):
                root_obj = root_obj.getContents().get(0)
        except Exception:
            pass

        self.DSLFeatureModel = root_obj

        generator = FeatureModelGenerator()

        some_xml = generator.generateXml(self.DSLFeatureModel.getFm())

        self.fm_xml = some_xml

        resolutionResources = RosToolingSupport.loadModel(
            ResolutionModelStandaloneSetupGenerated(),
            self.resSet,
            "resolution",
            self.get_parameter("resolution_path").value,
        )

        self.resolutionModelResource = resolutionResources.get(0)
        self.resolution_model = (ResolutionModel) @ self.resolutionModelResource.getContents().get(0)
        # for i in range(nmbRuns):
        #     epINE = FMExploration(0.0, confAction, actionConf, 2, maxAction, False, fm_path,"");			
        #     learn(epINE, resultsINE[k], QoS1resultsINE[k], QoS2resultsINE[k], evolution, deltaDecayRateInc);

    def typedb_inserts(self):
        from ros_typedb_msgs.srv import Query
        typedb_client = self.create_client(Query, f"/{self.kb_node}/query")
        in_effect_client = self.create_client(UpdateInEffect, f"/{self.kb_node}/update_in_effect")

        namer = NameProvider()
        namer.register(RosNameProvider())
        namer.register(ContextNameProvider(namer))
        architecture_gen = ArchitectureTQLGenerator(namer)
        context_generator = ContextModelGenerator(namer)
        req_gen = RequirementsModelGenerator(namer)
        self.systemName = self.resolution_model.getName()

        rossystems = Utilities.deriveFromImports(self.contextModelResource,
                                                 '.rossystem',
                                                 self.context_model.getImports(),
                                                 RosSystem.class_)

        initial_insert_query = ""
        for rsys in rossystems:
            initial_insert_query += str(architecture_gen.generateArchitectureInsertQuery(rsys,
                                                                                         True))

        initial_insert_query += str(context_generator.registerMeasurements(self.context_model))
        # initial_insert_query += str(self.generator.generateTQL(self.tacticsModel,
        #                                                        self.tacticsModelResource,
        #                                                        self.get_parameter("policy_mode").value))
        initial_insert_query += str(req_gen.generateInsertQuery(self.requirementsModel,
                                                                self.resolution_model.getName(),
                                                                False))

        while not typedb_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().info("Waiting for TypeDB to be ready")

        initial_insert_res = client_request_init(self, typedb_client,
                                                 Query.Request(query=initial_insert_query,
                                                               query_type=Query.Request.INSERT),
                                                 "Initial Model Load")

        if initial_insert_res is None or not initial_insert_res.success:
            self.get_logger().error("Failed to insert initial model. Aborting")
            sys.exit(1)

        in_eff_res = client_request_init(self, in_effect_client,
                                         UpdateInEffect.Request(all_requirements=True,
                                                                requirement_names=["MoveSafely",
                                                                                   "MoveQuickly"]),
                                         "UpdateInEffect")

        if in_eff_res is None or not in_eff_res.success:
            self.get_logger().error("Failed to update ineffect reqs . Aborting")
            sys.exit(1)

        request_measures(self)

    def configuration_delta(self, new_configuration_i):
        new_configuration = set(self.all_configurations[new_configuration_i])

        deselection_set = self.current_configuration - new_configuration
        selection_set = new_configuration - self.current_configuration

        reconfigurations = []
        for feat in deselection_set:
            desels = self.resolver.deselectionActions(feat)
            if (not desels.isEmpty()):
                reconfigurations.extend(list(desels))

        for feat in selection_set:
            sels = self.resolver.selectionActions(feat)
            if (not sels.isEmpty()):
                reconfigurations.extend(list(sels))

        # This is an assumption that the adaptation is successful.
        self.current_configuration = new_configuration
        return [reconf_to_adapt(reconf) for reconf in reconfigurations]

    def report(self):
        # Make CSV and stuff
        res = pd.DataFrame(data=self.results_dict)
        res.to_csv(str(int(time.time())) + "_fmlearning_results.csv")

        sys.exit(0)

    def requirements_callback(self, reqs_msg):
        if (self.time_step == self.num_rounds):
            self.report()

        for req in reqs_msg.requirements:
            if req.name == "MoveQuickly":
                self.quick_req_ful = req.fulfillment
            if req.name == "MoveSafely":
                self.safe_req_ful = req.fulfillment
        self.reward = mean_fulfillment(reqs_msg)

    def collectAllFeatures(self, feature, allFeatures):
        if (feature is not None):
            allFeatures.add(feature)
            for child in feature.getChildren():
                self.collectAllFeatures(child, allFeatures)

    def get_current_selected(self, fm: FeatureModel):
        allFeatures = ArrayList()
        self.collectAllFeatures(fm.getRoot(), allFeatures)
        return [str(feat.getName()) for feat in allFeatures if feat.isSelected()]

    def setup_learning(self):
        # Init Rewards with lowest reward possible
        # If higher values, this means one implicitly defines exploration via
        # optimistic initial values and thus may overlay
        # with the effect of e-greedy or FM-guided learning
        self.ql = QLearning(1, self.maxAction, self.exploration, False, self.initialReward)

        # Hyper-parameters for Q-Learning
        self.ql.setDiscountFactor(self.discountFactor)
        self.ql.setLearningRate(self.learningRate)

        # For smoothing the curves a little
        self.avgreward = [-1] * self.avg
        self.totalreward = 0
        self.avgQoS1 = [0] * self.avg
        self.totalQoS1 = 0
        self.avgQoS2 = [-1] * self.avg
        self.totalQoS2 = 0.0

        self.ql.setActions(self.maxAction)
        self.exploration.setMaxAction(self.maxAction)
        self.exploration.setPreviousMaxAction(0)

        self.epsilon = 1.0
        self.delta = 1.0

    def next_action(self):
        # Ported from the original java.

        # for(int i = 0; i < maxEpisodes; i++) {

        # Implement epsilon decay
        self.exploration.setEpsilon(self.epsilon)
        self.epsilon = self.epsilon * self.epsilonDecayRate

        # Implement delta decay
        # (certain amount of randomness also for FM strategies at beginning)
        self.exploration.setDelta(self.delta)
        self.delta = self.delta * self.deltaDecayRate

        # Select action and get reward for that action
        # Q-Learning uses "new" action, while SARSA uses the action from the
        # previous knowledge update (if available)

        action = self.ql.GetAction(0)

        return action

    def process_reward(self, action_i):
        # this is all reporting, don't know if we should keep it
        # # compute rewards (smoothed via avg)
        self.avgreward[self.time_step % self.avg] = self.reward
        self.totalreward = 0
        for k in range(self.avg):
            self.totalreward += self.avgreward[k]
        self.results_dict["results"][self.time_step] = self.totalreward/float(self.avg)

        # compute QoS values
        self.avgQoS1[self.time_step % self.avg] = self.quick_req_ful
        self.totalQoS1 = 0
        for k in range(self.avg):
            self.totalQoS1 += self.avgQoS1[k]

        self.results_dict["QResults"][self.time_step] = self.totalQoS1 / self.avg

        self.avgQoS2[self.time_step % self.avg] = self.safe_req_ful
        self.totalQoS2 = 0
        for k in range(self.avg):
            self.totalQoS2 += self.avgQoS2[k]

        self.results_dict["Sresults"][self.time_step] = self.totalQoS2/self.avg

        # Knowledge Update (this is where Q-Learning and SARSA differ)
        # Replay the experience twice to speed up learning in general
        # Inspiried by replay buffer from DQN
        self.ql.UpdateState(0, action_i, self.reward, 0)
        self.ql.UpdateState(0, action_i, self.reward, 0)

        self.time_step = self.time_step + 1
        # }

    def on_configure(self, state: State):
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

        self.create_service(AdaptArchitectureTactical,
                            '/adapt_architecture_tactical', self.tactic_adaptation_callback)
        return TransitionCallbackReturn.SUCCESS

    def tactic_adaptation_callback(self, request, response):
        self.process_reward(self.selected_action)  # prev action

        self.selected_action = self.next_action()

        adaptations = self.configuration_delta(self.selected_action)

        self.request_adaptation(adaptations)

        response.success = True
        return response

    def request_adaptation(self, adaptations: List[Adaptation]):
        request = AdaptArchitecture.Request(adaptations=adaptations)

        if self.aal_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info("r")
            future = self.aal_client.call_async(request)
            future.add_done_callback(self.postprocess)
        else:
            self.get_logger().error("AAL adaptation service not available")
            raise RuntimeError("AAL adaptation service not available")

    def postprocess(self, future):
        # self.get_logger().info("Postprocessing adaptation results.")
        result = future.result()

        if (result is None or not result.success):
            self.get_logger().error("Adaptation failed, cannot postprocess.")
            return


def main():
    rclpy.init()

    action_client = AdaptationEngine()

    mt_executor = MultiThreadedExecutor()
    mt_executor.add_node(action_client)
    action_client.executor = mt_executor
    mt_executor.spin()

    action_client.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
