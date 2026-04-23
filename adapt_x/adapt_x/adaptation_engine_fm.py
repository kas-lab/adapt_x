#!/usr/bin/env python3
import os
import rclpy
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from ament_index_python.packages import get_package_share_directory
from aal_msgs.srv import AdaptArchitectureTactical, AdaptArchitecture
from aal_msgs.msg import Adaptation
from typing import List
from datetime import datetime
from rclpy.parameter import parameter_value_to_python, Parameter
import pandas as pd
import jpype
import jpype.imports
from rclpy.executors import MultiThreadedExecutor
from statistics import mean
from rclpy.lifecycle import TransitionCallbackReturn


fm_jars_path = os.path.join(get_package_share_directory("adapt_x"), "jars", "fmlearning")
rem_jars_path = os.path.join(get_package_share_directory("adapt_x"), "jars", "rem")

fm_path = os.path.join(get_package_share_directory("adapt_x"), "config", "fmlearning", "TiagoFM.xml")

jpype.startJVM(
    classpath=[rem_jars_path + "/*"] + [fm_jars_path + "/*"]
)

from java.lang import String, Integer, Double
from java.nio.file import Paths
from icsoc20 import FMExploration, IExplorationPolicy, QLearning
from java.util import Map, HashMap
from de.ovgu.featureide.fm.core.io.manager import FeatureModelManager
from de.ovgu.featureide.fm.core.configuration import ConfigurationAnalyzer
from de.ovgu.featureide.fm.core.analysis.cnf.formula import FeatureModelFormula
from de.ovgu.featureide.fm.core.configuration import Configuration
from de.ovgu.featureide.fm.core.init import LibraryManager
from de.ovgu.featureide.fm.core.init import FMCoreLibrary

from eu.coresense.variability import FeatureModelStandaloneSetupGenerated
from eu.coresense.resolution import ResolutionModelStandaloneSetupGenerated
from de.fraunhofer.ipa.ros import RosStandaloneSetupGenerated
from de.fraunhofer.ipa.rossystem import RosSystemStandaloneSetupGenerated
from de.fraunhofer.ipa.ros2 import Ros2StandaloneSetupGenerated
from org.ros2.rcljava.rebet_java import RosToolingSupport
from org.eclipse.xtext.resource import XtextResourceSet

class AdaptationEngine(Node):

    def __init__(self):
        super().__init__("adaptation_engine")

        self.declare_parameter("resolution_path", "")
        self.declare_parameter("variability_path", "")
        self.declare_parameter("ros_path", "")
        self.declare_parameter("ros2_path", "")
        self.declare_parameter("rossystem_path", "")

        LibraryManager.registerLibrary(FMCoreLibrary.getInstance())
        confAction = HashMap()
        actionConf = HashMap()
        path = Paths.get(fm_path)
        featureModel = FeatureModelManager.load(path)

        formula = FeatureModelFormula(featureModel)
        configuration = Configuration(formula)
        configurationAnalyzer = ConfigurationAnalyzer(formula, configuration)
        nbConfigurations = configurationAnalyzer.number(100000)

        allConfigurations = configurationAnalyzer.getSolutions(100000)

        for featureList in allConfigurations:
            x = list(featureList)
            print(x)
        self.get_logger().info(f"num configurations {str(nbConfigurations)}")
        self.maxAction = Integer(nbConfigurations)

        self.exploration = FMExploration(0.0, confAction, actionConf, 2,
                                         self.maxAction, False, fm_path, "")
        self.get_logger().info("Adaptation Engine initialized and ready.")
        self.initialReward = -2
        self.learningRate = 0.0 #TODO
        self.discountFactor = 0.0#TODO
        self.epsilonDecayRate = 0.0 #TODO
        self.avg = 10  # number of rewards to be averaged (to smooth the curves a little)
        self.setup_learning()
        self.time_step = 0

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

        RosToolingSupport.loadModel(
            FeatureModelStandaloneSetupGenerated(),
            self.resSet,
            "var",
            self.get_parameter("variability_path").value,
        )

        RosToolingSupport.loadModel(
            ResolutionModelStandaloneSetupGenerated(),
            self.resSet,
            "resolution",
            self.get_parameter("resolution_path").value,
        )
        # for i in range(nmbRuns):
        #     epINE = FMExploration(0.0, confAction, actionConf, 2, maxAction, False, fm_path,"");			
        #     learn(epINE, resultsINE[k], QoS1resultsINE[k], QoS2resultsINE[k], evolution, deltaDecayRateInc);

    def setup_learning(self):
        # Init Rewards with lowest reward possible
        # If higher values, this means one implicitly defines exploration via optimistic initial values and
        # thus may overlay with the effect of e-greedy or FM-guided learning
        self.ql = QLearning(1, self.maxAction, self.exploration, False, self.initialReward)

        # Hyper-parameters for Q-Learning
        self.ql.setDiscountFactor(self.discountFactor)
        self.ql.setLearningRate(self.learningRate)

        # For smoothing the curves a little
        avgreward = [None]*self.avg
        for k in range(self.avg): 
            avgreward[k] = -1
        totalreward = 0
        # avgQoS1 = []
        # for k in range(self.avg):
        #     avgQoS1[k] = 0
        # totalQoS1 = 0
        # avgQoS2 = []
        # for k in range(self.avg):
        #     avgQoS2[k] = -1
        # totalQoS2 = 0.0

        self.ql.setActions(self.maxAction)
        self.exploration.setMaxAction(self.maxAction)
        self.exploration.setPreviousMaxAction(0)

        self.epsilon = 1.0
        self.delta = 1.0

    def next_action(self, results: list[float], Eresults: list[float],
              Mresults: list[float], deltaDecayRate: float):
        # Ported from the original java.

        # for(int i = 0; i < maxEpisodes; i++) {

        # Implement epsilon decay
        self.exploration.setEpsilon(self.epsilon)
        self.epsilon = self.epsilon * self.epsilonDecayRate

        # Implement delta decay
        # (certain amount of randomness also for FM strategies at beginning)
        self.exploration.setDelta(self.delta)
        self.delta = self.delta * deltaDecayRate

        # Select action and get reward for that action
        # Q-Learning uses "new" action, while SARSA uses the action from the
        # previous knowledge update (if available)

        action = self.ql.GetAction(0)

        return action

    def process_reward(self, action, reward):
        #this is all reporting, don't know if we should keep it
        # # compute rewards (smoothed via avg)
        self.avgreward[self.time_step % self.avg] = reward
        totalreward = 0
        for k in range(self.avg): totalreward += self.avgreward[k]
        self.results[self.time_step] = totalreward/float(self.avg)

        # compute QoS values
        # avgQoS1[self.time_step % avg] = QoS1[action];
        # totalQoS1 = 0;
        # for(int k = 0; k < avg; k++) totalQoS1 += avgQoS1[k];		
        # Eresults[self.time_step] = totalQoS1/(double)avg;

        # avgQoS2[i % avg] = QoS2[action];
        # totalQoS2 = 0;
        # for(int k = 0; k < avg; k++) totalQoS2 += avgQoS2[k];		
        # Mresults[self.time_step] = totalQoS2/(double)avg;

        # Knowledge Update (this is where Q-Learning and SARSA differ)
        # Replay the experience twice to speed up learning in general
        # Inspiried by replay buffer from DQN
        self.ql.UpdateState(0, action, reward, 0)
        self.ql.UpdateState(0, action, reward, 0)
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
        return TransitionCallbackReturn.SUCCESS



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
