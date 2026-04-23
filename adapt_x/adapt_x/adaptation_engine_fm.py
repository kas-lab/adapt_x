#!/usr/bin/env python3
import os
import sys
import rclpy
import json
from rclpy.lifecycle import Node
from rclpy.lifecycle import State
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.srv import GetParameters
from ros_typedb_msgs.srv import Query
from ros_typedb_msgs.msg import QueryResult, IndexList, ResultTree
from std_srvs.srv import Trigger
from std_msgs.msg import String as std_String
from aal_msgs.srv import AdaptArchitectureTactical, AdaptArchitecture
from adapt_x_msgs.srv import FetchApplicable, UpdateRunning, FetchAdaptations, SatisfiedTactic, UpdateHasParamValue, HypotheticalReqFulfillment, RequirementsFulfillment, HypotheticalReqsFulfillment
from aal_msgs.msg import Adaptation
from typing import List
from datetime import datetime
from rclpy.parameter import parameter_value_to_python, Parameter
import pandas as pd
import functools
import time
import jpype
import jpype.imports
from rclpy.executors import MultiThreadedExecutor
from statistics import mean

jars_path = os.path.join(get_package_share_directory("adapt_x"), "jars", "fmlearning")

jpype.startJVM(
    classpath=[jars_path + "/*"]
)

from java.lang import String, Integer
from icsoc20 import FMExploration


class AdaptationEngine(Node):

    def __init__(self):
        super().__init__("adaptation_engine")

        self.get_logger().info("Adaptation Engine initialized and ready.")
        x = FMExploration(0.0, None, None, 2, None, False, "", "")


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
