# SAS Module - Tactical Retreat - AdaptX
This houses the XText-based Adaptation Engine for the SAS module.
It houses .jars of the projects found in this [repository](https://github.com/CoreSenseEU/sas_module_metamodels)

Additionally, it facilitates the creation of ROS2 nodes which adapt other nodes, based on provided DSL entries. For example, for a Tiago robot [here](https://github.com/CoreSenseEU/rem_tiago).

The actual architecture of how we do so is changing right now, but it will probably consist of intercepting ROS2 goal requests, using this as a trigger to adapt nodes related to accomplishing the goal, and then passing along the goal to the intended recipient which will accomplish the goal.