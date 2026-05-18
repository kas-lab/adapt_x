# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy

from std_msgs.msg import String
from ros_monitors_functions.factory import MonitorFactory
from ros_monitors_functions.global_costmap_functions import GlobalCostmapChange
from nav_msgs.msg import OccupancyGrid


def main(args=None):
    rclpy.init(args=args)

    monitors = MonitorFactory()
    monitors.register_monitor(
            'costmap_change', GlobalCostmapChange())
    node = rclpy.create_node('minimal_subscriber')

    subscription = node.create_subscription(
        OccupancyGrid, '/global_costmap/costmap',
        lambda msg: node.get_logger().info('Was there an update?: "%s"' % str(monitors['costmap_change'](msg))), 10)
    subscription  # prevent unused variable warning

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
