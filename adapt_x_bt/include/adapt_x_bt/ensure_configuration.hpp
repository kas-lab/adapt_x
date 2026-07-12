// Copyright 2026 KAS-lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef ADAPT_X_BT__ENSURE_CONFIGURATION_HPP_
#define ADAPT_X_BT__ENSURE_CONFIGURATION_HPP_

#include <chrono>
#include <string>
#include <vector>

#include "behaviortree_cpp/decorator_node.h"
#include "behaviortree_cpp/bt_factory.h"

#include "rclcpp/rclcpp.hpp"

#include "aal_msgs/msg/adaptation.hpp"
#include "aal_msgs/srv/adapt_architecture.hpp"
#include "adapt_x_msgs/srv/fetch_plan_from_desired_features.hpp"

namespace adapt_x_bt
{

/**
 * Decorator that ensures a desired feature configuration is active before
 * running its child, then tears it down on completion or preemption.
 *
 * For both setup and teardown, the decorator performs two sequential service
 * calls: it first asks the tactical knowledge base for a reconfiguration
 * plan (fetch_plan_service_name), then asks the adaptation manager to
 * execute that plan (adapt_service_name).
 *
 * Ports:
 *   fetch_plan_service_name     (string) — FetchPlanFromDesiredFeatures service name
 *   adapt_service_name          (string) — AdaptArchitecture service name
 *   resolution_model            (string) — resolution model to scope the plan lookup to
 *   features_to_activate           (vector<string>) — features that must be active before child
 *   features_to_deactivate         (vector<string>) — features that must be inactive before child
 *   features_to_deactivate_on_exit (vector<string>)
 *       empty    → deactivate whatever was in features_to_activate (default)
 *       ["none"] → skip teardown entirely
 *       other    → deactivate exactly these features
 */
class EnsureConfiguration : public BT::DecoratorNode
{
public:
  using FetchPlanSrv = adapt_x_msgs::srv::FetchPlanFromDesiredFeatures;
  using AdaptSrv = aal_msgs::srv::AdaptArchitecture;

  static constexpr const char * FETCH_PLAN_SERVICE_PORT = "fetch_plan_service_name";
  static constexpr const char * ADAPT_SERVICE_PORT = "adapt_service_name";
  static constexpr const char * RESOLUTION_MODEL_PORT = "resolution_model";
  static constexpr const char * ACTIVATE_PORT = "features_to_activate";
  static constexpr const char * DEACTIVATE_PORT = "features_to_deactivate";
  static constexpr const char * ON_EXIT_PORT = "features_to_deactivate_on_exit";
  static constexpr const char * TEARDOWN_NONE = "none";

  // Bounded wait applied to each of the two service calls fired from halt(),
  // since halt() cannot rely on the tree being ticked again to make progress.
  static constexpr std::chrono::milliseconds HALT_SERVICE_TIMEOUT{2000};

  EnsureConfiguration(const std::string & name, const BT::NodeConfig & config);
  ~EnsureConfiguration() override = default;

  static BT::PortsList providedPorts();

  BT::NodeStatus tick() override;
  void halt() override;

private:
  enum class Phase
  {
    Idle,
    FetchingPlan,
    Adapting,
    RunningChild,
    TearingDownFetch,
    TearingDownAdapt
  };

  void initClients(
    const std::string & fetch_plan_service_name,
    const std::string & adapt_service_name);
  void sendFetchPlanAsync(
    const std::vector<std::string> & to_activate,
    const std::vector<std::string> & to_deactivate);
  void sendAdaptAsync(const std::vector<aal_msgs::msg::Adaptation> & plan);
  std::vector<std::string> computeTeardownList();
  void resetCycleState();
  // Bounded, blocking fetch+adapt chain used from halt(), where the tree is
  // not guaranteed to be ticked again to drive the async state machine.
  void runTeardownSynchronously(const std::vector<std::string> & teardown);

  rclcpp::Node::SharedPtr node_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Client<FetchPlanSrv>::SharedPtr fetch_plan_client_;
  rclcpp::Client<AdaptSrv>::SharedPtr adapt_client_;

  Phase phase_{Phase::Idle};
  BT::NodeStatus child_result_{BT::NodeStatus::IDLE};

  // Identifies the current request cycle so that a response belonging to an
  // already-abandoned cycle (e.g. after halt() resets the phase) is ignored
  // instead of corrupting the state of a newer cycle.
  uint64_t generation_{0};

  bool fetch_result_received_{false};
  bool fetch_result_success_{false};
  std::vector<aal_msgs::msg::Adaptation> fetched_plan_;

  bool adapt_result_received_{false};
  bool adapt_result_success_{false};

  std::string resolution_model_;
  std::vector<std::string> activated_features_;
};

}  // namespace adapt_x_bt

#endif  // ADAPT_X_BT__ENSURE_CONFIGURATION_HPP_
