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
#include "adapt_x_bt/ensure_configuration.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace adapt_x_bt
{

EnsureConfiguration::EnsureConfiguration(
  const std::string & name,
  const BT::NodeConfig & config)
: BT::DecoratorNode(name, config)
{
  node_ = rclcpp::Node::make_shared("ensure_cfg_" + name);
  callback_group_ = node_->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  executor_.add_callback_group(
    callback_group_, node_->get_node_base_interface());
}

BT::PortsList EnsureConfiguration::providedPorts()
{
  return {
    BT::InputPort<std::string>(
      FETCH_PLAN_SERVICE_PORT,
      "/fetch_plan_from_desired_features",
      "FetchPlanFromDesiredFeatures service name"),
    BT::InputPort<std::string>(
      ADAPT_SERVICE_PORT,
      "/adapt_architecture",
      "AdaptArchitecture service name"),
    BT::InputPort<std::string>(
      UPDATE_FEATURES_SERVICE_PORT,
      "/update_features_selected",
      "UpdateFeaturesSelected service name"),
    BT::InputPort<std::string>(
      RESOLUTION_MODEL_PORT,
      "",
      "Resolution model to scope the reconfiguration plan lookup to"),
    BT::InputPort<std::vector<std::string>>(
      ACTIVATE_PORT,
      "Features that must be active before the child runs"),
    BT::InputPort<std::vector<std::string>>(
      DEACTIVATE_PORT,
      std::vector<std::string>{},
      "Features that must be inactive before the child runs"),
    BT::InputPort<std::vector<std::string>>(
      ON_EXIT_PORT,
      std::vector<std::string>{},
      "Teardown override. Empty=deactivate features_to_activate. [\"none\"]=skip."),
  };
}

void EnsureConfiguration::initClients(
  const std::string & fetch_plan_service_name,
  const std::string & adapt_service_name,
  const std::string & update_features_service_name)
{
  fetch_plan_client_ = node_->create_client<FetchPlanSrv>(
    fetch_plan_service_name, rmw_qos_profile_services_default, callback_group_);
  adapt_client_ = node_->create_client<AdaptSrv>(
    adapt_service_name, rmw_qos_profile_services_default, callback_group_);
  update_features_client_ = node_->create_client<UpdateFeaturesSrv>(
    update_features_service_name,
    rmw_qos_profile_services_default,
    callback_group_);

  if (!fetch_plan_client_->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "EnsureConfiguration: service '%s' not available after 5s",
      fetch_plan_service_name.c_str());
  }
  if (!adapt_client_->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "EnsureConfiguration: service '%s' not available after 5s",
      adapt_service_name.c_str());
  }
  if (!update_features_client_->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "EnsureConfiguration: service '%s' not available after 5s",
      update_features_service_name.c_str());
  }
}

void EnsureConfiguration::resetCycleState()
{
  fetch_result_received_ = false;
  fetch_result_success_ = false;
  fetched_plan_.clear();
  fetched_feature_changes_ = adapt_x_msgs::msg::DesiredFeatures{};
  adapt_result_received_ = false;
  adapt_result_success_ = false;
  update_result_received_ = false;
  update_result_success_ = false;
}

void EnsureConfiguration::sendFetchPlanAsync(
  const std::vector<std::string> & to_activate,
  const std::vector<std::string> & to_deactivate)
{
  resetCycleState();

  auto request = std::make_shared<FetchPlanSrv::Request>();
  request->desired_features.active_features = to_activate;
  request->desired_features.inactive_features = to_deactivate;
  request->resolution_model = resolution_model_;

  const auto generation = generation_;
  fetch_plan_client_->async_send_request(
    request,
    [this, generation](rclcpp::Client<FetchPlanSrv>::SharedFuture future) {
      if (generation != generation_) {
        return;  // response belongs to an already-abandoned cycle
      }
      auto result = future.get();
      fetch_result_success_ = result->success;
      fetched_plan_ = result->adaptation_plan;
      fetched_feature_changes_ = result->feature_changes;
      fetch_result_received_ = true;
      if (!fetch_result_success_) {
        RCLCPP_WARN(
          node_->get_logger(),
          "EnsureConfiguration: fetch_plan_from_desired_features reported failure");
      }
    });
}

void EnsureConfiguration::sendUpdateFeaturesAsync(
  const adapt_x_msgs::msg::DesiredFeatures & feature_changes)
{
  update_result_received_ = false;
  update_result_success_ = false;

  auto request = std::make_shared<UpdateFeaturesSrv::Request>();
  request->feature_changes = feature_changes;

  const auto generation = generation_;
  update_features_client_->async_send_request(
    request,
    [this, generation](rclcpp::Client<UpdateFeaturesSrv>::SharedFuture future) {
      if (generation != generation_) {
        return;
      }
      auto result = future.get();
      update_result_success_ = result->success;
      update_result_received_ = true;
      if (!update_result_success_) {
        RCLCPP_WARN(
          node_->get_logger(),
          "EnsureConfiguration: update_features_selected failed: %s",
          result->message.c_str());
      }
    });
}

void EnsureConfiguration::sendAdaptAsync(
  const std::vector<aal_msgs::msg::Adaptation> & plan)
{
  adapt_result_received_ = false;
  adapt_result_success_ = false;

  auto request = std::make_shared<AdaptSrv::Request>();
  request->adaptations = plan;

  const auto generation = generation_;
  adapt_client_->async_send_request(
    request,
    [this, generation](rclcpp::Client<AdaptSrv>::SharedFuture future) {
      if (generation != generation_) {
        return;  // response belongs to an already-abandoned cycle
      }
      auto result = future.get();
      adapt_result_success_ = result->success;
      adapt_result_received_ = true;
      if (!adapt_result_success_) {
        RCLCPP_WARN(
          node_->get_logger(),
          "EnsureConfiguration: adapt_architecture reported failure");
      }
    });
}

std::vector<std::string> EnsureConfiguration::computeTeardownList()
{
  std::vector<std::string> on_exit;
  getInput(ON_EXIT_PORT, on_exit);

  if (!on_exit.empty()) {
    if (on_exit.size() == 1 && on_exit[0] == TEARDOWN_NONE) {
      return {};
    }
    return on_exit;
  }
  return activated_features_;
}

BT::NodeStatus EnsureConfiguration::tick()
{
  // ── Idle: first tick ────────────────────────────────────────────────────
  if (status() == BT::NodeStatus::IDLE) {
    std::string fetch_plan_service_name;
    getInput(FETCH_PLAN_SERVICE_PORT, fetch_plan_service_name);

    std::string adapt_service_name;
    getInput(ADAPT_SERVICE_PORT, adapt_service_name);

    std::string update_features_service_name;
    getInput(UPDATE_FEATURES_SERVICE_PORT, update_features_service_name);

    getInput(RESOLUTION_MODEL_PORT, resolution_model_);

    std::vector<std::string> to_activate;
    getInput(ACTIVATE_PORT, to_activate);

    std::vector<std::string> to_deactivate;
    getInput(DEACTIVATE_PORT, to_deactivate);

    activated_features_ = to_activate;

    initClients(
      fetch_plan_service_name,
      adapt_service_name,
      update_features_service_name);
    ++generation_;
    sendFetchPlanAsync(to_activate, to_deactivate);
    phase_ = Phase::FetchingPlan;
    setStatus(BT::NodeStatus::RUNNING);
    return BT::NodeStatus::RUNNING;
  }

  // ── FetchingPlan: waiting for fetch_plan_from_desired_features ──────────
  if (phase_ == Phase::FetchingPlan) {
    executor_.spin_some();
    if (!fetch_result_received_) {
      return BT::NodeStatus::RUNNING;
    }
    if (!fetch_result_success_) {
      phase_ = Phase::Idle;
      return BT::NodeStatus::FAILURE;
    }
    sendAdaptAsync(fetched_plan_);
    phase_ = Phase::Adapting;
    return BT::NodeStatus::RUNNING;
  }

  // ── Adapting: waiting for adapt_architecture ────────────────────────────
  if (phase_ == Phase::Adapting) {
    executor_.spin_some();
    if (!adapt_result_received_) {
      return BT::NodeStatus::RUNNING;
    }
    if (!adapt_result_success_) {
      phase_ = Phase::Idle;
      return BT::NodeStatus::FAILURE;
    }
    sendUpdateFeaturesAsync(fetched_feature_changes_);
    phase_ = Phase::UpdatingFeatures;
    return BT::NodeStatus::RUNNING;
  }

  // ── UpdatingFeatures: persist setup only after adaptation succeeds ───────
  if (phase_ == Phase::UpdatingFeatures) {
    executor_.spin_some();
    if (!update_result_received_) {
      return BT::NodeStatus::RUNNING;
    }
    if (!update_result_success_) {
      phase_ = Phase::Idle;
      return BT::NodeStatus::FAILURE;
    }
    phase_ = Phase::RunningChild;
  }

  // ── RunningChild ─────────────────────────────────────────────────────────
  if (phase_ == Phase::RunningChild) {
    BT::NodeStatus child_status = child_node_->executeTick();

    if (child_status == BT::NodeStatus::RUNNING) {
      return BT::NodeStatus::RUNNING;
    }

    // Child finished (SUCCESS or FAILURE)
    child_result_ = child_status;
    resetChild();

    auto teardown = computeTeardownList();
    if (teardown.empty()) {
      phase_ = Phase::Idle;
      return child_result_;
    }

    ++generation_;
    sendFetchPlanAsync({}, teardown);
    phase_ = Phase::TearingDownFetch;
    return BT::NodeStatus::RUNNING;
  }

  // ── TearingDownFetch ─────────────────────────────────────────────────────
  if (phase_ == Phase::TearingDownFetch) {
    executor_.spin_some();
    if (!fetch_result_received_) {
      return BT::NodeStatus::RUNNING;
    }
    if (!fetch_result_success_) {
      // Teardown failure never overrides the child's result.
      phase_ = Phase::Idle;
      return child_result_;
    }
    sendAdaptAsync(fetched_plan_);
    phase_ = Phase::TearingDownAdapt;
    return BT::NodeStatus::RUNNING;
  }

  // ── TearingDownAdapt ─────────────────────────────────────────────────────
  if (phase_ == Phase::TearingDownAdapt) {
    executor_.spin_some();
    if (!adapt_result_received_) {
      return BT::NodeStatus::RUNNING;
    }
    if (!adapt_result_success_) {
      phase_ = Phase::Idle;
      return child_result_;
    }
    sendUpdateFeaturesAsync(fetched_feature_changes_);
    phase_ = Phase::TearingDownUpdate;
    return BT::NodeStatus::RUNNING;
  }

  // ── TearingDownUpdate ───────────────────────────────────────────────────
  if (phase_ == Phase::TearingDownUpdate) {
    executor_.spin_some();
    if (!update_result_received_) {
      return BT::NodeStatus::RUNNING;
    }
    phase_ = Phase::Idle;
    return child_result_;
  }

  // Should never reach here
  return BT::NodeStatus::FAILURE;
}

void EnsureConfiguration::runTeardownSynchronously(
  const std::vector<std::string> & teardown)
{
  if (!fetch_plan_client_ || !adapt_client_ || !update_features_client_) {
    return;
  }

  ++generation_;  // invalidate any tick()-driven callback still in flight

  auto fetch_request = std::make_shared<FetchPlanSrv::Request>();
  fetch_request->desired_features.inactive_features = teardown;
  fetch_request->resolution_model = resolution_model_;

  auto fetch_future = fetch_plan_client_->async_send_request(fetch_request);

  auto deadline = std::chrono::steady_clock::now() + HALT_SERVICE_TIMEOUT;
  while (std::chrono::steady_clock::now() < deadline) {
    executor_.spin_some();
    if (fetch_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (fetch_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
    RCLCPP_WARN(
      node_->get_logger(),
      "EnsureConfiguration: halt() teardown fetch_plan_from_desired_features timed out; "
      "adapt_architecture will not be called");
    return;
  }

  auto fetch_result = fetch_future.get();
  if (!fetch_result->success) {
    RCLCPP_WARN(
      node_->get_logger(),
      "EnsureConfiguration: halt() teardown fetch_plan_from_desired_features reported failure");
    return;
  }

  auto adapt_request = std::make_shared<AdaptSrv::Request>();
  adapt_request->adaptations = fetch_result->adaptation_plan;

  auto adapt_future = adapt_client_->async_send_request(adapt_request);

  deadline = std::chrono::steady_clock::now() + HALT_SERVICE_TIMEOUT;
  while (std::chrono::steady_clock::now() < deadline) {
    executor_.spin_some();
    if (adapt_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (adapt_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
    RCLCPP_WARN(
      node_->get_logger(),
      "EnsureConfiguration: halt() teardown adapt_architecture timed out");
    return;
  }

  if (!adapt_future.get()->success) {
    RCLCPP_WARN(
      node_->get_logger(),
      "EnsureConfiguration: halt() teardown adapt_architecture reported failure");
    return;
  }

  auto update_request = std::make_shared<UpdateFeaturesSrv::Request>();
  update_request->feature_changes = fetch_result->feature_changes;
  auto update_future = update_features_client_->async_send_request(update_request);

  deadline = std::chrono::steady_clock::now() + HALT_SERVICE_TIMEOUT;
  while (std::chrono::steady_clock::now() < deadline) {
    executor_.spin_some();
    if (update_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (update_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
    RCLCPP_WARN(
      node_->get_logger(),
      "EnsureConfiguration: halt() teardown update_features_selected timed out");
    return;
  }

  const auto update_result = update_future.get();
  if (!update_result->success) {
    RCLCPP_WARN(
      node_->get_logger(),
      "EnsureConfiguration: halt() teardown update_features_selected failed: %s",
      update_result->message.c_str());
  }
}

void EnsureConfiguration::halt()
{
  auto teardown = computeTeardownList();

  resetChild();
  phase_ = Phase::Idle;
  activated_features_.clear();

  if (!teardown.empty()) {
    runTeardownSynchronously(teardown);
  }
}

}  // namespace adapt_x_bt
