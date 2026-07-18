//  Copyright 2026 KAS-lab
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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "behaviortree_cpp/bt_factory.h"
#include "rclcpp/rclcpp.hpp"

#include "adapt_x_bt/ensure_configuration.hpp"

// ─── Aliases ─────────────────────────────────────────────────────────────────

using FetchPlanSrv = adapt_x_bt::EnsureConfiguration::FetchPlanSrv;
using AdaptSrv = adapt_x_bt::EnsureConfiguration::AdaptSrv;
using UpdateFeaturesSrv = adapt_x_bt::EnsureConfiguration::UpdateFeaturesSrv;

namespace
{

std::vector<aal_msgs::msg::Adaptation> makeTestPlan(const std::string & node_name)
{
  aal_msgs::msg::Adaptation adaptation;
  adaptation.adaptation_target = aal_msgs::msg::Adaptation::DESIREDSTATE;
  adaptation.node_name = node_name;
  return {adaptation};
}

}  // namespace

// ─── A child node that never finishes on its own ─────────────────────────────

class AlwaysRunning : public BT::SyncActionNode
{
public:
  AlwaysRunning(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  static BT::PortsList providedPorts() {return {};}

  BT::NodeStatus tick() override {return BT::NodeStatus::RUNNING;}
};

// ─── Mock fetch_plan_from_desired_features server ────────────────────────────

class MockFetchPlanServer
{
public:
  struct ReceivedRequest
  {
    std::vector<std::string> active_features;
    std::vector<std::string> inactive_features;
    std::string resolution_model;
  };

  MockFetchPlanServer(
    rclcpp::Node::SharedPtr node,
    bool succeed,
    std::vector<aal_msgs::msg::Adaptation> plan_to_return)
  : succeed_(succeed), plan_to_return_(std::move(plan_to_return))
  {
    server_ = node->create_service<FetchPlanSrv>(
      "/fetch_plan_from_desired_features",
      [this](
        const std::shared_ptr<FetchPlanSrv::Request> request,
        std::shared_ptr<FetchPlanSrv::Response> response) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          received_requests_.push_back(
            {request->desired_features.active_features,
              request->desired_features.inactive_features,
              request->resolution_model});
        }
        requests_received_++;
        response->success = succeed_;
        response->adaptation_plan = plan_to_return_;
        response->feature_changes.active_features =
        request->desired_features.active_features;
        response->feature_changes.inactive_features =
        request->desired_features.inactive_features;
      });
  }

  int requests_received() const {return requests_received_.load();}

  std::vector<ReceivedRequest> received_requests() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_requests_;
  }

private:
  bool succeed_;
  std::vector<aal_msgs::msg::Adaptation> plan_to_return_;
  std::atomic<int> requests_received_{0};
  mutable std::mutex mutex_;
  std::vector<ReceivedRequest> received_requests_;
  rclcpp::Service<FetchPlanSrv>::SharedPtr server_;
};

// ─── Mock update_features_selected server ──────────────────────────────────

class MockUpdateFeaturesServer
{
public:
  explicit MockUpdateFeaturesServer(rclcpp::Node::SharedPtr node, bool succeed)
  : succeed_(succeed)
  {
    server_ = node->create_service<UpdateFeaturesSrv>(
      "/update_features_selected",
      [this](
        const std::shared_ptr<UpdateFeaturesSrv::Request> request,
        std::shared_ptr<UpdateFeaturesSrv::Response> response) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          received_changes_.push_back(request->feature_changes);
        }
        requests_received_++;
        response->success = succeed_;
        if (!succeed_) {
          response->message = "mock persistence failure";
        }
      });
  }

  int requests_received() const {return requests_received_.load();}

  std::vector<adapt_x_msgs::msg::DesiredFeatures> received_changes() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_changes_;
  }

private:
  bool succeed_;
  std::atomic<int> requests_received_{0};
  mutable std::mutex mutex_;
  std::vector<adapt_x_msgs::msg::DesiredFeatures> received_changes_;
  rclcpp::Service<UpdateFeaturesSrv>::SharedPtr server_;
};

// ─── Mock adapt_architecture server ──────────────────────────────────────────

class MockAdaptServer
{
public:
  MockAdaptServer(rclcpp::Node::SharedPtr node, bool succeed)
  : succeed_(succeed)
  {
    server_ = node->create_service<AdaptSrv>(
      "/adapt_architecture",
      [this](
        const std::shared_ptr<AdaptSrv::Request> request,
        std::shared_ptr<AdaptSrv::Response> response) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          received_plans_.push_back(request->adaptations);
        }
        requests_received_++;
        response->success = succeed_;
      });
  }

  int requests_received() const {return requests_received_.load();}

  std::vector<std::vector<aal_msgs::msg::Adaptation>> received_plans() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_plans_;
  }

private:
  bool succeed_;
  std::atomic<int> requests_received_{0};
  mutable std::mutex mutex_;
  std::vector<std::vector<aal_msgs::msg::Adaptation>> received_plans_;
  rclcpp::Service<AdaptSrv>::SharedPtr server_;
};

// ─── Test fixture ─────────────────────────────────────────────────────────────

class EnsureConfigurationTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    server_node_ = rclcpp::Node::make_shared("mock_server_node");
    factory_ = std::make_unique<BT::BehaviorTreeFactory>();
    factory_->registerNodeType<adapt_x_bt::EnsureConfiguration>("EnsureConfiguration");
    factory_->registerNodeType<AlwaysRunning>("AlwaysRunning");
  }

  void TearDown() override
  {
    if (server_executor_) {
      server_executor_->cancel();
    }
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
    mock_fetch_server_.reset();
    mock_adapt_server_.reset();
    mock_update_server_.reset();
    server_executor_.reset();
  }

  void startServers(
    bool fetch_succeed = true,
    bool adapt_succeed = true,
    bool update_succeed = true,
    std::vector<aal_msgs::msg::Adaptation> plan = makeTestPlan("mock_node"))
  {
    mock_fetch_server_ = std::make_unique<MockFetchPlanServer>(
      server_node_, fetch_succeed, std::move(plan));
    mock_adapt_server_ = std::make_unique<MockAdaptServer>(server_node_, adapt_succeed);
    mock_update_server_ = std::make_unique<MockUpdateFeaturesServer>(
      server_node_, update_succeed);
    server_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    server_executor_->add_node(server_node_);
    server_thread_ = std::thread([this]() {server_executor_->spin();});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::string makeXml(
    const std::string & child_type,
    const std::string & features_to_activate,
    const std::string & features_to_deactivate = "",
    const std::string & on_exit = "",
    const std::string & resolution_model = "")
  {
    std::string xml =
      R"(<root BTCPP_format="4"><BehaviorTree ID="T">)"
      R"(<EnsureConfiguration )"
      R"(features_to_activate=")" + features_to_activate + R"(")";
    if (!features_to_deactivate.empty()) {
      xml += R"( features_to_deactivate=")" + features_to_deactivate + R"(")";
    }
    if (!on_exit.empty()) {
      xml += R"( features_to_deactivate_on_exit=")" + on_exit + R"(")";
    }
    if (!resolution_model.empty()) {
      xml += R"( resolution_model=")" + resolution_model + R"(")";
    }
    xml += "><" + child_type + "/>";
    xml += "</EnsureConfiguration></BehaviorTree></root>";
    return xml;
  }

  BT::NodeStatus tickUntilDone(BT::Tree & tree, int max_ticks = 300)
  {
    BT::NodeStatus status = BT::NodeStatus::RUNNING;
    for (int i = 0; i < max_ticks && status == BT::NodeStatus::RUNNING; ++i) {
      status = tree.tickOnce();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return status;
  }

  rclcpp::Node::SharedPtr server_node_;
  std::unique_ptr<MockFetchPlanServer> mock_fetch_server_;
  std::unique_ptr<MockAdaptServer> mock_adapt_server_;
  std::unique_ptr<MockUpdateFeaturesServer> mock_update_server_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> server_executor_;
  std::thread server_thread_;
  std::unique_ptr<BT::BehaviorTreeFactory> factory_;
};

// ─── Tests ────────────────────────────────────────────────────────────────────

TEST_F(EnsureConfigurationTest, SetupSuccessChildSuccessDefaultTeardown)
{
  startServers();
  auto tree = factory_->createTreeFromText(
    makeXml("AlwaysSuccess", "identify_feature", "explore_feature"));

  EXPECT_EQ(tickUntilDone(tree), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(mock_fetch_server_->requests_received(), 2);
  EXPECT_EQ(mock_adapt_server_->requests_received(), 2);
  EXPECT_EQ(mock_update_server_->requests_received(), 2);

  auto fetch_reqs = mock_fetch_server_->received_requests();
  ASSERT_EQ(fetch_reqs.size(), 2u);
  EXPECT_EQ(fetch_reqs[0].active_features, std::vector<std::string>{"identify_feature"});
  EXPECT_EQ(fetch_reqs[0].inactive_features, std::vector<std::string>{"explore_feature"});
  // Teardown must deactivate what was activated (default behaviour)
  EXPECT_TRUE(fetch_reqs[1].active_features.empty());
  EXPECT_EQ(fetch_reqs[1].inactive_features, std::vector<std::string>{"identify_feature"});

  auto adapt_plans = mock_adapt_server_->received_plans();
  ASSERT_EQ(adapt_plans.size(), 2u);
  // adapt_architecture must receive exactly the plan fetch_plan returned.
  ASSERT_EQ(adapt_plans[0].size(), 1u);
  EXPECT_EQ(adapt_plans[0][0].node_name, "mock_node");

  auto changes = mock_update_server_->received_changes();
  ASSERT_EQ(changes.size(), 2u);
  EXPECT_EQ(
    changes[0].active_features,
    std::vector<std::string>{"identify_feature"});
  EXPECT_EQ(
    changes[0].inactive_features,
    std::vector<std::string>{"explore_feature"});
}

TEST_F(EnsureConfigurationTest, SetupSuccessChildFailureTeardownFires)
{
  startServers();
  auto tree = factory_->createTreeFromText(makeXml("AlwaysFailure", "identify_feature"));

  EXPECT_EQ(tickUntilDone(tree), BT::NodeStatus::FAILURE);
  EXPECT_EQ(mock_fetch_server_->requests_received(), 2);
  EXPECT_EQ(mock_adapt_server_->requests_received(), 2);
  EXPECT_EQ(mock_update_server_->requests_received(), 2);
}

TEST_F(EnsureConfigurationTest, FetchPlanFailureReturnsFailureChildNeverRuns)
{
  startServers(/*fetch_succeed=*/ false, /*adapt_succeed=*/ true);
  auto tree = factory_->createTreeFromText(makeXml("AlwaysSuccess", "identify_feature"));

  EXPECT_EQ(tickUntilDone(tree), BT::NodeStatus::FAILURE);
  // fetch_plan is queried once (setup); it fails, so adapt_architecture and
  // the teardown round trip are never reached.
  EXPECT_EQ(mock_fetch_server_->requests_received(), 1);
  EXPECT_EQ(mock_adapt_server_->requests_received(), 0);
  EXPECT_EQ(mock_update_server_->requests_received(), 0);
}

TEST_F(EnsureConfigurationTest, AdaptFailureReturnsFailureChildNeverRuns)
{
  startServers(/*fetch_succeed=*/ true, /*adapt_succeed=*/ false);
  auto tree = factory_->createTreeFromText(makeXml("AlwaysSuccess", "identify_feature"));

  EXPECT_EQ(tickUntilDone(tree), BT::NodeStatus::FAILURE);
  EXPECT_EQ(mock_fetch_server_->requests_received(), 1);
  EXPECT_EQ(mock_adapt_server_->requests_received(), 1);
  EXPECT_EQ(mock_update_server_->requests_received(), 0);
}

TEST_F(EnsureConfigurationTest, UpdateFailureReturnsFailureAfterAdaptation)
{
  startServers(
    /*fetch_succeed=*/ true,
    /*adapt_succeed=*/ true,
    /*update_succeed=*/ false);
  auto tree = factory_->createTreeFromText(makeXml("AlwaysSuccess", "identify_feature"));

  EXPECT_EQ(tickUntilDone(tree), BT::NodeStatus::FAILURE);
  EXPECT_EQ(mock_fetch_server_->requests_received(), 1);
  EXPECT_EQ(mock_adapt_server_->requests_received(), 1);
  EXPECT_EQ(mock_update_server_->requests_received(), 1);
}

TEST_F(EnsureConfigurationTest, TeardownSkippedWithNoneKeyword)
{
  startServers();
  auto tree = factory_->createTreeFromText(
    makeXml("AlwaysSuccess", "identify_feature", "", "none"));

  EXPECT_EQ(tickUntilDone(tree), BT::NodeStatus::SUCCESS);
  // Only the setup round trip — no teardown.
  EXPECT_EQ(mock_fetch_server_->requests_received(), 1);
  EXPECT_EQ(mock_adapt_server_->requests_received(), 1);
  EXPECT_EQ(mock_update_server_->requests_received(), 1);
}

TEST_F(EnsureConfigurationTest, CustomTeardownListUsed)
{
  startServers();
  auto tree = factory_->createTreeFromText(
    makeXml("AlwaysSuccess", "identify_feature", "", "custom_feature_a"));

  EXPECT_EQ(tickUntilDone(tree), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(mock_fetch_server_->requests_received(), 2);

  auto fetch_reqs = mock_fetch_server_->received_requests();
  ASSERT_EQ(fetch_reqs.size(), 2u);
  EXPECT_EQ(fetch_reqs[1].inactive_features, std::vector<std::string>{"custom_feature_a"});
}

TEST_F(EnsureConfigurationTest, SetupRequestCarriesFeaturesAndResolutionModel)
{
  startServers();
  auto tree = factory_->createTreeFromText(
    makeXml("AlwaysSuccess", "feature_a;feature_b", "feature_c", "", "res_mock"));

  EXPECT_EQ(tickUntilDone(tree), BT::NodeStatus::SUCCESS);

  auto fetch_reqs = mock_fetch_server_->received_requests();
  ASSERT_GE(fetch_reqs.size(), 1u);
  EXPECT_EQ(fetch_reqs[0].active_features, (std::vector<std::string>{"feature_a", "feature_b"}));
  EXPECT_EQ(fetch_reqs[0].inactive_features, std::vector<std::string>{"feature_c"});
  EXPECT_EQ(fetch_reqs[0].resolution_model, "res_mock");
}

TEST_F(EnsureConfigurationTest, HaltDuringChildRunsTeardownSynchronously)
{
  startServers();
  auto tree = factory_->createTreeFromText(makeXml("AlwaysRunning", "identify_feature"));

  // Tick until setup completes and the (always-RUNNING) child has been
  // reached at least once.
  for (int i = 0; i < 300 && mock_update_server_->requests_received() < 1; ++i) {
    tree.tickOnce();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(mock_fetch_server_->requests_received(), 1);
  ASSERT_EQ(mock_adapt_server_->requests_received(), 1);
  ASSERT_EQ(mock_update_server_->requests_received(), 1);

  // halt() performs the teardown fetch+adapt round trip via a bounded
  // synchronous wait, so both mock servers should have been hit again by
  // the time this call returns — no further ticking required.
  tree.haltTree();

  EXPECT_EQ(mock_fetch_server_->requests_received(), 2);
  EXPECT_EQ(mock_adapt_server_->requests_received(), 2);
  EXPECT_EQ(mock_update_server_->requests_received(), 2);
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
