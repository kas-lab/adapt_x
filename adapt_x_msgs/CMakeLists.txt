cmake_minimum_required(VERSION 3.8)
project(adapt_x_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(aal_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/FetchRules.srv"
  "srv/UpdateRunning.srv"
  "srv/UpdateInEffect.srv"
  "srv/UpdateHasParamValue.srv"
  "srv/FetchResolutions.srv"
  "srv/FetchPlan.srv"
  "srv/FetchPlanFromDesiredFeatures.srv"
  "srv/FetchPolicy.srv"
  "srv/RequirementsFulfillment.srv"
  "srv/SatisfiedRule.srv"
  "srv/HypotheticalReqFulfillment.srv"
  "srv/HypotheticalReqsFulfillment.srv"
  "srv/GetState.srv"
  "msg/RelaxedRequirement.msg"
  "msg/RelaxedRequirements.msg"
  "msg/SymbolicState.msg"
  "msg/TableEntry.msg"
  "msg/Resolution.msg"
  "msg/TacticRule.msg"
  "msg/DesiredFeatures.msg"
  DEPENDENCIES std_msgs rcl_interfaces aal_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
