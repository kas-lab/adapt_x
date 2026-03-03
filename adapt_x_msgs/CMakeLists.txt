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
  "srv/FetchApplicable.srv"
  "srv/UpdateRunning.srv"
  "srv/UpdateInEffect.srv"
  "srv/UpdateHasParamValue.srv"
  "srv/FetchAdaptations.srv"
  "srv/RequirementsFulfillment.srv"
  "srv/SatisfiedTactic.srv"
  DEPENDENCIES std_msgs rcl_interfaces aal_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
