# Repository Guidelines

## Project Structure & Module Organization

This repository contains two ROS 2 packages. `adapt_x/` is the Python package that implements the AdaptX adaptation engine, JPype/Java bridge integration, ROS nodes, tests, bundled JARs, and configuration files. Runtime Python modules live in `adapt_x/adapt_x/`; tests live in `adapt_x/test/`; Java dependencies are under `adapt_x/jars/`; feature-model configuration is under `adapt_x/config/`. `adapt_x_msgs/` is an `ament_cmake` interface package containing message definitions in `msg/` and service definitions in `srv/`.

## Build, Test, and Development Commands

Run commands from the workspace root, usually `~/ros_workspaces/suave_rebetmc_ws`, after sourcing ROS Humble:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select adapt_x adapt_x_msgs
source install/setup.bash
colcon test --packages-select adapt_x adapt_x_msgs --event-handlers console_direct+
colcon test-result --verbose
```

Use `ros2 run adapt_x setup_node`, `ros2 run adapt_x adaptation_engine`, or another console script from `adapt_x/setup.py` to run installed nodes locally.

For source-tree tests that import generated ROS interfaces, preserve the
installed overlay on `PYTHONPATH`:

```bash
source install/setup.bash
PYTHONPATH=src/adapt_x/adapt_x:$PYTHONPATH \
  python3 -m pytest -q src/adapt_x/adapt_x/test/test_proxy_action_server.py
```

Do not replace `PYTHONPATH` after sourcing the overlay; doing so hides generated
packages such as `aal_msgs`.

## Proxy Action Server

- Adaptation strategies may return a string, a list of strings, or no action.
  Normalize this at the proxy boundary and execute one combined AAL request per
  batch.
- Keep feature names, resolution actions, and adaptation plans local to the
  coroutine. Do not reintroduce shared `current_feature`, current-plan, or
  completion flags for per-batch flow.
- A `FetchPlan` response with no resolution actions is a successful no-op. If
  the entire batch is empty, skip AAL and parameter-state postprocessing. For a
  mixed batch, postprocess only features with nonempty plans.
- Read the `adaptation_period` ROS parameter immediately before each wait and
  wait after every continuous-loop iteration, including no-op iterations, so
  runtime parameter updates change the effective rate.
- Terminal adaptation errors best-effort cancel an accepted forwarded goal,
  trigger shutdown, abort the proxy goal, and return an empty action result.
- ROS 2 Humble action execute callbacks may be coroutines and may await
  `rclpy.task.Future` objects. Humble `Executor` has no `create_future()`; use
  `Future(executor=self.executor)` and complete it from a ROS timer for a
  cooperative delay. Destroy the temporary timer afterward. Do not assume
  arbitrary `asyncio` awaitables work under the `rclpy` executor.
- `MultiThreadedExecutor` plus `ReentrantCallbackGroup` permits callback
  progress, but shared fields such as `client_goal_handle` and
  `result_received` remain unsafe across simultaneous goals. Treat the proxy as
  effectively single-goal until state is per-goal or concurrent goals are
  rejected.
- `FetchPlan.srv` currently returns only an action array. Empty therefore
  conflates an already-satisfied transition with a required transition that has
  no configured plan; extend the service response before callers need to
  distinguish those cases.

## Coding Style & Naming Conventions

Python targets ROS 2 Humble on Python 3.10. Follow PEP 8 with 4-space indentation, `snake_case` modules/functions, and descriptive ROS node names. Prefer `pathlib.Path` for filesystem logic. Keep comments short and focused on non-obvious behavior. New Python files should include the workspace Apache-2.0 copyright header. ROS messages and services use PascalCase filenames, for example `RelaxedRequirement.msg` and `FetchPlan.srv`.

## Testing Guidelines

Tests use `pytest` through `colcon test`. Keep Python tests in `adapt_x/test/` with `test_*.py` names. The package already includes lint checks for copyright, Flake8, and PEP 257, so run the full package test command before opening a PR. Add focused tests for generated TypeQL output, adaptation model behavior, or node setup whenever changing those areas.

## Commit & Pull Request Guidelines

Recent history uses short, direct commit summaries such as `save tql in setup_node.py` and `small adjustments`. Prefer concise imperative messages that name the changed component, for example `fix setup_node tql output`. Pull requests should describe behavior changes, list test commands run, link related issues, and call out changes to bundled JARs, message/service definitions, or generated TypeQL because those affect downstream ROS packages.

## Configuration & Dependencies

Do not remove bundled JARs or config files unless the Java-side model dependencies are updated together. After editing `adapt_x_msgs/msg` or `adapt_x_msgs/srv`, rebuild and source the workspace so generated interfaces are available to Python nodes.
