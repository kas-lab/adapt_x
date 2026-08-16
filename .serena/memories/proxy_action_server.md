# Proxy Action Server

- Strategies may return one action, multiple actions, or no action. Normalize at the proxy boundary and make one combined AAL request per batch.
- Keep batch feature names, resolution actions, and adaptation plans local; do not restore shared current-feature/current-plan/completion fields.
- Empty aggregate resolution actions are a successful no-op: skip AAL and parameter-state postprocessing. Mixed batches postprocess only nonempty feature plans.
- Read `adaptation_period` dynamically and wait after every continuous-loop iteration, including no-op iterations.
- Terminal adaptation errors best-effort cancel accepted forwarded work, trigger shutdown, abort the proxy goal, and return an empty result.
- Humble coroutine callbacks await `rclpy.task.Future`. `Executor` has no `create_future()`; use `Future(executor=self.executor)`. Cooperative delays use a ROS timer to complete the future and destroy the timer afterward. Do not assume arbitrary `asyncio` awaitables work.
- Multithreaded/reentrant execution permits callback progress but does not make `client_goal_handle`, `result_received`, and other shared goal fields concurrency-safe. Treat the proxy as single-goal until state is per-goal or concurrent goals are rejected.
- `FetchPlan.srv` has no status/reason fields, so empty conflates “already satisfied” and “missing configured plan.”
- Source `install/setup.bash` and use `PYTHONPATH=src/adapt_x/adapt_x:$PYTHONPATH` for source tests needing generated ROS interfaces.
