# Copyright 2026 KAS-lab
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

import asyncio
import threading
from types import SimpleNamespace

from aal_msgs.msg import Adaptation
import adapt_x.proxy_action_server as proxy_action_server
from adapt_x.proxy_action_server import desired_features_from_actions
from adapt_x.proxy_action_server import normalize_actions
from adapt_x.proxy_action_server import parse_feature_action
from adapt_x.proxy_action_server import ProxyActionServer
from adapt_x.proxy_action_server import validate_operation_mode

import pytest


class FakeClient:
    """Record requests and return configured responses asynchronously."""

    def __init__(self, responses, available=True):
        self.responses = iter(responses)
        self.available = available
        self.requests = []

    def wait_for_service(self, timeout_sec):
        return self.available

    def call_async(self, request):
        self.requests.append(request)
        response = next(self.responses)

        async def complete():
            return response

        return complete()


class FakeLogger:
    """Minimal logger accepted by the execution helper."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def info(self, message, **kwargs):
        pass

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)


class FakeForwardedGoal:
    """Record cancellation attempts for a forwarded action goal."""

    def __init__(self, cancel_error=None):
        self.accepted = True
        self.cancel_error = cancel_error
        self.cancel_calls = 0

    def cancel_goal_async(self):
        self.cancel_calls += 1

        async def complete():
            if self.cancel_error is not None:
                raise self.cancel_error
            return SimpleNamespace()

        return complete()


class FakeServerGoal:
    """Record termination of the proxy action goal."""

    def __init__(self):
        self.abort_calls = 0

    def abort(self):
        self.abort_calls += 1


class FakeAction:
    """Provide the generated action Result API used by the proxy."""

    class Result:
        """Empty action result double."""


def make_proxy(fetch_response, adaptation_response, update_response):
    """Create the state required by execute_actions without a ROS node."""
    return SimpleNamespace(
        _monitor_client=SimpleNamespace(_enabled=True),
        fetch_plan_client=FakeClient([fetch_response]),
        aal_client=FakeClient([adaptation_response]),
        update_features_selected_client=FakeClient([update_response]),
        system_name='suave_resolution',
        get_logger=lambda: FakeLogger(),
    )


def execute(proxy, actions):
    """Invoke the unbound coroutine against the lightweight proxy double."""
    return asyncio.run(ProxyActionServer.execute_actions(proxy, actions))


def abort_for_error(proxy, server_goal, phase='Continuous'):
    """Invoke the unbound terminal-error coroutine."""
    return asyncio.run(ProxyActionServer._abort_for_adaptation_error(
        proxy, server_goal, phase, RuntimeError('boom')))


def sleep(proxy, seconds):
    """Invoke the unbound ROS timer sleep coroutine."""
    return asyncio.run(ProxyActionServer._sleep(proxy, seconds))


def test_normalize_actions_accepts_none_string_and_list():
    assert normalize_actions(None) == []
    assert normalize_actions('select_HIGH') == ['select_HIGH']
    assert normalize_actions(
        ['deselect_LOW', 'select_HIGH']) == [
            'deselect_LOW', 'select_HIGH']


def test_parse_feature_action():
    assert parse_feature_action('select_HIGH_ALTITUDE') == (
        'HIGH_ALTITUDE', True)
    assert parse_feature_action('deselect_LOW_ALTITUDE') == (
        'LOW_ALTITUDE', False)
    with pytest.raises(ValueError, match='Unsupported feature adaptation'):
        parse_feature_action('HIGH_ALTITUDE')


def test_desired_features_from_actions_deduplicates_identical_requests():
    desired = desired_features_from_actions([
        'select_HIGH_ALTITUDE',
        'deselect_LOW_ALTITUDE',
        'select_HIGH_ALTITUDE',
    ])

    assert desired.active_features == ['HIGH_ALTITUDE']
    assert desired.inactive_features == ['LOW_ALTITUDE']


def test_desired_features_from_actions_rejects_conflicts():
    with pytest.raises(ValueError, match='Conflicting adaptations'):
        desired_features_from_actions([
            'select_HIGH_ALTITUDE',
            'deselect_HIGH_ALTITUDE',
        ])


def test_desired_features_from_actions_rejects_empty_feature_name():
    with pytest.raises(ValueError, match='empty feature name'):
        desired_features_from_actions(['select_'])


def test_execute_actions_plans_batch_and_persists_actual_changes():
    low_adaptation = Adaptation(node_name='low_node')
    high_adaptation = Adaptation(node_name='high_node')
    feature_changes = proxy_action_server.DesiredFeatures(
        active_features=['HIGH_ALTITUDE'],
        inactive_features=['LOW_ALTITUDE', 'MEDIUM_ALTITUDE'])
    proxy = make_proxy(
        SimpleNamespace(
            success=True,
            adaptation_plan=[low_adaptation, high_adaptation],
            feature_changes=feature_changes),
        SimpleNamespace(success=True),
        SimpleNamespace(success=True),
    )

    assert execute(
        proxy,
        ['deselect_LOW_ALTITUDE', 'select_HIGH_ALTITUDE']) is True

    assert len(proxy.fetch_plan_client.requests) == 1
    plan_request = proxy.fetch_plan_client.requests[0]
    assert plan_request.resolution_model == 'suave_resolution'
    assert plan_request.desired_features.active_features == [
        'HIGH_ALTITUDE']
    assert plan_request.desired_features.inactive_features == [
        'LOW_ALTITUDE']
    adaptations = proxy.aal_client.requests[0].adaptations
    assert [adaptation.node_name for adaptation in adaptations] == [
        'low_node', 'high_node']
    update_request = proxy.update_features_selected_client.requests[0]
    assert update_request.feature_changes.active_features == [
        'HIGH_ALTITUDE']
    assert update_request.feature_changes.inactive_features == [
        'LOW_ALTITUDE', 'MEDIUM_ALTITUDE']
    assert proxy._monitor_client._enabled is True


def test_execute_actions_rejects_unsuccessful_plan():
    proxy = make_proxy(
        SimpleNamespace(success=False),
        SimpleNamespace(success=True),
        SimpleNamespace(success=True),
    )

    with pytest.raises(RuntimeError, match='Failed to fetch adaptation plan'):
        execute(proxy, ['select_HIGH_ALTITUDE'])

    assert proxy.aal_client.requests == []
    assert proxy.update_features_selected_client.requests == []


def test_execute_actions_rejects_conflict_before_planning():
    proxy = make_proxy(
        SimpleNamespace(success=True),
        SimpleNamespace(success=True),
        SimpleNamespace(success=True),
    )

    with pytest.raises(ValueError, match='Conflicting adaptations'):
        execute(proxy, [
            'select_HIGH_ALTITUDE',
            'deselect_HIGH_ALTITUDE',
        ])

    assert proxy.fetch_plan_client.requests == []
    assert proxy.aal_client.requests == []
    assert proxy.update_features_selected_client.requests == []


def test_execute_actions_does_not_persist_failed_adaptation():
    proxy = make_proxy(
        SimpleNamespace(
            success=True,
            adaptation_plan=[Adaptation(node_name='high_node')],
            feature_changes=proxy_action_server.DesiredFeatures(
                active_features=['HIGH_ALTITUDE'])),
        SimpleNamespace(success=False),
        SimpleNamespace(success=True),
    )

    with pytest.raises(RuntimeError, match='Adaptation failed'):
        execute(proxy, ['select_HIGH_ALTITUDE'])

    assert proxy.update_features_selected_client.requests == []
    assert proxy._monitor_client._enabled is True


def test_execute_actions_skips_aal_when_all_features_are_satisfied():
    proxy = make_proxy(
        SimpleNamespace(
            success=True,
            adaptation_plan=[],
            feature_changes=proxy_action_server.DesiredFeatures()),
        SimpleNamespace(success=True),
        SimpleNamespace(success=True),
    )

    assert execute(proxy, ['select_HIGH_ALTITUDE']) is True

    assert proxy.aal_client.requests == []
    assert proxy.update_features_selected_client.requests == []
    assert proxy._monitor_client._enabled is True


def test_execute_actions_persists_logical_change_with_empty_aal_plan():
    proxy = make_proxy(
        SimpleNamespace(
            success=True,
            adaptation_plan=[],
            feature_changes=proxy_action_server.DesiredFeatures(
                active_features=['TASK'])),
        SimpleNamespace(success=True),
        SimpleNamespace(success=True),
    )

    assert execute(proxy, ['select_TASK']) is True

    assert proxy.aal_client.requests[0].adaptations == []
    changes = (
        proxy.update_features_selected_client.requests[0].feature_changes)
    assert changes.active_features == ['TASK']


def test_execute_actions_reports_selected_state_persistence_failure():
    proxy = make_proxy(
        SimpleNamespace(
            success=True,
            adaptation_plan=[Adaptation(node_name='high_node')],
            feature_changes=proxy_action_server.DesiredFeatures(
                active_features=['HIGH_ALTITUDE'])),
        SimpleNamespace(success=True),
        SimpleNamespace(success=False),
    )

    with pytest.raises(
            RuntimeError, match='Failed to update feature selected state'):
        execute(proxy, ['select_HIGH_ALTITUDE'])

    assert len(proxy.aal_client.requests) == 1
    assert len(proxy.update_features_selected_client.requests) == 1


def test_abort_for_adaptation_error_cancels_shuts_down_and_aborts():
    logger = FakeLogger()
    forwarded_goal = FakeForwardedGoal()
    server_goal = FakeServerGoal()
    shutdown_calls = []
    proxy = SimpleNamespace(
        client_goal_handle=forwarded_goal,
        interface_type=FakeAction,
        get_logger=lambda: logger,
        trigger_shutdown=lambda: shutdown_calls.append(True),
    )

    result = abort_for_error(proxy, server_goal)

    assert isinstance(result, FakeAction.Result)
    assert forwarded_goal.cancel_calls == 1
    assert shutdown_calls == [True]
    assert server_goal.abort_calls == 1
    assert logger.errors == ['Continuous adaptation failed: boom']


def test_abort_for_adaptation_error_continues_after_cleanup_failures():
    logger = FakeLogger()
    forwarded_goal = FakeForwardedGoal(RuntimeError('cancel failed'))
    server_goal = FakeServerGoal()

    def fail_shutdown():
        raise RuntimeError('shutdown failed')

    proxy = SimpleNamespace(
        client_goal_handle=forwarded_goal,
        interface_type=FakeAction,
        get_logger=lambda: logger,
        trigger_shutdown=fail_shutdown,
    )

    result = abort_for_error(proxy, server_goal, phase='Initial')

    assert isinstance(result, FakeAction.Result)
    assert forwarded_goal.cancel_calls == 1
    assert server_goal.abort_calls == 1
    assert logger.warnings == [
        'Failed to cancel forwarded action goal: cancel failed',
        'Failed to shut down adaptation proxy: shutdown failed',
    ]


def test_sleep_yields_until_timer_fires_and_destroys_timer(monkeypatch):
    timers = []
    destroyed_timers = []

    class FakeFuture:

        def __init__(self, executor):
            self.future = asyncio.get_running_loop().create_future()

        def __await__(self):
            return self.future.__await__()

        def done(self):
            return self.future.done()

        def set_result(self, result):
            self.future.set_result(result)

    def create_timer(seconds, callback, callback_group):
        timer = SimpleNamespace(
            seconds=seconds, callback_group=callback_group)
        timers.append(timer)
        asyncio.get_running_loop().call_soon(callback)
        return timer

    monkeypatch.setattr(proxy_action_server, 'Future', FakeFuture)
    callback_group = object()
    proxy = SimpleNamespace(
        executor=object(),
        parallel_cb_group=callback_group,
        create_timer=create_timer,
        destroy_timer=destroyed_timers.append,
    )

    sleep(proxy, 0.1)

    assert timers[0].seconds == 0.1
    assert timers[0].callback_group is callback_group
    assert destroyed_timers == timers


def test_get_adaptation_period_reads_current_parameter_value():
    parameter_values = iter([0.1, 0.5])

    def get_parameter(name):
        assert name == 'adaptation_period'
        return SimpleNamespace(value=next(parameter_values))

    proxy = SimpleNamespace(get_parameter=get_parameter)

    assert ProxyActionServer._get_adaptation_period(proxy) == 0.1
    assert ProxyActionServer._get_adaptation_period(proxy) == 0.5


def test_get_adaptation_period_rejects_non_positive_values():
    proxy = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value=0.0))

    with pytest.raises(
            ValueError, match='adaptation_period must be greater than zero'):
        ProxyActionServer._get_adaptation_period(proxy)


def test_validate_operation_mode_accepts_supported_modes():
    assert validate_operation_mode('goal_bound', '') == 'goal_bound'
    assert validate_operation_mode(
        'goal_bound', 'qlearning') == 'goal_bound'
    assert validate_operation_mode('autonomous', 'tactic') == 'autonomous'


def test_validate_operation_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match='Unsupported operation_mode'):
        validate_operation_mode('always_on', 'tactic')


def test_validate_operation_mode_requires_tactic_for_autonomous():
    with pytest.raises(
            ValueError, match='requires the tactic adaptation strategy'):
        validate_operation_mode('autonomous', 'qlearning')


def test_expose_adaptation_interface_autonomous_self_activates(monkeypatch):
    def forbid_action_server(*args, **kwargs):
        raise AssertionError('ActionServer must not be created')

    monkeypatch.setattr(
        proxy_action_server, 'ActionServer', forbid_action_server)
    published = []
    activations = []
    proxy = SimpleNamespace(
        operation_mode='autonomous',
        ready_publisher=SimpleNamespace(publish=published.append),
        trigger_activate=lambda: activations.append(True),
    )

    ProxyActionServer._expose_adaptation_interface(proxy)

    assert len(published) == 1
    assert activations == [True]


def test_expose_adaptation_interface_goal_bound_creates_server(monkeypatch):
    servers = []

    class FakeActionServer:
        """Record proxy action server construction arguments."""

        def __init__(self, node, interface_type, name, **kwargs):
            servers.append((interface_type, name, kwargs))

    monkeypatch.setattr(
        proxy_action_server, 'ActionServer', FakeActionServer)
    published = []
    activations = []
    proxy = SimpleNamespace(
        operation_mode='goal_bound',
        interface_type=FakeAction,
        interface_name='spiral_search',
        parallel_cb_group=object(),
        execute_callback=object(),
        goal_callback=object(),
        cancel_callback=object(),
        ready_publisher=SimpleNamespace(publish=published.append),
        trigger_activate=lambda: activations.append(True),
    )

    ProxyActionServer._expose_adaptation_interface(proxy)

    assert isinstance(proxy._action_server, FakeActionServer)
    assert servers[0][0] is FakeAction
    assert servers[0][1] == '/proxy_spiral_search'
    assert len(published) == 1
    assert activations == []


def test_start_autonomous_loop_starts_exactly_one_task():
    created = []
    loop_done = threading.Event()
    loop_done.set()
    proxy = SimpleNamespace(
        _autonomous_task=None,
        _autonomous_stop=True,
        _autonomous_loop_done=loop_done,
        _autonomous_loop=object(),
        executor=SimpleNamespace(
            create_task=lambda coro: created.append(coro) or 'task'),
    )

    ProxyActionServer._start_autonomous_loop(proxy)

    assert proxy._autonomous_stop is False
    assert proxy._autonomous_task == 'task'
    assert len(created) == 1
    assert not loop_done.is_set()

    proxy._autonomous_stop = True
    ProxyActionServer._start_autonomous_loop(proxy)

    assert len(created) == 1
    assert proxy._autonomous_stop is True


def test_stop_autonomous_loop_waits_for_running_task():
    waits = []
    proxy = SimpleNamespace(
        operation_mode='autonomous',
        _autonomous_stop=False,
        _autonomous_task=object(),
        _autonomous_loop_done=SimpleNamespace(
            wait=lambda: waits.append(True)),
    )

    ProxyActionServer._stop_autonomous_loop(proxy)

    assert proxy._autonomous_stop is True
    assert waits == [True]


def test_run_adaptation_iteration_executes_strategy_actions():
    executed = []

    async def execute_actions(actions):
        executed.append(actions)

    proxy = SimpleNamespace(
        strategy=SimpleNamespace(
            on_adaptation_loop=lambda: ['select_RECOVER_MOTION']),
        execute_actions=execute_actions,
    )

    asyncio.run(ProxyActionServer._run_adaptation_iteration(proxy))

    assert executed == [['select_RECOVER_MOTION']]


def test_run_adaptation_iteration_skips_empty_batch():
    async def execute_actions(actions):
        raise AssertionError('execute_actions must not be called')

    proxy = SimpleNamespace(
        strategy=SimpleNamespace(on_adaptation_loop=lambda: None),
        execute_actions=execute_actions,
    )

    asyncio.run(ProxyActionServer._run_adaptation_iteration(proxy))


def make_autonomous_proxy(monitored=True):
    """Create the state required by the autonomous loop coroutine."""
    logger = FakeLogger()
    loop_done = threading.Event()
    proxy = SimpleNamespace(
        monitored=monitored,
        _autonomous_stop=False,
        _autonomous_task=object(),
        _autonomous_loop_done=loop_done,
        get_logger=lambda: logger,
        message=lambda msg: None,
        _get_adaptation_period=lambda: 0.1,
        trigger_deactivate=lambda: proxy.deactivations.append(True),
        deactivations=[],
        iterations=[],
        sleeps=[],
    )
    return proxy, logger


def run_autonomous_loop(proxy):
    """Invoke the unbound autonomous loop coroutine."""
    asyncio.run(ProxyActionServer._autonomous_loop(proxy))


def test_autonomous_loop_executes_iterations_until_stopped():
    proxy, logger = make_autonomous_proxy()

    async def run_iteration():
        proxy.iterations.append(True)

    async def fake_sleep(seconds):
        proxy.sleeps.append(seconds)
        if len(proxy.sleeps) == 2:
            proxy._autonomous_stop = True

    proxy._run_adaptation_iteration = run_iteration
    proxy._sleep = fake_sleep

    run_autonomous_loop(proxy)

    assert proxy.iterations == [True, True]
    assert proxy.sleeps == [0.1, 0.1]
    assert proxy._autonomous_task is None
    assert proxy.deactivations == []
    assert logger.errors == []


def test_autonomous_loop_waits_for_first_monitoring_result():
    proxy, logger = make_autonomous_proxy(monitored=False)

    async def run_iteration():
        proxy.iterations.append(True)

    async def fake_sleep(seconds):
        proxy.sleeps.append(seconds)
        if len(proxy.sleeps) == 1:
            proxy.monitored = True
        else:
            proxy._autonomous_stop = True

    proxy._run_adaptation_iteration = run_iteration
    proxy._sleep = fake_sleep

    run_autonomous_loop(proxy)

    assert proxy.iterations == [True]
    assert len(proxy.sleeps) == 2


def test_autonomous_loop_deactivates_after_iteration_error():
    proxy, logger = make_autonomous_proxy()

    async def run_iteration():
        proxy.iterations.append(True)
        raise RuntimeError('kb unavailable')

    async def fake_sleep(seconds):
        proxy.sleeps.append(seconds)

    proxy._run_adaptation_iteration = run_iteration
    proxy._sleep = fake_sleep

    run_autonomous_loop(proxy)

    assert proxy.iterations == [True]
    assert proxy.sleeps == []
    assert logger.errors == [
        'Autonomous adaptation terminated: kb unavailable']
    assert proxy.deactivations == [True]
    assert proxy._autonomous_task is None
    assert proxy._autonomous_loop_done.is_set()


def test_autonomous_loop_terminal_error_deactivates():
    proxy, logger = make_autonomous_proxy()

    async def run_iteration():
        proxy.iterations.append(True)

    async def fake_sleep(seconds):
        proxy.sleeps.append(seconds)

    def bad_period():
        raise ValueError('adaptation_period must be greater than zero')

    proxy._run_adaptation_iteration = run_iteration
    proxy._sleep = fake_sleep
    proxy._get_adaptation_period = bad_period

    run_autonomous_loop(proxy)

    assert proxy.iterations == [True]
    assert proxy.sleeps == []
    assert logger.errors == [
        'Autonomous adaptation terminated: '
        'adaptation_period must be greater than zero']
    assert proxy.deactivations == [True]
    assert proxy._autonomous_task is None
    assert proxy._autonomous_loop_done.is_set()
