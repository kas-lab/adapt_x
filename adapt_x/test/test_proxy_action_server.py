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
from types import SimpleNamespace

from aal_msgs.msg import Adaptation
import adapt_x.proxy_action_server as proxy_action_server
from adapt_x.proxy_action_server import normalize_actions
from adapt_x.proxy_action_server import parse_feature_action
from adapt_x.proxy_action_server import ProxyActionServer

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

    def info(self, message):
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


def make_proxy(fetch_responses, adaptation_response, update_responses):
    """Create the state required by execute_actions without a ROS node."""
    low_adaptation = Adaptation(node_name='low_node')
    high_adaptation = Adaptation(node_name='high_node')
    return SimpleNamespace(
        _monitor_client=SimpleNamespace(_enabled=True),
        fetch_plan_client=FakeClient(fetch_responses),
        aal_client=FakeClient([adaptation_response]),
        update_has_param_cli=FakeClient(update_responses),
        resolutions={
            'deactivate_low': [low_adaptation],
            'activate_high': [high_adaptation],
        },
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


def test_execute_actions_aggregates_plan_and_updates_each_feature():
    proxy = make_proxy(
        [
            SimpleNamespace(resolution_actions=['deactivate_low']),
            SimpleNamespace(resolution_actions=['activate_high']),
        ],
        SimpleNamespace(success=True),
        [SimpleNamespace(success=True), SimpleNamespace(success=True)],
    )

    assert execute(
        proxy,
        ['deselect_LOW_ALTITUDE', 'select_HIGH_ALTITUDE']) is True

    assert [request.feature_name for request in proxy.fetch_plan_client.requests] == [
        'LOW_ALTITUDE', 'HIGH_ALTITUDE']
    assert [request.activation for request in proxy.fetch_plan_client.requests] == [
        False, True]
    adaptations = proxy.aal_client.requests[0].adaptations
    assert [adaptation.node_name for adaptation in adaptations] == [
        'low_node', 'high_node']
    assert [
        request.feature_name
        for request in proxy.update_has_param_cli.requests
    ] == ['LOW_ALTITUDE', 'HIGH_ALTITUDE']
    assert proxy._monitor_client._enabled is True


def test_execute_actions_deduplicates_resolution_actions():
    proxy = make_proxy(
        [
            SimpleNamespace(resolution_actions=['deactivate_low']),
            SimpleNamespace(
                resolution_actions=['deactivate_low', 'activate_high']),
        ],
        SimpleNamespace(success=True),
        [SimpleNamespace(success=True), SimpleNamespace(success=True)],
    )

    execute(proxy, ['deselect_LOW_ALTITUDE', 'select_HIGH_ALTITUDE'])

    adaptations = proxy.aal_client.requests[0].adaptations
    assert [adaptation.node_name for adaptation in adaptations] == [
        'low_node', 'high_node']


def test_execute_actions_does_not_postprocess_failed_adaptation():
    proxy = make_proxy(
        [SimpleNamespace(resolution_actions=['activate_high'])],
        SimpleNamespace(success=False),
        [],
    )

    with pytest.raises(RuntimeError, match='Adaptation failed'):
        execute(proxy, ['select_HIGH_ALTITUDE'])

    assert proxy.update_has_param_cli.requests == []
    assert proxy._monitor_client._enabled is True


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
