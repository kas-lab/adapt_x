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

"""Test adaptation while keeping or restarting the underlying action goal."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from adapt_x import proxy_action_server as proxy

import pytest


def completed(value=None):
    """Return a future whose callbacks execute immediately."""
    future = Future()
    future.set_result(value)
    return future


def make_node(cancel):
    """Supply only the collaborators used by the adaptation helper."""
    node = SimpleNamespace(
        get_parameter=Mock(return_value=SimpleNamespace(value=cancel)),
        message=Mock(), get_logger=lambda: Mock(),
        ignore_result_on_cancel=False, cancel_complete=False,
        cancel_goal=Mock(return_value=completed()),
        strategy=SimpleNamespace(
            on_adaptation_loop=Mock(return_value='SEL_X')),
        execute_action=Mock(), _action_client=Mock(),
        goal_request=object(), feedback=object(), feedback_callback=Mock(),
        goal_received=False, result_received=False,
        get_clock=lambda: SimpleNamespace(now=lambda: 'now'), start_grace=None)
    node._on_cancel_goal_complete = lambda _: setattr(
        node, 'cancel_complete', True)
    node._on_goal_sent = lambda _: setattr(node, 'goal_received', True)
    node._action_client.send_goal_async.return_value = completed()
    return node


@pytest.mark.parametrize('action', ['SEL_X', None])
def test_adapt_without_cancelling(action):
    """Do not cancel or resend when the parameter is false."""
    node = make_node(False)
    node.strategy.on_adaptation_loop.return_value = action
    proxy.ProxyActionServer._adapt_running_goal(node)
    node.cancel_goal.assert_not_called()
    node._action_client.send_goal_async.assert_not_called()
    assert not node.ignore_result_on_cancel
    assert node.start_grace is None
    if action is not None:
        node.execute_action.assert_called_once_with(action)
    else:
        node.execute_action.assert_not_called()


def test_normal_result_during_adaptation_is_retained():
    """A result arriving while AAL executes must not be ignored."""
    node = make_node(False)
    result = object()
    node.execute_action.side_effect = lambda _: (
        proxy.ProxyActionServer.get_result_callback(
            node, completed(SimpleNamespace(result=result, status=4))))
    proxy.ProxyActionServer._adapt_running_goal(node)
    assert node.result_received
    assert node.result is result
    assert node.result_status == 4
    node._action_client.send_goal_async.assert_not_called()


def test_cancel_adapt_resend_uses_one_parameter_snapshot(monkeypatch):
    """Preserve the original path even if the parameter changes mid-cycle."""
    monkeypatch.setattr(proxy.time, 'sleep', Mock())
    reconstruct = Mock(return_value='resent')
    monkeypatch.setattr(proxy, 'construct_repeat_goal', reconstruct)
    node = make_node(True)
    events = []
    node.cancel_goal.side_effect = lambda: (
        events.append('cancel'), completed())[1]

    def adapt(_):
        events.append('adapt')
        node.get_parameter.return_value.value = False

    node.execute_action.side_effect = adapt
    node._action_client.send_goal_async.side_effect = lambda *a, **kw: (
        events.append('resend'), completed())[1]
    proxy.ProxyActionServer._adapt_running_goal(node)
    assert events == ['cancel', 'adapt', 'resend']
    node.get_parameter.assert_called_once_with('cancel_goal_on_adaptation')
    assert node.start_grace == 'now'


def test_live_adaptation_failure_propagates():
    """Use existing task-abort handling rather than retrying on AAL failure."""
    node = make_node(False)
    node.execute_action.side_effect = RuntimeError('AAL failed')
    with pytest.raises(RuntimeError, match='AAL failed'):
        proxy.ProxyActionServer._adapt_running_goal(node)
    node.cancel_goal.assert_not_called()
    node._action_client.send_goal_async.assert_not_called()


def test_explicit_client_cancel_is_unconditional():
    """Client cancellation is independent of adaptation policy."""
    node = make_node(False)
    response = proxy.ProxyActionServer.cancel_callback(node, Mock())
    assert response == proxy.CancelResponse.ACCEPT
    node.cancel_goal.assert_called_once()
    node.get_parameter.assert_not_called()
