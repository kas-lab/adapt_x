# Copyright 2026 KAS-lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://apache.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

from aal_msgs.msg import Adaptation
from adapt_x.tactical_adaptation_node import TacticalAdaptationNode


class _Logger:
    def info(self, message):
        pass

    def error(self, message):
        pass


def _make_node():
    node = TacticalAdaptationNode.__new__(TacticalAdaptationNode)
    node.interfaces = set()
    node.resolution_model = "rebetmc"
    node.rule_dict = {}
    node.rule_to_actions = {}
    node.fetch_applicable_client = object()
    node.fetch_actions_client = object()
    node.satisfied_tactic_client = object()
    node.aal_client = object()
    node.get_logger = lambda: _Logger()
    return node


def test_build_interface_cache_fetches_tactics_and_actions():
    node = _make_node()
    adaptation = Adaptation()
    calls = []

    def call_service(client, request, identifier, timeout_sec=10.0):
        calls.append((identifier, request))
        if identifier == "Fetch Applicable Tactics":
            return SimpleNamespace(success=True, tactics=["rule_a", "rule_b"])
        if identifier == "Fetch Resolution Actions":
            return SimpleNamespace(success=True, adaptations=[adaptation])
        raise AssertionError(f"Unexpected service call: {identifier}")

    node._call_service = call_service

    rule_dict, rule_to_actions = node._build_interface_cache({"navigate_to_pose"})

    assert rule_dict == {"navigate_to_pose": ["rule_a", "rule_b"]}
    assert rule_to_actions == {"rule_a": [adaptation], "rule_b": [adaptation]}
    assert calls[0][1].resolution_model == "rebetmc"
    assert calls[0][1].interface_name == "navigate_to_pose"
    assert calls[1][1].tactic_name == "rule_a"
    assert calls[2][1].tactic_name == "rule_b"


def test_update_interfaces_is_atomic_when_refresh_fails():
    node = _make_node()
    node.interfaces = {"navigate_to_pose"}
    node.rule_dict = {"navigate_to_pose": ["rule_a"]}
    node.rule_to_actions = {"rule_a": [Adaptation()]}

    def refresh_interface_cache(interfaces):
        raise RuntimeError("KB unavailable")

    node._refresh_interface_cache = refresh_interface_cache

    request = SimpleNamespace(
        add_interfaces=["detect_object_srv"],
        remove_interfaces=["navigate_to_pose"],
    )
    response = SimpleNamespace(interfaces=[], success=True, message="")

    result = node.update_interfaces_callback(request, response)

    assert result.success is False
    assert result.interfaces == ["navigate_to_pose"]
    assert "KB unavailable" in result.message
    assert node.interfaces == {"navigate_to_pose"}
    assert node.rule_dict == {"navigate_to_pose": ["rule_a"]}


def test_tactic_adaptation_uses_first_satisfied_rule_and_calls_aal():
    node = _make_node()
    rule_a_adaptation = Adaptation()
    rule_b_adaptation = Adaptation()
    node.rule_dict = {"navigate_to_pose": ["rule_a", "rule_b"]}
    node.rule_to_actions = {
        "rule_a": [rule_a_adaptation],
        "rule_b": [rule_b_adaptation],
    }
    aal_requests = []

    def call_service(client, request, identifier, timeout_sec=10.0):
        if identifier == "Satisfied Tactic":
            assert request.tactics == ["rule_a", "rule_b"]
            return SimpleNamespace(success=True, satisfied=[False, True])
        if identifier == "Adapt Architecture":
            aal_requests.append(request)
            return SimpleNamespace(success=True)
        raise AssertionError(f"Unexpected service call: {identifier}")

    node._call_service = call_service

    request = SimpleNamespace(
        child_description=SimpleNamespace(interface_name="navigate_to_pose")
    )
    response = SimpleNamespace(success=False)

    result = node.tactic_adaptation_callback(request, response)

    assert result.success is True
    assert len(aal_requests) == 1
    assert aal_requests[0].adaptations == [rule_b_adaptation]
