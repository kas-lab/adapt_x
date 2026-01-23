# Copyright 2025 KAS Lab
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

import pytest

from typedb_tactics.typedb_tactical_retreat_interface import TacticalRetreatInterface
from adapt_x.adaptation_engine import process_set_parameter, find_subquery
from ros_typedb.ros_typedb_interface import fetch_query_result_to_ros_msg
from aal_msgs.msg import Adaptation


@pytest.fixture(scope='module')
def kb_interface():
    kb_interface = TacticalRetreatInterface(
        'localhost:1729',
        'test_tactics_model',
        ['../typedb_tactics/schemas/data_structure/data_structure.tql',
            '../typedb_tactics/schemas/context_model/context_model.tql',
            '../typedb_tactics/schemas/logical_expressions/logical_expression.tql',
            '../typedb_tactics/schemas/ros_model/ros_model.tql',
            '../typedb_tactics/schemas/feature_model/feature_model.tql',
            '../typedb_tactics/schemas/tactics_model/tactics_model.tql',
            '../typedb_tactics/schemas/discover_tactics_model/discover_tactics_model.tql'],
        ['../typedb_tactics/schemas/ros_model/ros_model_example.tql',
            '../typedb_tactics/schemas/tactics_model/tactics_example.tql'],
        force_database=True,
        force_data=True,
        infer=True)
    yield kb_interface
    # kb_interface.delete_database()


def test_fetch_to_adaptation(kb_interface):

    tactics = kb_interface.get_applicable_tactics(
        'rebetmc', '/navigate_to_pose')
    assert len(tactics) == 3

    from eu.coresense.adaptation.generator import TQLGenerator
    from eu.coresense.common import NameProvider

    generator = TQLGenerator(NameProvider())
    query = generator.generateFetchApplicableTactics("rebetmc", "/navigate_to_pose")

    dict_result = kb_interface.fetch_database(str(query))
    response = fetch_query_result_to_ros_msg(dict_result)
    index_lists, index = find_subquery(response.results,
                                       str(TQLGenerator.PARAM_VALUES_SUBQUERY_NAME))
    result_tree = response.results[index]
    adaptation = process_set_parameter(result_tree, index_lists)
    print(str(adaptation))
    assert adaptation.adaptation_target == Adaptation.ROSPARAMETER
    assert adaptation.node_name == "velocity_smoother"
    assert adaptation.parameter_adaptation.name == "max_velocity"
