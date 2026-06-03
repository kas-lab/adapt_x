import os

from adapt_x.tql_output import write_generated_tql


def test_write_generated_tql_writes_query_to_configured_path(tmp_path):
    output_path = tmp_path / "generated" / "initial_insert_query.tql"
    query = "insert\n$node isa ros-node;\n"

    saved_path = write_generated_tql(query, str(output_path), str(tmp_path))

    assert output_path.read_text(encoding="utf-8") == query
    assert saved_path == str(output_path)


def test_write_generated_tql_uses_default_share_relative_path(tmp_path):
    query = "insert\n$system isa ros-system;\n"

    saved_path = write_generated_tql(query, "", str(tmp_path))

    expected_path = tmp_path / "generated" / "initial_insert_query.tql"
    assert expected_path.read_text(encoding="utf-8") == query
    assert saved_path == os.fspath(expected_path)


def test_write_generated_tql_uses_configured_default_filename(tmp_path):
    query = "insert\n$mission isa mission;\n"

    saved_path = write_generated_tql(
        query,
        "",
        str(tmp_path),
        "custom_initial_insert.tql",
    )

    expected_path = tmp_path / "generated" / "custom_initial_insert.tql"
    assert expected_path.read_text(encoding="utf-8") == query
    assert saved_path == os.fspath(expected_path)


def test_write_generated_tql_prefers_configured_path_over_filename(tmp_path):
    output_path = tmp_path / "explicit" / "query.tql"
    query = "insert\n$explicit isa output;\n"

    saved_path = write_generated_tql(
        query,
        str(output_path),
        str(tmp_path),
        "ignored.tql",
    )

    assert output_path.read_text(encoding="utf-8") == query
    assert saved_path == os.fspath(output_path)
    assert not (tmp_path / "generated" / "ignored.tql").exists()
