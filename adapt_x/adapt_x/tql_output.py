import os


def write_generated_tql(tql_query, output_path, default_base_dir,
                        filename="initial_insert_query.tql"):
    if output_path == "":
        output_path = os.path.join(
            default_base_dir,
            "generated",
            filename,
        )

    output_path = os.path.expanduser(str(output_path))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_path, filename)
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(tql_query)

    return output_path
