import json
from pathlib import Path

from tqdm import tqdm
from with_argparse import with_argparse


@with_argparse
def types_conll(
    conll_files: list[Path],
    output_file: Path
):
    span_types = set()
    link_types = set()
    for file in conll_files:
        with file.open() as f:
            json_file = json.load(f)
        for elem in tqdm(json_file, desc=file.name):
            for entity in elem['entities']:
                span_types.add(entity['type'])
            for relation in elem['relations']:
                link_types.add(relation['type'])
        print(file.as_posix())

    span_types = list(span_types)
    link_types = list(link_types)
    print(f"found %d span types: %s" % (len(span_types), '; '.join(span_types)))
    print(f"found %d link types: %s" % (len(link_types), '; '.join(link_types)))

    with output_file.open("w") as f:
        json.dump(
            {
                "entities": {
                    span_type: {"short": span_type}
                    for span_type in span_types
                },
                "relations": {
                    link_type: {"short": link_type}
                    for link_type in link_types
                }
            }, f
        )
    pass


types_conll()
