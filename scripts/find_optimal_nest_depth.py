import itertools
import json
from pathlib import Path

from with_argparse import with_argparse


def same_pos(e1, e2):
    return e1['start'] == e2['start'] and e1['end'] == e2['end']


@with_argparse(use_glob={"dataset_files"})
def find_optimal_nest_depth(dataset_files: list[Path]):
    max_beginning_inside = 0
    for file in dataset_files:
        if any(key in file.name for key in {'types', 'example'}):
            continue
        with file.open() as f:
            json_blob = json.load(f)
        print(file.name)
        for elem in json_blob:
            entities = elem["entities"]
            for entity in entities:
                beginning_inside_of_entity = [
                    other['start'] for other in elem['entities']
                    if (
                        entity['start'] < other['start'] < entity['end']
                        and not same_pos(entity, other)
                    )
                ]
                beginning_inside_of_entity = set(beginning_inside_of_entity)
                max_beginning_inside = max(max_beginning_inside, len(beginning_inside_of_entity))
    print(max_beginning_inside + 1)


find_optimal_nest_depth()
