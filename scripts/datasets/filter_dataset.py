import collections
import glob
from collections import defaultdict
from json import load, dump
from pathlib import Path

from tqdm import tqdm

from with_argparse import with_argparse


@with_argparse
def filter_dataset(
    path: Path,
    files: list[str]
):
    def flatten(c):
        return [a for b in c for a in b]

    def apply_glob(c, root_dir):
        if glob.has_magic(c):
            return glob.glob(c, root_dir=root_dir)
        return [c]

    for file in flatten([apply_glob(file, root_dir=path) for file in files]):
        if "types" in file:
            continue
        with open(path / file) as f:
            json_blob = load(f)
        items, new_items = [], []
        for item in (tq := tqdm(json_blob)):
            ends_at_pos = defaultdict(list)
            for entity in item['entities']:
                ends_at_pos[entity['end']].append(entity)

            new_item = {
                'tokens': item['tokens'],
                'entities': [],
                'relations': [],
            }
            for pos, entities in ends_at_pos.items():
                if len(entities) == 1:
                    continue
                for entity in entities:
                    new_item['entities'].append(entity)

                new_items.append(new_item)
                items.append(item)
                tq.set_postfix(collections.OrderedDict(items=len(items)))
                break
        with open(path / (file.replace(".json", "") + "_entities.json"), "w") as f:
            dump(new_items, f)
        with open(path / (file.replace(".json", "") + "_filtered.json"), "w") as f:
            dump(items, f)


filter_dataset()
