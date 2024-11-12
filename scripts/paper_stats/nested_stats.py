import json

from transformers import AutoTokenizer
from with_argparse import with_argparse

from iter.datasets import CoNLL


@with_argparse
def nested_stats(dataset: str, tokenizer: str = 't5-small'):
    dataset = CoNLL.from_name(dataset, tokenizer=AutoTokenizer.from_pretrained(tokenizer))
    dataset.setup_dataset()

    def same_pos(e1, e2):
        return e1['start'] == e2['start'] and e1['end'] == e2['end']

    any_total = 0
    any_overlays = 0
    any_nesteds = 0
    for split, filename in dataset.splits.items():
        if split == "types":
            continue
        with open(dataset.split_path(filename)) as f:
            elements = json.load(f)

        any_total += len(elements)
        split_any_nesteds = 0
        split_any_overlays = 0
        for elem in elements:
            any_overlay = False
            any_nested = False

            in_use = {}
            for entity in elem['entities']:
                for pos in range(entity['start'], entity['end'], 1):
                    if pos not in in_use:
                        in_use[pos] = entity
                    elif not same_pos(in_use[pos], entity):
                        any_overlay = True
            any_overlays += any_overlay
            split_any_overlays += any_overlay
            for entity in elem['entities']:
                beginning_inside_this = [
                    other for other in elem['entities']
                    if (
                        entity['start'] <= other['start'] < entity['end']
                        and not same_pos(entity, other)
                    )
                ]
                if len(beginning_inside_this) != 0:
                    any_nested = True

            any_nesteds += any_nested
            split_any_nesteds += any_nested
        print(split)
        print(split_any_nesteds, split_any_nesteds / len(elements))
        print(split_any_overlays, split_any_overlays / len(elements))

    print(any_total, dataset.name)
    print(any_overlays, any_overlays / any_total)
    print(any_nesteds, any_nesteds / any_total)
    pass


nested_stats()
