import json
from collections import defaultdict, OrderedDict
from pathlib import Path

from tqdm import tqdm
from with_argparse import with_argparse


@with_argparse
def most_frequent_ne(conll_files: list[Path], topk: int = 5):
    for conll_file in conll_files:
        counts_by_ne = defaultdict(lambda: defaultdict(int))

        with conll_file.open() as f:
            conll_json = json.load(f)
        for entry in tqdm(conll_json):
            words = entry["tokens"]
            for entity in entry["entities"]:
                entity_text = ' '.join(words[entity["start"]:entity["end"]])
                counts_by_ne[entity["type"]][entity_text] += 1

        counts_by_ne = OrderedDict(counts_by_ne)
        for ne, counts_by_span in counts_by_ne.items():
            counts_by_span = [
                (span, count)
                for span, count in counts_by_span.items()
            ]
            counts_by_span.sort(key=lambda x: x[1], reverse=True)
            print(f"named entity = %s" % ne)
            for i, (span, count) in enumerate(counts_by_span[:topk]):
                print("\t (%d) freq = %d\t span = %s" % (i + 1, count, span))


most_frequent_ne()
