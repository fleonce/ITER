from collections import defaultdict
from json import load, dumps
from pathlib import Path

import numpy as np
from tqdm import tqdm

cwd = Path.cwd()
lengths_per_split = dict()
for name in ["train", "test", "dev"]:
    with (cwd / "datasets" / "conll03" / f"conll03_{name}.json").open() as f:
       json_blob = load(f)

    lengths = []
    for entry in tqdm(json_blob, leave=False):
        lengths.append(len(entry["sentences"]))
    lengths_per_split[name] = np.array(lengths).mean()

lengths_per_split['avg'] = np.array(list(lengths_per_split.values())).mean()
print(dumps(lengths_per_split, indent=2))
