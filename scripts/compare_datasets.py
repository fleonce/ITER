from pathlib import Path

import torch

from iter.datasets.util import SafetensorsDataset
from with_argparse import with_argparse


@with_argparse
def compare_datasets(a: Path, b: Path=None):
    a = SafetensorsDataset.load_from_file(a)
    b = SafetensorsDataset.load_from_file(b)

    for elem_a, elem_b in zip(a, b):
        for k, v in elem_a.items():
            t_a: torch.Tensor = v
            t_b = elem_b[k]
            if t_a.is_sparse:
                t_a = t_a.to_dense()
                t_b = t_b.to_dense()
            assert torch.equal(t_a, t_b)


compare_datasets()
