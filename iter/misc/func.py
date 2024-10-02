import typing

import torch
from torch import Tensor

K = typing.TypeVar("K")
V = typing.TypeVar("V")
U = typing.TypeVar("U")


def map_if(m: dict[K, V], pred: typing.Callable[[V], bool], func: typing.Callable[[V], U], inplace=False):
    if inplace:
        for k, v in m.items():
            if pred(v):
                m[k] = func(v)
    else:
        return {k: func(v) if pred(v) else v for k, v in m.items()}


def prefix_dict(d: dict[str, typing.Any], prefix: str):
    return {prefix + k: v for k, v in d.items()}


def sparse_bool_to_dense_if_sparse(x: torch.Tensor):
    if x.is_sparse:
        return sparse_bool_to_dense(x)
    return x


def sparse_bool_to_dense(x: torch.Tensor):
    return x.int().to_dense().bool()


def batched_index_gen(x: Tensor, *, min_size: typing.Optional[int] = None) -> Tensor:
    max_indices: Tensor = x.to(torch.long).sum(-1).max()
    torch.clamp_min_(max_indices, min_size if min_size is not None else 0)

    bs = x.size(0)
    seq_len = x.size(1)
    indices = torch.arange(seq_len, device=x.device).expand(bs, -1).clone()
    indices.masked_fill_(~x, seq_len)
    top_k, _ = torch.topk(indices, max_indices.item(), -1, False)
    top_k.masked_fill_(top_k.eq(seq_len), -1)
    return top_k
