from typing import Iterable, TypeVar, Optional, Callable, cast
from typing_extensions import Self

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

_T = TypeVar('_T')
_C = TypeVar('_C')


def first(
    iterable: Iterable[_T],
    func: Callable[[_T], _C],
    default: Optional[_C] = None
) -> _C:
    try:
        value = next(iter(iterable))
        return func(value)
    except StopIteration:
        if default is None:
            raise
        return default


class Batch(dict[str, Tensor]):
    def to(self, device) -> Self:
        for k, v in self.items():
            self[k] = v.to(device)
        return self

    def with_prefix(self, prefix: str):
        out = self.__class__()
        for k, v in self.items():
            out[prefix + k] = v
        return out

    def rename(self, key: str, new_key: str):
        copy = Batch(self)
        value = copy.pop(key)
        copy[new_key] = value
        return copy

    def __setattr__(self, key: str, value: Tensor):
        self[key] = value

    def __getattr__(self, item: str) -> Tensor:
        return self[item]


class DataCollatorForITER:
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, dataset):
        if dataset is None:
            raise ValueError("Dataset must be specified")
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset

    def pad_sequence_with_key(self, k: str, tensors: list[Tensor]):
        pad_value = 0
        if k == 'input_ids':
            pad_value = self.tokenizer.pad_token_id
        elif k == 'actions':
            pad_value = 4
        return pad_sequence(tensors, batch_first=True, padding_value=pad_value)

    def pad_pair_flag(self, k: str, seq_len: int, num_l: int, num_r: int, tensors: list[Tensor]):
        bs = len(tensors)
        num_targets = first(tensors, lambda t: t.size(-1))
        dims: tuple[int, ...]
        dims = (bs, seq_len, num_l, num_targets)
        if k == "rr_pair_flag":
            dims = (bs, seq_len, num_r, num_targets)
            if tensors[0].dim() > 3:
                dims = (bs, seq_len, num_l, num_r, num_l, num_targets)
        tensor = torch.stack(tensors, dim=0).coalesce()
        tensor = torch.sparse_coo_tensor(tensor.indices(), tensor.values(), size=dims)
        return tensor.int().to_dense().bool()

    def __call__(self, inp: list[dict[str, Tensor]]):
        bs = len(inp)
        batch = Batch()
        keys = inp[0].keys()
        for k in keys - {'lr_pair_flag', 'rr_pair_flag'}:
            tensors = [inp[i][k] for i in range(bs)]
            if tensors[0].dim() == 0 or tensors[0].numel() == 1:
                batch[k] = torch.stack(tensors, dim=0)
            elif tensors[0].dim() == 1:
                batch[k] = self.pad_sequence_with_key(k, tensors)
        seq_len = batch["input_ids"].size(1)
        actions = batch["actions"]
        num_l = int((actions & (1 << 0)).ne(0).sum(dim=-1).max())  # calc num_l from actions + input_actions
        num_r = int((actions & (1 << 1)).ne(0).sum(dim=-1).max())
        for k in keys & {'lr_pair_flag', 'rr_pair_flag'}:
            tensors = [inp[i][k] for i in range(bs)]
            batch[k] = self.pad_pair_flag(k, seq_len, num_l, num_r, tensors)

        if "attention_mask" not in batch:
            batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id)

        return batch
