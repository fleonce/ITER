from dataclasses import dataclass
from typing import Optional

from torch import Tensor
from torch.nn import Module, Linear
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from iter import ITERConfig
from iter.modeling_features import FeaturesMixin
from iter.utils import NamedTuple2


class FFN(Module):
    gate: Linear
    wi: Linear
    wo: Linear
    act: Module

    def __init__(self, config: ITERConfig, in_channels: int = 1, out_channels: int | str = 1): ...

    def forward(self, x: Tensor) -> Tensor: ...

    __call__ = FFN.forward


class IsLeft(FFN):
    __call__ = IsLeft.forward

class IsSpan(FFN):
    __call__ = IsSpan.forward

class IsLink(FFN):
    __call__ = IsLink.forward

@dataclass
class ITERBaseOutput(NamedTuple2):
    loss: Tensor
    hidden_state: Tensor
    l_loss: Tensor
    lr_loss: Tensor

@dataclass
class ITEROutput(ITERBaseOutput):
    rr_loss: Tensor

@dataclass
class ITERGenerateBaseCache(NamedTuple2):
    closest_left_indices: Tensor
    closest_left_mask: Tensor
    closest_left_hidden: Tensor

@dataclass
class ITERGenerateBaseOutput(NamedTuple2):
    actions: Tensor
    lr_pair_flag: Tensor
    cache: Optional[ITERGenerateBaseCache]

@dataclass
class ITERGenerateOutput(ITERGenerateBaseOutput):
    rr_pair_flag: Tensor
    rr_pair_probabilities: Tensor

@dataclass
class ITERGenerationOutput(NamedTuple2):
    actions: Tensor
    lr_pair_flag: Tensor
    rr_pair_flag: Tensor
    rr_pair_probabilities: Optional[Tensor]
    entities: Optional[list]
    links: Optional[list]

class ITER(PreTrainedModel, FeaturesMixin):
    is_l: IsLeft
    lr_score: IsSpan
    config: ITERConfig

    def __init__(self, config: ITERConfig): ...

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
    ) -> ITERBaseOutput: ...

    def generate(
        self,
        inputs: Optional[Tensor] = None,
        **kwargs,
    ) -> ITERGenerationOutput: ...

    def _post_generate_decoding(
        self,
        inputs: Tensor,
        generation_output: ITERGenerateBaseOutput,
        decode: Optional[bool] = None,
    ) -> ITERGenerationOutput: ...

    def resize_num_types(self, new_num_types: int): ...

    def _generate_forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        return_cache: bool = False,
    ) -> ITERGenerateBaseOutput: ...

    def _is_left_forward(
        self,
        x: Tensor,
        actions: Tensor
    ) -> Tensor: ...

    def _is_span_forward(
        self,
        x: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
    ) -> Tensor: ...

    def _merge_representations(self, head: Tensor, tail: Tensor): ...

class ITERForRelationExtraction(ITER):
    rr_score: IsLink
    num_links: int
    threshold: float

    def __init__(self, config: ITERConfig): ...

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
        rr_pair_flag: Optional[Tensor] = None,
    ) -> ITEROutput: ...

    def resize_num_links(self, new_num_links: int): ...

    def _is_link_forward(
        self,
        x: Tensor,
        actions: Tensor,
        rr_pair_flag: Tensor
    ) -> Tensor: ...

    def generate(
        self,
        inputs: Optional[Tensor] = None,
        **kwargs,
    ) -> ITERGenerationOutput: ...

    def _post_generate_decoding(
        self,
        inputs: Tensor,
        generation_output: ITERGenerateBaseOutput,
        decode: Optional[bool] = None,
    ) -> ITERGenerationOutput: ...

    def _generate_forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        return_cache: Optional[bool] = False,
    ) -> ITERGenerateOutput: ...

    def _merge_link_representations(self, head: Tensor, tail: Tensor): ...