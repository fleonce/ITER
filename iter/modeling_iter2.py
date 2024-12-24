import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torchcheck
from torch import Tensor
from torch.nn import Dropout, Linear, Module
from torchcheck import batched_index_padded, assert_true
from torchcheck.ops import expand_dim, inclusive_cumsum, unsqueeze_dims, unsqueeze_and_expand_dims
from transformers import PreTrainedModel, PreTrainedTokenizerBase, add_start_docstrings
from transformers.activations import ACT2FN

from iter import ITERConfig
from iter.generation_utils import decode_actions_and_pairings
from iter.modeling_features import FeaturesMixin
from iter.utils import NamedTuple2

VERIFY_GENERATION = os.environ.get("ITER_VERIFY_GENERATION", "false") == "true"
NEG_INF = -20000

class FFN(Module):
    gate: Linear
    wi: Linear
    wo: Linear
    act: Module

    def __init__(self, config: ITERConfig, in_channels: int = 1, out_channels: int | str = 1):
        super().__init__()
        self.config = config
        self.dropout = Dropout(config.dropout)
        inner_scale = (config.use_scale + 1) if in_channels > 1 else 1
        in_channels = in_channels // (1 + config.is_feature_sum_representations)
        out_channels = out_channels if isinstance(out_channels, int) else getattr(config, out_channels)
        self.gate = Linear(
            in_channels * config.d_model,
            inner_scale * config.d_ff,
            False
        ) if config.use_gate else None
        self.wi = Linear(in_channels * config.d_model, inner_scale * config.d_ff, False)
        self.wo = Linear(inner_scale * config.d_ff, out_channels, False)
        self.act = ACT2FN[config.activation_fn]

    def resize(self, size: int):
        d_out, d_in = self.wo.weight.size()
        if size == d_out:
            return
        wo = self.wo
        self.wo = Linear(d_in, size, bias=False, device=wo.weight.device, dtype=wo.weight.dtype)
        copy = min(size, wo.weight.size(0))
        if copy > 0:
            self.wo.weight[:copy, :] = wo.weight[:copy, :]

    def forward(self, x: Tensor) -> Tensor:
        if self.gate is None:
            x = self.wi(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.wo(x)
        else:
            gate = self.gate(x)
            gate = self.act(gate)
            x = self.wi(x)
            x = gate * x
            x = self.dropout(x)
            x = self.wo(x)
        return x


class IsLeft(FFN):
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Representation the input of shape [B, N, H]
                        there B is the batch dimension, N the sequence length and H the hidden dimension size

        Returns: A Tensor of shape [B, N, 1], where each logit `> 0` indicates the beginning of a span
        """
        return super().forward(x)


class IsSpan(FFN):
    def __init__(self, config: ITERConfig):
        super().__init__(config, in_channels=2, out_channels="num_types")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Representation the input of shape [B, N, H]
                        where B is the batch dimension, N the sequence length and H the hidden dimension size

        Returns: A Tensor of shape [B, N, num_types], where each logit `> 0` indicates the end of a span.
        The type of the span is inferred from the argmax:

        >>> types = self.lr_score(x).argmax(dim=-1)
        """
        return super().forward(x)


class IsLink(FFN):
    def __init__(self, config: ITERConfig):
        num_links = config.num_links * (1 + config.is_feature_negsample_link_types)
        super().__init__(config, in_channels=2, out_channels=num_links)


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
    rr_pair_flag: Optional[Tensor]
    rr_pair_probabilities: Optional[Tensor]
    entities: Optional[list]
    links: Optional[list]


def mask_as_one_hot(mask: Tensor) -> Tensor:
    torchcheck.assert_dtype(mask, dtype=torch.bool)
    return F.one_hot(mask.long(), num_classes=2)


ITER_ACTION_LEFT_BRACKET_BIT = 0
ITER_ACTION_RIGHT_BRACKET_BIT = 1
ITER_ACTION_PAD_BIT = 2


def is_action(actions: Tensor, bit: int, as_one_hot: bool = False, dtype: Optional[torch.dtype] = None) -> Tensor:
    x = actions.bitwise_and(1 << bit).ne(0)
    if as_one_hot:
        x = mask_as_one_hot(x)
    if dtype is not None:
        x = x.to(dtype)
    return x


class ITER(PreTrainedModel, FeaturesMixin):
    config_class = ITERConfig
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    num_types: int
    num_links: int
    max_nest_depth: int

    def __init__(
        self,
        config: ITERConfig,
    ):
        super().__init__(config)
        model_cls = config.guess_model_class()
        model_kwargs = config.model_kwargs
        self.model = model_cls.from_pretrained(
            config.transformer_config.name_or_path, **model_kwargs,
        )
        self.config = config
        self.tokenizer = config.guess_tokenizer_class(True).from_pretrained(
            config.transformer_config.name_or_path,
            model_max_length=self.config.max_length,
            **config.tokenizer_kwargs(use_fast=True)
        )
        self.num_types = config.num_types
        self.max_nest_depth = config.max_nest_depth
        self.features = config.features

        self.is_l = IsLeft(self.config)
        self.lr_score = IsSpan(self.config)

    def resize_num_types(self, new_num_types: int):
        self.lr_score.resize(new_num_types)
        self.num_types = new_num_types

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
    ) -> ITERBaseOutput:
        """
        Args:
            input_ids (Tensor): The tokens the model receives as input of shape [B, N]
            attention_mask (Tensor): The attention mask corresponding to the ``input_ids``
            actions (Tensor): Target actions for each position in a sequence. Shape: [B, N]
                There are two actions the model can perform at a given position: left `[` and right `]`
                bracket pairings. Each action is encoded by a different bit in ``actions`` being set: bit ``0`` for
                left and bit ``1`` for right bracket pairings.
            lr_pair_flag (Tensor): A Boolean tensor of shape [B, N, num_left, num_types], where a `True` value at
                position (b, 1<=n<=N, l, t) indicates a pairing between the l-th bracket and position n of type t.

        Returns: A tuple (loss, hidden state, l_loss, lr_loss)
        """
        # [B, N, H]
        x = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )[0]

        lr_loss = self._is_span_forward(
            x,
            actions,
            lr_pair_flag
        )
        l_loss = self._is_left_forward(x, actions)

        loss = l_loss + lr_loss
        if self.is_feature_batch_average_loss:
            loss = loss / input_ids.size(0)
        return ITERBaseOutput(loss, x, l_loss, lr_loss)

    def generate(
            self,
            inputs: Optional[Tensor] = None,
            **kwargs,
    ) -> ITERGenerationOutput:
        attention_mask = kwargs.get("attention_mask")
        decode_output = kwargs.get("decode_output", True)

        x = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            return_dict=False
        )[0]

        generation_output = self._generate_forward(
            x,
            attention_mask,
        )

        return self._post_generate_decoding(
            inputs,
            generation_output,
            decode_output
        )

    def _post_generate_decoding(
        self,
        inputs: Tensor,
        generation_output: ITERGenerateBaseOutput,
        decode: Optional[bool] = None,
    ) -> ITERGenerationOutput:
        actions = generation_output.actions
        lr_pair_flag = generation_output.lr_pair_flag
        rr_pair_flag = None
        rr_pair_probabilities = None

        entities, links = None, None
        if hasattr(generation_output, "rr_pair_flag"):
            rr_pair_flag = generation_output.rr_pair_flag
            if hasattr(generation_output, "rr_pair_probabilities"):
                rr_pair_probabilities = generation_output.rr_pair_probabilities
            else:
                rr_pair_probabilities = rr_pair_flag.to(torch.float)

        if decode:
            entities, links = decode_actions_and_pairings(
                self,
                inputs,
                actions,
                lr_pair_flag,
                rr_pair_flag,
                rr_pair_probabilities,
                self.config.entity_types,
                self.config.link_types,
            )

        return ITERGenerationOutput(
            actions,
            lr_pair_flag,
            rr_pair_flag,
            rr_pair_probabilities,
            entities,
            links
        )

    def _generate_forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        return_cache: bool = False,
    ) -> ITERGenerateBaseOutput:
        batch_size, seq_len, hidden = x.size()
        nested_inference = self.is_feature_nest_depth_gt_1
        attention_mask = attention_mask.bool()

        # [B, N]
        logits = self.is_l(x).squeeze(-1)
        indices = inclusive_cumsum(attention_mask.long(), 1)

        is_left = torch.gt(logits, 0) & attention_mask
        # [B, num_l]
        num_left = is_left.sum(dim=1).amax().clamp_min(self.max_nest_depth)
        is_left_positions = x.new_empty((batch_size, num_left), dtype=torch.long)
        is_left_positions, is_left_mask = batched_index_padded(is_left, 0, out=is_left_positions, return_mask=True)

        if not nested_inference:
            # [B, N]
            closest_left_indices = torch.cumsum(is_left, dim=1) - 1
            closest_left_mask = torch.ge(closest_left_indices, 0) & attention_mask
            closest_left_indices.masked_fill_(~closest_left_mask, 0)

            # [B, num_l]
            closest_left_positions = torch.gather(
                is_left_positions,
                dim=1,
                index=closest_left_indices
            ).masked_fill(~closest_left_mask, 0)
        else:
            # [B, N, nest_depth, dim]
            x = expand_dim(x.unsqueeze(2), 2, self.max_nest_depth)

            # [B, N, num_l]
            distance_to_previous_left = indices.unsqueeze(2) - is_left_positions.unsqueeze(1)
            distance_to_previous_left.masked_fill_(~is_left_mask.unsqueeze(1), -1)
            distance_to_previous_left_mask = distance_to_previous_left >= 0
            distance_to_previous_left_mask &= attention_mask.unsqueeze(2).bool()
            distance_to_previous_left.masked_fill_(~distance_to_previous_left_mask, torch.iinfo(torch.long).max)

            # [B, N, nest_depth]
            closest_left_indices = torch.topk(
                distance_to_previous_left,
                k=self.max_nest_depth,
                dim=2,
                largest=False
            )[1]
            # ... replace indices 0..num_l by their positions 0..N
            closest_left_positions = torch.gather(
                expand_dim(is_left_positions.unsqueeze(1), 1, seq_len),
                dim=2,
                index=closest_left_indices
            )
            # ... and copy their mask
            closest_left_mask = torch.gather(
                distance_to_previous_left_mask,
                dim=2,
                index=closest_left_indices
            )

        # [B, N, dim] | [B, N, nest_depth, dim]
        is_left_hidden = torch.gather(
            x,
            dim=1,
            index=unsqueeze_and_expand_dims(
                closest_left_positions,
                (2 + nested_inference,),
                (2 + nested_inference, hidden)
            )
        )

        # [B, N, 2*dim | dim] | [B, N, nest_depth, 2*dim | dim]
        logits = self._merge_representations(is_left_hidden, x)
        # [B, N, num_types] | [B, N, nest_depth, num_types]
        logits = self.lr_score(logits)

        float_inf = float("-inf")
        logits.masked_fill_(~closest_left_mask.unsqueeze(2 + nested_inference), float_inf)
        logits.masked_fill_(~closest_left_mask.unsqueeze(2 + nested_inference), float_inf)
        logits.masked_fill_(~unsqueeze_dims(attention_mask, (2,) * (1 + nested_inference)).bool(), float_inf)

        if nested_inference:
            # [B, N, nest_depth, 1]
            compare_to = logits[..., [-1]]
            # [B, N, nest_depth, num_types - 1]
            denominator = logits[..., :-1] > compare_to
            # [B, N, nest_depth]
            denominator = denominator.any(dim=3)
            # [B, N]
            is_right = denominator.any(dim=2)
        else:
            # [B, N]
            denominator = logits.logsumexp(dim=2, keepdim=False)
            is_right = torch.gt(denominator, 0)

        # [B, N]
        actions = (is_left + is_right * 2) * attention_mask

        # (1,)
        num_left = is_left_positions.size(1)
        # [B, N, num_left, num_types]
        lr_pair_flag = x.new_zeros(
            (batch_size, seq_len, num_left, self.num_types),
            dtype=torch.bool
        )
        if nested_inference:
            lr_pair_flag[..., -1] = True

        if not nested_inference:
            # [B, N, 1, 1]
            scatter_indices = unsqueeze_dims(closest_left_indices, 2, 3)
        else:
            # [B, N, nest_depth, 1]
            scatter_indices = unsqueeze_dims(closest_left_indices, 3)
        scatter_indices = expand_dim(scatter_indices, 3, self.num_types)

        # [B, N, 1] | [B, N, nest_depth]
        one_hot_type_input = logits.argmax(dim=2 + nested_inference, keepdim=not nested_inference)
        if nested_inference:
            one_hot_type_input = one_hot_type_input.masked_fill_(~closest_left_mask, self.num_types - 1)
        # [B, N, 1 | nest_depth, num_types]
        one_hot_type_choice = F.one_hot(one_hot_type_input, num_classes=self.num_types).to(torch.bool)
        if not nested_inference:
            one_hot_type_choice = one_hot_type_choice * logits.logsumexp(dim=2, keepdim=True).gt(0).unsqueeze(3)

        # perform the scatter into lr_pair_flag, which we use to decode the output
        lr_pair_flag = torch.scatter(
            lr_pair_flag,
            dim=2,
            index=scatter_indices,
            src=one_hot_type_choice
        )
        cache = None
        if return_cache:
            cache = ITERGenerateBaseCache(closest_left_indices, closest_left_mask, is_left_hidden)
        return ITERGenerateBaseOutput(actions, lr_pair_flag, cache)

    def _is_left_forward(
        self,
        x: Tensor,
        actions: Tensor
    ) -> Tensor:
        attention_mask = ~is_action(actions, ITER_ACTION_PAD_BIT)
        # [B, N, 1]
        is_left = self.is_l(x)
        no_action = torch.zeros_like(is_left)
        # [B, N, 2]
        is_left = torch.cat((no_action, is_left), dim=2)

        # [B, N]
        numerator = torch.logsumexp(
            is_left + torch.where(
                is_action(actions, ITER_ACTION_LEFT_BRACKET_BIT, True, torch.bool),
                0.,
                torch.finfo(x.dtype).min
            ),
            dim=2
        ) * attention_mask
        denominator = torch.logsumexp(is_left, dim=2) * attention_mask
        return denominator.sub(numerator).mul(attention_mask).sum()

    def _is_span_forward(
        self,
        x: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
    ) -> Tensor:
        bs, seq_len, dim = x.shape
        num_l = lr_pair_flag.size(2)
        device = x.device

        # [B, N]
        attention_mask = ~is_action(actions, ITER_ACTION_PAD_BIT)
        is_left = is_action(actions, ITER_ACTION_LEFT_BRACKET_BIT)
        is_right = is_action(actions, ITER_ACTION_RIGHT_BRACKET_BIT)

        # [B, num_l]
        left_positions = x.new_empty((bs, num_l), dtype=torch.long)
        left_positions, left_mask = batched_index_padded(is_left, 0, out=left_positions, return_mask=True)

        # [B, N, num_l]
        distance_to_previous_left = inclusive_cumsum(attention_mask.long(), 1).unsqueeze(2) - left_positions.unsqueeze(1)
        is_after_or_at_left = distance_to_previous_left >= 0
        is_after_or_at_left = is_after_or_at_left & attention_mask.unsqueeze(2) & left_mask.unsqueeze(1)

        nest_depth = min(self.max_nest_depth, num_l)
        if nest_depth == 0:
            return x.new_zeros(tuple(), dtype=x.dtype)

        # [B, N, nest_depth]
        closest_left_weights: Tensor
        closest_left_indices: Tensor
        closest_left_weights, closest_left_indices = torch.where(
            (distance_to_previous_left >= 0) & is_after_or_at_left,
            distance_to_previous_left,
            torch.iinfo(distance_to_previous_left.dtype).max
        ).topk(nest_depth, dim=2, largest=False)
        closest_left_mask = closest_left_weights < seq_len

        # [B, N, nest_depth]
        closest_left_positions = torch.gather(
            expand_dim(left_positions.unsqueeze(1), 1, seq_len),
            dim=2,
            index=closest_left_indices
        )
        # [B, N, nest_depth]
        closest_left_is_after_or_at_left = torch.gather(
            is_after_or_at_left,
            dim=2,
            index=closest_left_indices
        )
        # [B, N, nest_depth, num_types]
        closest_left_lr_pair_flag = torch.gather(
            lr_pair_flag,
            dim=2,
            index=expand_dim(closest_left_indices.unsqueeze(3), 3, self.num_types)
        ) * closest_left_mask.unsqueeze(3)

        x = expand_dim(x.unsqueeze(2), 2, nest_depth)
        # prepare closest_left_positions such that we can gather from x in dim=1
        # [B, N, nest_depth, H]
        closest_left_hidden = torch.gather(
            x,
            dim=1,
            index=expand_dim(closest_left_positions.unsqueeze(3), 3, dim).clamp_min(0)
        )

        logits = self._merge_representations(closest_left_hidden, x)
        # [B, N, nest_depth, num_types]
        logits = self.lr_score(logits)
        logits = logits + (~closest_left_is_after_or_at_left.unsqueeze(3) * NEG_INF)

        mask = closest_left_is_after_or_at_left.any(dim=2, keepdim=True)
        nested_training = self.is_feature_nest_depth_gt_1
        if nested_training:
            assert_true(self.is_feature_extra_lr_class, "For datasets with nest_depth > 1, you need to add an "
                                                        "additional NONE class to the num_types dimension.")
            mask = mask & closest_left_lr_pair_flag.any(dim=3)
            denominator = torch.logsumexp(
                logits,
                dim=3
            ) * mask
            numerator = torch.logsumexp(
                logits + (~closest_left_lr_pair_flag * NEG_INF),
                dim=3
            ) * mask

            denominator = denominator.unsqueeze(3)
            numerator = numerator.unsqueeze(3)
        else:
            denominator = torch.logsumexp(
                logits,
                dim=(2, 3)
            ).unsqueeze(2) * mask
            numerator = torch.logsumexp(
                logits + (~closest_left_lr_pair_flag * NEG_INF),
                dim=(2, 3)
            ).unsqueeze(2) * mask

        no_action = torch.zeros_like(denominator)
        denominator = torch.logsumexp(
            torch.cat((no_action, denominator), dim=2 + nested_training),
            dim=2 + nested_training
        )

        numerator = torch.logsumexp(
            torch.cat((
                no_action + torch.where(
                    ~unsqueeze_dims(is_right, (2,) * (1 + nested_training)),
                    0,
                    torch.finfo(no_action.dtype).min
                ),
                numerator
            ), dim=2 + nested_training),
            dim=2 + nested_training
        )
        sub_mask = attention_mask
        if nested_training:
            sub_mask = attention_mask.unsqueeze(2)

        return denominator.sub(numerator).mul(sub_mask).sum()

    def _merge_representations(self, head: Tensor, tail: Tensor):
        if self.is_feature_sum_representations:
            raise NotImplementedError

        torch._check(head.dim() == tail.dim())
        return torch.cat((head, tail), dim=-1)


class ITERForRelationExtraction(ITER):
    def __init__(self, config: ITERConfig):
        random_state = torch.get_rng_state()
        super().__init__(config)
        torch.set_rng_state(random_state)

        self.threshold = config.threshold
        self.num_links = config.num_links

        model_cls = config.guess_model_class()
        model_kwargs = config.model_kwargs
        self.model = model_cls.from_pretrained(
            config.transformer_config.name_or_path, **model_kwargs,
        )
        self.is_l = IsLeft(config)
        self.rr_score = IsLink(config)
        self.lr_score = IsSpan(config)

    def resize_num_links(self, new_num_links: int):
        self.rr_score.resize(new_num_links)
        self.num_links = new_num_links

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
        rr_pair_flag: Tensor = None,
    ) -> ITEROutput:
        """
        Args:
            input_ids (Tensor): The tokens the model receives as input of shape [B, N]
            attention_mask (Tensor): The attention mask corresponding to the ``input_ids``
            actions (Tensor): Target actions for each position in a sequence. Shape: [B, N]
                There are two actions the model can perform at a given position: left `[` and right `]`
                bracket pairings. Each action is encoded by a different bit in ``actions`` being set: bit ``0`` for
                left and bit ``1`` for right bracket pairings.
            lr_pair_flag (Tensor): A Boolean tensor of shape [B, N, num_left, num_types], where a `True` value at
                position (b, 1<=n<=N, l, t) indicates a pairing between the l-th bracket and position n of type t.
            rr_pair_flag (Tensor): A Boolean tensor of shape [B, N, num_right, num_links], where a `True` value at
                position (b, 1<=n<=N, r, t) indicates a link between the r-th span and the span ending at n of type t.

        Returns: A tuple (loss, l_loss, lr_loss, rr_loss)
        """
        # [B, N, H]
        x = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )[0]

        lr_loss = self._is_span_forward(
            x,
            actions,
            lr_pair_flag
        )
        rr_loss = self._is_link_forward(
            x,
            actions,
            rr_pair_flag
        )
        l_loss = self._is_left_forward(x, actions)
        loss = l_loss + lr_loss + rr_loss
        if self.is_feature_batch_average_loss:
            loss = loss / input_ids.size(0)
        return ITEROutput(loss, x, l_loss, lr_loss, rr_loss)

    def _merge_link_representations(self, head: Tensor, tail: Tensor):
        torch._check(head.dim() == tail.dim())
        return torch.cat((head, tail), dim=-1)

    def _is_link_forward(
        self,
        x: Tensor,
        actions: Tensor,
        rr_pair_flag: Tensor
    ) -> Tensor:
        assert not self.is_feature_ner_only
        if self.is_feature_use_lse_rr_loss:
            raise NotImplementedError()

        nested_training = self.is_feature_nest_depth_gt_1
        bs, seq_len, dim = x.shape
        attention_mask = actions.bitwise_and(4).eq(0)
        is_right = actions.bitwise_and(2).ne(0)
        is_left = actions.bitwise_and(1).ne(0)

        is_right_positions, is_right_mask = batched_index_padded(is_right, 0, return_mask=True)
        if is_right_positions.size(1) == 0:
            return x.new_zeros(tuple())

        # [B, num_right, dim]
        is_right_hidden = torch.gather(
            x,
            dim=1,
            index=unsqueeze_and_expand_dims(
                is_right_positions,
                (2,), (2, dim)
            )
        ) * is_right_mask.unsqueeze(2)

        if nested_training:
            assert self.is_feature_extra_lr_class

            is_left_positions, is_left_mask = batched_index_padded(is_left, 0, return_mask=True)

            distance_to_previous_left = inclusive_cumsum(attention_mask.long(), 1).unsqueeze(2) - is_left_positions.unsqueeze(1)
            distance_to_previous_left.masked_fill_(~is_left_mask.unsqueeze(1), -1)

            is_after_or_at_left = distance_to_previous_left >= 0
            is_after_or_at_left = is_after_or_at_left & attention_mask.unsqueeze(2) & is_left_mask.unsqueeze(1)

            nest_depth = min(self.max_nest_depth, is_left_positions.size(1))
            if nest_depth == 0:
                return x.new_zeros(tuple())
            num_right = is_right_positions.size(1)
            num_left = is_left_positions.size(1)

            # [B, N, nest_depth]
            closest_left_weights: Tensor
            closest_left_indices: Tensor
            closest_left_weights, closest_left_indices = torch.where(
                (distance_to_previous_left >= 0) & is_after_or_at_left,
                distance_to_previous_left,
                distance_to_previous_left + (torch.iinfo(distance_to_previous_left.dtype).max >> 4)
            ).topk(nest_depth, dim=2, largest=False)
            closest_left_mask = closest_left_weights < seq_len

            # [B, N, nest_depth]
            closest_left_positions = torch.gather(
                expand_dim(is_left_positions.unsqueeze(1), 1, seq_len),
                dim=2,
                index=closest_left_indices
            )

            x = expand_dim(x.unsqueeze(2), 2, nest_depth)
            # prepare closest_left_positions such that we can gather from x in dim=1
            # [B, N, nest_depth, H]
            closest_left_hidden = torch.gather(
                x,
                dim=1,
                index=expand_dim(closest_left_positions.unsqueeze(3), 3, dim).clamp_min(0)
            )

            # [B, N, nest_depth, H]
            # for 0 <= n <= N: out[n] = x[n] + x[closest_left[x]]
            closest_span_hidden = torch.add(x, closest_left_hidden)

            # [B, num_right, nest_depth]
            is_right_closest_left_mask = torch.gather(
                closest_left_mask,
                dim=1,
                index=unsqueeze_and_expand_dims(is_right_positions, 2, (2, nest_depth))
            )
            is_right_closest_left_indices = torch.gather(
                closest_left_indices,
                dim=1,
                index=unsqueeze_and_expand_dims(is_right_positions, 2, (2, nest_depth))
            )
            # [B, num_right, nest_depth, H]
            is_right_closest_left_hidden = torch.gather(
                closest_span_hidden,
                dim=1,
                index=unsqueeze_and_expand_dims(
                    is_right_positions,
                    (2, 3),
                    (2, nest_depth), (3, dim),
                )
            )

            if rr_pair_flag.size(3) == 0:
                return x.new_zeros(tuple())
            # rr_pair_flag has dim [B, N, num_left, num_right, num_left, T]
            # dim0 = B
            # dim2 = num_left (sequence)
            # dim4 = num_left (span)
            # [B, N, nest_depth, num_right, num_left, T]
            num_left_to_nest_depth_index = unsqueeze_and_expand_dims(
                # [B, N, nest_depth] -> [B, N, nest_depth, num_right, num_left, T]
                closest_left_indices,
                (3, 4, 5),
                (3, num_right), (4, num_left), (5, self.num_links),
            )
            rr_pair_flag = torch.gather(
                rr_pair_flag,
                dim=2,
                index=num_left_to_nest_depth_index
            )
            # [B, N, nest_depth, num_right, nest_depth, T]
            # here is the problem
            rr_pair_flag = torch.gather(
                rr_pair_flag,
                dim=4,
                index=unsqueeze_and_expand_dims(
                    is_right_closest_left_indices,
                    (1, 2, 5),
                    (1, seq_len), (2, nest_depth), (5, self.num_links)
                )
            )

            # [B, N, nest_depth, num_right, nest_depth, 2*dim]
            logits = self._merge_link_representations(
                unsqueeze_and_expand_dims(is_right_closest_left_hidden, (1, 2,), (1, seq_len), (2, nest_depth)),
                unsqueeze_and_expand_dims(closest_span_hidden, (3, 4), (3, num_right), (4, nest_depth))
            )
            # [B, N, nest_depth, num_right, nest_depth, T]
            logits = self.rr_score(logits)
            # [B, N, is_right]
            logits_mask = is_right_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
            # closest_left_mask: [B, N, nest_depth]
            # is_right_closest_left_mask: [B, num_right, nest_depth]
            logits_mask = unsqueeze_dims(logits_mask, 2, 4)
            logits_mask = logits_mask & unsqueeze_dims(closest_left_mask, 3, 4) & unsqueeze_dims(is_right_closest_left_mask, 1, 2)
        else:
            num_r = rr_pair_flag.size(2)

            # [B, N, num_right, 2*dim]
            logits = self._merge_link_representations(
                unsqueeze_and_expand_dims(is_right_hidden, (1,), (1, seq_len)),
                unsqueeze_and_expand_dims(x, (2,), (2, num_r))
            )
            # [B, N, num_right, num_links]
            logits = self.rr_score(logits)
            logits_mask = is_right_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
            if self.is_feature_in_development:
                logits_mask = is_right_mask.unsqueeze(1) & is_right.unsqueeze(2)

        if not logits_mask.any():
            return logits.new_zeros(tuple())

        input = logits[logits_mask]
        target = rr_pair_flag[logits_mask].float()
        if self.is_feature_extra_rr_class:
            return F.cross_entropy(input, target, reduction="sum")
        else:
            return F.binary_cross_entropy_with_logits(input, target, reduction="sum")

    def _generate_forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        return_cache: Optional[bool] = False,
    ) -> ITERGenerateOutput:
        base_output = super()._generate_forward(x, attention_mask, return_cache=True)

        bs, seq_len, dim = x.shape
        nested_inference = self.is_feature_nest_depth_gt_1
        assert not nested_inference or self.is_feature_extra_lr_class

        # [B, N]
        actions = base_output.actions
        is_left = actions.bitwise_and(1).ne(0)
        is_right = actions.bitwise_and(2).ne(0)

        is_left_positions, is_left_mask = batched_index_padded(is_left, 0, return_mask=True)
        num_left = is_left_positions.size(1)

        # [B, num_right]
        is_right_positions, is_right_mask = batched_index_padded(is_right, 0, return_mask=True)
        num_right = is_right_positions.size(1)

        dims = (bs, seq_len, num_right, self.num_links)
        if nested_inference:
            dims = (bs, seq_len, num_left, num_right, num_left, self.num_links)
        rr_pair_flag = x.new_zeros(dims, dtype=torch.bool)
        rr_pair_probabilities = x.new_zeros(dims)

        if num_right <= 0:
            return ITERGenerateOutput(
                base_output.actions,
                base_output.lr_pair_flag,
                None,
                rr_pair_flag,
                rr_pair_probabilities
            )

        # [B, num_r, dim]
        is_right_hidden = torch.gather(
            x,
            dim=1,
            index=expand_dim(is_right_positions.unsqueeze(2), 2, dim),
        )

        closest_left_mask = None
        if nested_inference:
            # [B, num_right, nest_depth, dim]
            is_right_hidden = expand_dim(
                is_right_hidden.unsqueeze(2), 2, self.max_nest_depth
            )
            closest_left_hidden = base_output.cache[2]

            # [B, num_right, nest_depth, dim]
            closest_left_hidden_at_right_positions = torch.gather(
                closest_left_hidden,
                dim=1,
                index=unsqueeze_dims(is_right_positions, 2, 3).expand(-1, -1, self.max_nest_depth, dim)
            )
            is_right_hidden = is_right_hidden + closest_left_hidden_at_right_positions


            # [B, num_right, nest_depth]
            closest_left_mask = torch.gather(
                # [B, seq_len, nest_depth]
                base_output.cache.closest_left_mask,
                dim=1,
                index=unsqueeze_and_expand_dims(
                    is_right_positions,
                    2,
                    (2, self.max_nest_depth)
                )
            )
        pass

        if not nested_inference:
            # [B, 1, num_right, dim]
            is_right_hidden = is_right_hidden

            def proj(inp: Tensor, proj_dim: int):
                return unsqueeze_and_expand_dims(inp, (proj_dim,), (proj_dim, num_right))

            # [B, num_right, num_right, dim]
            inner = proj(is_right_hidden, 1)
            outer = proj(is_right_hidden, 2)
            # [B, num_right, num_right]
            inner_mask = proj(is_right_mask, 1)
            outer_mask = proj(is_right_mask, 2)
        else:
            def proj(inp: Tensor, is_inner: bool = False, is_mask: bool = False):
                if is_inner:
                    return unsqueeze_and_expand_dims(
                        inp,
                        (1, 2),
                        (1, num_right), (2, self.max_nest_depth),
                    )
                else:
                    return unsqueeze_and_expand_dims(
                        inp,
                        (3, 4),
                        (3, num_right), (4, self.max_nest_depth),
                    )
            # [B, num_right, nest_depth, num_right, nest_depth, dim]
            inner = proj(is_right_hidden, True, False)
            outer = proj(is_right_hidden, False, False)
            # [B, num_right, nest_depth, num_right, nest_depth]
            inner_outer_mask = unsqueeze_dims(is_right_mask, 2) * closest_left_mask

            inner_mask = proj(inner_outer_mask, True, True)
            outer_mask = proj(inner_outer_mask, False, True)

        compare_mask = inner_mask & outer_mask
        compare = self._merge_link_representations(inner, outer)
        logits = self.rr_score(compare)
        logits.masked_fill_(~compare_mask.unsqueeze(compare_mask.dim()), float("-inf"))
        if self.is_feature_use_lse_rr_loss or self.is_feature_ce_loss:
            assert self.is_feature_extra_rr_class
            outcome = F.one_hot(logits.argmax(dim=-1), num_classes=self.num_links)
            outcome = outcome[..., :-1]
        else:
            assert not self.is_feature_extra_rr_class
            outcome = logits.sigmoid() > self.threshold

        head_nest, tail_nest = None, None
        head_left_index, tail_left_index = None, None
        # [B, N] | [B, N, nest_depth]
        closest_left_indices = base_output.cache.closest_left_indices
        closest_left_mask = base_output.cache.closest_left_mask

        for indices in outcome.nonzero():  # ugly, causes host<->device synchronization
            link_probability = logits[indices.unbind()].sigmoid()
            if nested_inference:
                batch, head, head_nest, tail, tail_nest, link_type = indices
            else:
                batch, head, tail, link_type = indices

            head_position = is_right_positions[batch, head]
            tail_position = is_right_positions[batch, tail]
            if nested_inference:
                head_left_index = closest_left_indices[batch, head_position, head_nest]
                tail_left_index = closest_left_indices[batch, tail_position, tail_nest]
                if VERIFY_GENERATION:
                    assert closest_left_mask[batch, head_position, head_nest]
                    assert closest_left_mask[batch, tail_position, tail_nest]

            # PL Marker trained their models to predict relationships in both ways
            # ... so head -> link -> tail + tail -> inverse link -> head and in case one is missing,
            # the model still scores correctly because of the set-based F1 eval strategy
            reverse = link_type >= self.num_links
            if reverse:
                if nested_inference:
                    dims = (batch, tail_position, tail_left_index, head, head_left_index, link_type % self.num_links)
                else:
                    dims = (batch, tail_position, head, link_type % self.num_links)
            else:
                if nested_inference:
                    dims = (batch, head_position, head_left_index, tail, tail_left_index, link_type)
                else:
                    dims = (batch, head_position, tail, link_type)
            rr_pair_flag[dims] = True
            rr_pair_probabilities[dims] = link_probability

        return ITERGenerateOutput(
                base_output.actions,
                base_output.lr_pair_flag,
                None,
                rr_pair_flag,
                rr_pair_probabilities
            )
