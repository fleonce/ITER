import dataclasses
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import logsumexp, Tensor
from torch.nn import Module, Linear, LayerNorm, Dropout
from iter.misc.func import batched_index_gen
from transformers import T5EncoderModel, T5TokenizerFast, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput

from .configuration_iter import ITERConfig
from .modeling_features import FeaturesMixin

NEG_INF = -20000


class GatedFFModule(Module):
    gate: Linear
    wi: Linear
    wo: Linear
    dropout: Dropout
    act: Module
    o: Linear

    def __init__(self, config: ITERConfig):
        super().__init__()
        self.config = config
        self.dropout = Dropout(config.dropout)
        self.o = None

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
        if self.o is not None:
            x = self.act(x)
            return self.o(x)
        if self.gate is None:
            w = self.wi(x)
            w = self.act(w)
            w = self.dropout(w)
            return self.wo(w)
        gate = self.act(self.gate(x))
        w = self.wi(x)
        w = w * gate
        w = self.dropout(w)
        return self.wo(w)


class IsL(GatedFFModule):
    def __init__(self, config: ITERConfig):
        super().__init__(config)
        self.gate = Linear(config.d_model, config.d_ff, bias=False) if config.use_gate else None
        self.wi = Linear(config.d_model, config.d_ff, bias=False)
        self.wo = Linear(config.d_ff, 1, bias=False)
        self.act = ACT2FN[config.activation_fn]
        self.o = Linear(config.d_model, 1, bias=False) if not config.use_mlp else None


class IsLR(GatedFFModule):
    def __init__(self, config: ITERConfig):
        super().__init__(config)
        num_types = config.num_types
        scale = 2 if config.use_scale else 1
        input_scale = 2 if not config.is_feature_sum_representations else 1
        self.gate = Linear(input_scale * config.d_model, scale * config.d_ff, bias=False) if config.use_gate else None
        self.wi = Linear(input_scale * config.d_model, scale * config.d_ff, bias=False)
        self.wo = Linear(scale * config.d_ff, num_types, bias=False)
        self.act = ACT2FN[config.activation_fn]
        self.o = Linear(input_scale * config.d_model, num_types, bias=False) if not config.use_mlp else None


class IsRR(GatedFFModule):
    def __init__(self, config: ITERConfig):
        super().__init__(config)
        num_links = config.num_links
        scale = 2 if config.use_scale else 1
        self.out_dim = num_links * (1 + config.is_feature_negsample_link_types)
        self.gate = Linear(2 * config.d_model, scale * config.d_ff, bias=config.use_bias) if config.use_gate else None
        self.wi = Linear(2 * config.d_model, scale * config.d_ff, bias=config.use_bias)
        self.wo = Linear(scale * config.d_ff, self.out_dim, bias=config.use_bias)
        self.act = ACT2FN[config.activation_fn]
        self.o = Linear(2 * config.d_model, self.out_dim, bias=config.use_bias) if not config.use_mlp else None


class SumRR:
    def __init__(self, _config: ITERConfig):
        super().__init__()

    def __call__(self, x: Tensor, y: Tensor):
        return self.forward(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return torch.cat((x, y), dim=-1)


class SumLR(Module):

    def __init__(self, config: ITERConfig):
        super().__init__()
        if config.is_feature_sum_representations:
            self.act = ACT2FN[config.activation_fn]
            self.hi = Linear(config.d_model, config.d_ff, bias=True)
            self.ho = Linear(config.d_ff, config.d_model, bias=True)
            self.ti = Linear(config.d_model, config.d_ff, bias=True)
            self.to_ = Linear(config.d_ff, config.d_model, bias=True)
            self.layer_norm = LayerNorm(config.d_model, eps=1e-6, bias=True)
        else:
            self.act = None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.act is None:
            return torch.cat((x, y), dim=-1)
        # x, y = self.act(x), self.act(y)
        x = self.hi(x)
        x = self.act(x)
        x = self.ho(x)
        y = self.ti(y)
        y = self.act(y)
        y = self.to_(y)
        z = x + y
        z = self.layer_norm(z)
        return z


@dataclasses.dataclass
class ITEROutput:
    loss: Tensor
    l_loss: Tensor = None
    lr_loss: Tensor = None
    rr_loss: Tensor = None

    def as_tuple(self):
        return self.loss.item(), self.l_loss.item(), self.lr_loss.item(), self.rr_loss.item()


class ITER(PreTrainedModel, FeaturesMixin):
    config_class = ITERConfig
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    model: T5EncoderModel
    tokenizer: T5TokenizerFast
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
        self.num_links = config.num_links
        self.max_nest_depth = config.max_nest_depth
        self.verifying = False
        self.features = config.features

        self.is_l = IsL(self.config)
        if not self.is_feature_dont_set_rr_score:
            self.rr_score = IsRR(self.config)
        self.lr_score = IsLR(self.config)
        self.sum = SumLR(self.config)
        self.sum_rr = SumRR(self.config)

        self.rr_loss = F.binary_cross_entropy_with_logits
        if self.is_feature_ce_loss:
            assert self.is_feature_extra_rr_class
            self.rr_loss = F.cross_entropy
        self.threshold = config.threshold
        if self.is_feature_ner_only and not self.is_feature_dont_set_rr_score:
            self.rr_score.requires_grad_(False)

    def hash(self):
        vals = []
        for name, param in self.named_parameters():
            vals.append(param.sum(dtype=torch.float64))
        return torch.stack(vals).mean()

    def resize_num_types(self, new_num_types: int):
        self.lr_score.resize(new_num_types)
        self.num_types = new_num_types

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
    ):
        ner_hidden_state, ere_hidden_state = self.forward_base_model(
            model=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        )
        assert actions is not None
        lr_denom, lr_numer = self.train_forward(
            ner_hidden_state,
            actions,
            lr_pair_flag,
        )
        rr_loss = self.train_forward_rr_loss(
            ere_hidden_state,
            actions,
            rr_pair_flag,
        )
        action_logits = self.is_l(ner_hidden_state)
        no_action = torch.zeros_like(action_logits)
        action_logits = torch.cat((no_action, action_logits), dim=-1)

        is_l = torch.ne(actions & (1 << 0), 0)
        is_r = torch.ne(actions & (1 << 1), 0)
        one_hot = torch.stack((~is_l, is_l), dim=-1)
        numer = logsumexp(
            action_logits + torch.where(
                one_hot,
                0.,
                torch.finfo(action_logits.dtype).min
            ), dim=-1) * attention_mask
        denom = logsumexp(action_logits, dim=-1) * attention_mask

        if lr_denom.dim() > 2 and self.max_nest_depth > 1:  # nested training
            no_action = no_action.unsqueeze(-1).expand(-1, -1, lr_denom.size(-1), -1)
            lr_denom = logsumexp(torch.cat((
                no_action,
                lr_denom.unsqueeze(-1),
            ), dim=-1), dim=-1)

            lr_numer = logsumexp(torch.cat((
                no_action + torch.where(
                    ~is_r.unsqueeze(-1).unsqueeze(-1),
                    0,
                    torch.finfo(no_action.dtype).min
                ),
                lr_numer.unsqueeze(-1)
            ), dim=-1), dim=-1)
            assert not torch.any(((lr_denom - lr_numer) * attention_mask.unsqueeze(-1)) > 1000)
            lr_loss = ((lr_denom - lr_numer) * attention_mask.unsqueeze(-1))
        else:
            lr_denom = logsumexp(torch.cat((
                no_action,
                lr_denom
            ), dim=-1), dim=-1) * attention_mask

            lr_numer = logsumexp(torch.cat((
                no_action + torch.where(
                    ~is_r,
                    0,
                    torch.finfo(no_action.dtype).min
                ).unsqueeze(-1),
                lr_numer
            ), dim=-1), dim=-1) * attention_mask
            # assert not torch.any(((lr_denom - lr_numer) * attention_mask) > 1000)
            lr_loss = ((lr_denom - lr_numer) * attention_mask)

        lr_loss = lr_loss.sum()
        rr_loss = rr_loss.sum()
        loss = l_loss = ((denom - numer) * attention_mask).sum()
        loss = loss + lr_loss
        loss = loss + rr_loss
        loss = loss  # / attention_mask.sum()
        if self.is_feature_batch_average_loss:
            loss = loss / input_ids.size(0)
        return ITEROutput(
            loss=loss,
            l_loss=l_loss,
            lr_loss=lr_loss,
            rr_loss=rr_loss,
        )

    def train_forward(
            self,
            hidden_state: Tensor,
            actions: Tensor,
            lr_pair_flag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        bs, seq_len, dim = hidden_state.shape
        device = hidden_state.device
        # (bs, seq_len)
        indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
        is_pad = torch.eq(actions, 4)
        is_l = torch.ne(actions & (1 << 0), 0)
        is_r = torch.ne(actions & (1 << 1), 0)

        # assert is_r.sum() > 0
        l_indices = batched_index_gen(is_l)
        l_indices_mask = torch.ge(l_indices, 0)
        # (bs, num_r = is_r.sum(dim=-1).max())
        num_r = is_r.sum(dim=-1).max()
        num_l = is_l.sum(dim=-1).max()
        r_indices = batched_index_gen(is_r)
        r_indices_mask = torch.ge(r_indices, 0)
        # (bs, seq_len, num_l)
        distance_to_previous_l = indices.unsqueeze(2) - l_indices.unsqueeze(1)
        # (bs, seq_len, num_l)
        is_after_l = distance_to_previous_l > 0
        is_after_l = is_after_l & (~is_pad.unsqueeze(2)) & (l_indices_mask.unsqueeze(1))
        is_after_or_at_l = distance_to_previous_l >= 0
        is_after_or_at_l = is_after_or_at_l & (~is_pad.unsqueeze(2)) & (l_indices_mask.unsqueeze(1))

        # nest_depth <==> how many brackets to the left a right bracket can look
        nest_depth = min(self.max_nest_depth, l_indices.size(-1))
        if nest_depth == 0:
            # there are no entities in any of the examples in the batch
            # so we simply return a loss of zero for both rr and lr pairings
            denom_numer = torch.zeros_like(actions, dtype=hidden_state.dtype).unsqueeze(-1)
            return denom_numer, denom_numer
        assert nest_depth > 0

        # retrieve the indices of the left brackets [ preceding the token at each position :math:`i`
        # weight according to distance + if i-th token is after or at bracket :math:`i`
        l_weights = -distance_to_previous_l + (is_after_or_at_l * 10000)
        # get the indices of the :math:`nest_depth` nearest left brackets
        l_nearest = l_weights.topk(nest_depth, dim=2)[1]
        # get a mask for l_nearest
        l_nearest_mask = torch.gather(
            l_weights, dim=-1, index=l_nearest
        ) > seq_len  # (bs, seq_len, nest_depth)
        # get the positions of the `nest_depth` nearest left brackets
        l_nearest_pos = torch.gather(
            l_indices.unsqueeze(1).expand(-1, seq_len, -1),
            dim=2,
            index=l_nearest
        )  # (bs, seq_len, nest_depth)
        # get the hidden states for each of the left brackets `l_nearest_pos`
        #  we need to expand the hidden_state from
        #  (bs, seq_len, hidden) => (bs, seq_len, nest_depth, hidden)
        hidden_states = hidden_state.unsqueeze(2).expand(-1, -1, nest_depth, -1)
        #  in the same fashion we have to expand l_nearest_pos:
        #  (bs, seq_len, nest_depth) => (bs, seq_len, nest_depth, hidden)
        l_nearest_pos_hidden = l_nearest_pos.unsqueeze(-1).expand(-1, -1, -1, dim)
        l_nearest_pos_hidden_mask = l_nearest_pos_hidden >= 0
        l_nearest_pos_hidden = torch.where(l_nearest_pos_hidden_mask, l_nearest_pos_hidden, 0)
        # now we can get the hidden states of shape
        # (bs, seq_len, nest_depth, hidden)
        l_nearest_hidden = torch.gather(
            hidden_states,
            dim=1,
            index=l_nearest_pos_hidden
        )
        # in the same way, we transform is_after_or_at_l into kept_is_after_or_at_l
        # (bs, seq_len, nest_depth)
        kept_is_after_or_at_l = torch.gather(
            is_after_or_at_l,
            dim=2,
            index=l_nearest
        )
        # ... and lr_pair_flag into
        # (bs, seq_len, nest_depth, num_types)
        l_nearest_pos_pair_flag = l_nearest.unsqueeze(-1).expand(-1, -1, -1, self.num_types)
        kept_lr_pair_flag = torch.gather(
            lr_pair_flag,
            dim=2,
            index=l_nearest_pos_pair_flag
        ) * l_nearest_mask.unsqueeze(-1)

        # using hidden_states and l_nearest_hidden, we can now obtain all combinations
        #  of the two by concatenating the two into one hidden state
        #  shape: (bs, seq_len, nest_depth, 2 * hidden)
        lr_hidden = self.sum(l_nearest_hidden, hidden_states)  # torch.cat((l_nearest_hidden, hidden_states), dim=-1)
        #  ... and obtain the score for these combinations:
        #  shape: (bs, seq_len, nest_depth, num_types)
        lr_score = self.lr_score(lr_hidden)
        #  ... mask out any invalid combinations using is_after_or_at_l
        lr_score = lr_score + (~kept_is_after_or_at_l.unsqueeze(-1) * NEG_INF)

        # finally, we can compute numerator and denominator:
        #  shape: (bs, seq_len, 1)
        denom, numer = self.train_forward_denom_numer(lr_score, kept_lr_pair_flag, is_after_or_at_l)
        return denom, numer

    def train_forward_denom_numer(
            self,
            lr_score: Tensor,
            kept_lr_pair_flag: Tensor,
            is_after_or_at_l: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.is_feature_nest_depth_gt_1 and self.is_feature_extra_lr_class:
            return self.nested_train_forward_denom_numer(lr_score, kept_lr_pair_flag, is_after_or_at_l)

        denom = torch.logsumexp(
            lr_score, dim=(2, 3), keepdim=False
        ).unsqueeze(-1) * is_after_or_at_l.any(dim=2, keepdim=True)

        f_min = -20000  # `-infinity`
        numer = torch.logsumexp(
            lr_score + (~kept_lr_pair_flag * f_min),
            dim=(2, 3), keepdim=False
        ).unsqueeze(-1) * is_after_or_at_l.any(dim=2, keepdim=True)
        return denom, numer

    def train_forward_rr_loss(
            self,
            hidden_state: Tensor,
            actions: Tensor,
            rr_pair_flag: Tensor,
    ) -> Tensor:
        if self.is_feature_ner_only:
            return torch.zeros(1, device=hidden_state.device, dtype=hidden_state.dtype)

        if self.is_feature_use_lse_rr_loss:
            return self.train_forward_rr_lse_loss(hidden_state, actions, rr_pair_flag)

        if self.is_feature_nest_depth_gt_1 and self.is_feature_extra_lr_class:
            return self.nested_train_forward_rr_loss(actions, hidden_state, rr_pair_flag)
        orig_hidden_state = hidden_state
        bs, seq_len, dim = hidden_state.shape
        # (bs, seq_len)
        is_pad = torch.eq(actions, 4)
        is_r = torch.ne(actions & (1 << 1), 0)

        # (bs, num_r = is_r.sum(dim=-1).max())
        r_indices = batched_index_gen(is_r)
        num_r = r_indices.size(1)
        r_indices_mask = torch.ge(r_indices, 0)

        assert r_indices.size(-1) == is_r.sum(dim=-1).max()
        if num_r == 0:
            return torch.zeros(1, device=hidden_state.device, dtype=hidden_state.dtype)
        assert not self.is_feature_nest_depth_gt_1
        # we are in the easy case!

        # find all right hidden states:
        #  shape: (bs, num_r, 1)
        r_indices = r_indices.masked_fill(~r_indices_mask, 0).unsqueeze(-1)

        # find their corresponding hidden states:
        #  shape: (bs, num_r, hidden)
        hidden_state = torch.gather(
            orig_hidden_state,
            dim=1,
            index=r_indices.expand(-1, -1, dim))
        r_other_hidden_state = hidden_state * r_indices_mask.unsqueeze(-1)

        # shape: (bs, seq_len, num_r, hidden)
        r_hidden_state = orig_hidden_state.unsqueeze(2).expand(-1, -1, num_r, -1)

        # unsqueeze in the seq_len length dim
        #  shape: (bs, 1, num_r, hidden)
        r_other_hidden_state = r_other_hidden_state.unsqueeze(1)
        # expand in the seq_len dim
        #  shape: (bs, seq_len, num_r, hidden)
        r_other_hidden_state = r_other_hidden_state.expand(-1, r_hidden_state.size(1), -1, -1)

        # cat both hidden states together
        #  shape: (bs, seq_len, num_r, 2 * hidden)
        rr_hidden_state = self.sum_rr(r_other_hidden_state, r_hidden_state)

        # calculate rr_score
        #  shape: (bs, seq_len, num_r, num_links)
        rr_score = self.rr_score(rr_hidden_state)

        # calculate a mask for rr_score
        #  shape: (bs, seq_len, num_r)
        rr_mask = r_indices_mask.unsqueeze(1)
        rr_mask = rr_mask.expand(-1, r_hidden_state.size(1), -1)
        rr_mask = rr_mask & (~is_pad).unsqueeze(-1)
        return self.rr_loss(rr_score[rr_mask], rr_pair_flag[rr_mask].float(), reduction="none")

    def train_forward_rr_lse_loss(
            self,
            hidden_state: Tensor,
            actions: Tensor,
            rr_pair_flag: Tensor,
    ) -> Tensor:
        bs, seq_len, dim = hidden_state.shape
        # (bs, seq_len)
        is_r = torch.ne(actions & (1 << 1), 0)
        is_pad = torch.eq(actions, 4)

        r_indices = batched_index_gen(is_r)
        num_r = r_indices.size(1)
        r_indices_mask = torch.ge(r_indices, 0)

        assert r_indices.size(-1) == is_r.sum(dim=-1).max()
        if num_r == 0:
            return torch.zeros(1, device=hidden_state.device, dtype=hidden_state.dtype)
        assert not self.is_feature_nest_depth_gt_1
        assert self.max_nest_depth == 1
        # we are in the easy case!

        r_indices.clamp_min_(0)
        r_hidden_state = torch.gather(
            hidden_state,
            dim=1,
            index=r_indices.unsqueeze(-1).expand(-1, -1, dim)
        ) if not self.is_feature_negative_lse_examples else hidden_state

        post_r_hidden_state = torch.gather(
            hidden_state,
            dim=1,
            index=r_indices.unsqueeze(-1).expand(-1, -1, dim)
        ) if self.is_feature_negative_lse_examples else None

        # shape: (bs, num_r or seq_len, num_r, hidden)
        r_rr_pair_flag = rr_pair_flag.gather(
            dim=1, index=r_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_r, self.num_links)
        ) if not self.is_feature_negative_lse_examples else rr_pair_flag

        post_r_hidden_state = r_hidden_state if not self.is_feature_negative_lse_examples else post_r_hidden_state
        post_r_hidden_state = post_r_hidden_state.unsqueeze(1).expand(
            (-1, num_r, -1, -1) if not self.is_feature_negative_lse_examples else (-1, seq_len, -1, -1))
        pre_r_hidden_state = r_hidden_state.unsqueeze(2).expand(-1, -1, num_r, -1)

        # cat both hidden states together
        #  shape: (bs, seq_len, num_r, 2 * hidden)
        rr_hidden_state = torch.cat((post_r_hidden_state, pre_r_hidden_state), dim=-1)

        # calculate rr_score
        #  shape: (bs, seq_len, num_r, num_links)
        rr_score = self.rr_score(rr_hidden_state)

        rr_denom = torch.logsumexp(rr_score, dim=-1)
        rr_numer = (
                rr_score + (~r_rr_pair_flag).mul(NEG_INF)
        ).logsumexp(dim=-1)

        # calculate a mask for rr_score
        #  shape: (bs, num_r, num_r)
        rr_mask = (r_indices_mask.unsqueeze(-1) * r_indices_mask.unsqueeze(1)) if not self.is_feature_negative_lse_examples else (r_indices_mask.unsqueeze(1) * ~is_pad.unsqueeze(-1))

        return (rr_denom - rr_numer) * rr_mask

    def forward_base_model(
            self,
            model: Module | None,
            input_ids: Tensor,
            attention_mask: Tensor,
            output_hidden_states: bool = False,
            output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if self.is_feature_use_t5_decoder:
            assert isinstance(self.model, AT5Model)
            output = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=attention_mask,
                return_dict=True,
            )
            dec_hidden = output.last_hidden_state
            enc_hidden = output.encoder_last_hidden_state
            hidden = enc_hidden + dec_hidden
            return hidden, hidden
        use_kv_model = model is not None
        model = model or self.model
        outputs: BaseModelOutput = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or self.is_feature_offset_ner_hidden_state,
            output_attentions=output_attentions,
            return_dict=True
        )
        output_hidden_state = outputs.last_hidden_state
        hidden_state = outputs.last_hidden_state
        if hidden_state.dim() == 4:  # BiT5
            hidden_state = hidden_state[0]
        _, seq_len, hidden = hidden_state.shape
        if use_kv_model and self.feature_attend_kv_project is not None:
            hidden_state = self.feature_attend_kv_project(hidden_state)

        ner_hidden_state = hidden_state
        ere_hidden_state = hidden_state
        if output_hidden_state.dim() == 4:
            ere_hidden_state = output_hidden_state[0]
        if self.is_feature_offset_ner_hidden_state:
            num_hidden_states = len(outputs.hidden_states)
            ner_hidden_state = outputs.hidden_states[-(num_hidden_states // 2)]

        return ner_hidden_state, ere_hidden_state

    def generate(
            self,
            inputs: Optional[Tensor] = None,
            **kwargs,
    ) -> tuple[Tensor, Tensor, list, list]:
        assert inputs is not None

        attention_mask = kwargs.get("attention_mask")
        entity_types = kwargs.get("entity_types")
        link_types = kwargs.get("link_types")

        hidden_states = self.forward_base_model(
            model=None,
            input_ids=inputs,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        )

        inference_forward_fn = self.inference_forward
        if self.is_feature_nest_depth_gt_1 and self.is_feature_extra_lr_class:
            inference_forward_fn = self.nested_inference_forward

        actions, pairings, links = inference_forward_fn(
            attention_mask,
            *hidden_states,
        )

        decoded_pairings, decoded_links = self.decode_actions_and_pairings(
            input_ids=inputs,
            actions=actions,
            pairings=pairings,
            links=links,
            entity_types=entity_types,
            link_types=link_types,
        )

        return actions, pairings, None, decoded_pairings, decoded_links

    def inference_forward(
            self,
            attention_mask: Tensor,
            hidden_state: Tensor,
            ere_hidden_state: Tensor,
    ):
        batch_size, seq_len, hidden = hidden_state.size()
        logits = self.is_l(hidden_state).squeeze(-1)  # (bs, seq_len)
        is_l = torch.gt(logits, 0).logical_and(attention_mask)  # (bs, seq_len)
        is_l_pos = batched_index_gen(is_l, min_size=1)
        is_l_indices = is_l.cumsum(dim=-1) - 1
        is_l_mask = torch.ge(is_l_indices, 0) & attention_mask.bool()
        is_l_indices.masked_fill_(~is_l_mask, 0)
        l_positions = torch.gather(
            is_l_pos,
            dim=1, index=is_l_indices
        )

        # take care when is_l_pos is negative
        l_positions.masked_fill_(~is_l_mask, 0)
        is_l_indices_hidden = l_positions.unsqueeze(-1).expand(-1, -1, hidden)  # (bs, seq_len, hidden)
        is_l_hidden = torch.gather(
            hidden_state,
            dim=1, index=is_l_indices_hidden
        )  # (bs, seq_len, hidden)
        pass
        lr_hidden_state = self.sum(is_l_hidden, hidden_state)
        lr_logits = self.lr_score(lr_hidden_state)

        if self.is_feature_extra_lr_class:
            lr_logits = lr_logits[..., :-1]
        lr_logits: Tensor
        lr_logits.masked_fill_(~is_l_mask.unsqueeze(-1), float("-inf"))
        lr_logits.masked_fill_(~attention_mask.bool().unsqueeze(-1), float("-inf"))
        denominator = lr_logits.logsumexp(dim=-1, keepdim=False)

        is_r = torch.gt(denominator, 0)

        actions = (
                          is_l * (1 << 0) + is_r * (1 << 1)
                  ) * attention_mask

        num_l = is_l_pos.size(-1)
        lr_pair_flag = hidden_state.new_zeros((batch_size, seq_len, num_l, self.num_types), dtype=torch.bool)
        scatter_indices = is_l_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.num_types)
        scatter_src = F.one_hot(lr_logits.argmax(dim=-1, keepdim=True), num_classes=self.num_types).to(torch.bool)
        scatter_src = scatter_src * lr_logits.logsumexp(dim=-1, keepdim=True).unsqueeze(-1).gt(0)
        lr_pair_flag = torch.scatter(lr_pair_flag, dim=2, index=scatter_indices, src=scatter_src)

        if self.is_feature_ner_only:
            return actions, lr_pair_flag, None

        is_r_pos = batched_index_gen(is_r)  # (bs, num_r)
        is_r_pos_mask = torch.ge(is_r_pos, 0)
        is_r_pos.masked_fill_(~is_r_pos_mask, 0)
        num_r = is_r_pos.size(-1)
        links = torch.zeros((actions.size(0), actions.size(1), num_r, self.num_links), dtype=torch.long)

        # (bs, num_r) => (bs, num_r, hidden)
        is_r_pos_hidden = is_r_pos.unsqueeze(-1).expand(-1, -1, hidden)
        is_r_pos_hidden = torch.gather(
            ere_hidden_state,
            dim=1, index=is_r_pos_hidden,
        )

        # a little more overhead but batched computation => profit at the cost of memory requirement
        is_rr_pos_hidden = is_r_pos_hidden.unsqueeze(1)  # (bs, 1, num_r, hidden)
        is_rr_pos_hidden = is_rr_pos_hidden.expand(-1, num_r, -1, -1)  # (bs, num_r, num_r, hidden)
        is_rr_pos_mask = is_r_pos_mask.unsqueeze(1)
        is_rr_pos_mask = is_rr_pos_mask.expand(-1, num_r, -1)

        is_rr_global_hidden = is_r_pos_hidden.unsqueeze(2)  # (bs, num_r, 1, hidden)
        is_rr_global_hidden = is_rr_global_hidden.expand(-1, -1, num_r, -1)  # (bs, num_r, num_r, hidden)
        is_rr_global_mask = is_r_pos_mask.unsqueeze(2)
        is_rr_global_mask = is_rr_global_mask.expand(-1, -1, num_r)

        is_rr_hidden = self.sum_rr(is_rr_pos_hidden, is_rr_global_hidden)
        is_rr_mask = is_rr_pos_mask & is_rr_global_mask
        is_rr_logits = self.rr_score(is_rr_hidden)
        is_rr_logits.masked_fill_(~is_rr_mask.unsqueeze(-1), float("-inf"))
        if self.is_feature_use_lse_rr_loss or self.is_feature_ce_loss:
            is_rr_score = is_rr_logits.argmax(dim=-1)
            is_rr = F.one_hot(is_rr_score, num_classes=self.num_links)
            is_rr = is_rr[..., :-1]
        else:
            if self.is_feature_extra_rr_class:
                is_rr_logits = is_rr_logits[..., :-1]
            is_rr_score = is_rr_logits.sigmoid()
            is_rr = is_rr_score > self.threshold

        for (batch, head, tail, tp) in is_rr.nonzero():
            if tp >= self.num_links:
                tail_pos = is_r_pos[batch, tail]
                links[batch, tail_pos, head, tp % self.num_links] = 1
                continue
            head_pos = is_r_pos[batch, head]
            links[batch, head_pos, tail, tp] = 1
        return actions, lr_pair_flag, links

    def decode_actions_and_pairings(
            self,
            input_ids: Tensor,
            actions: Tensor,
            pairings: Tensor,
            links: Optional[Tensor],
            entity_types: list[str],
            link_types: list[str],
    ):
        if self.is_feature_nest_depth_gt_1 and self.is_feature_extra_lr_class:
            return self.nested_decode_actions_and_pairings(
                input_ids,
                actions,
                pairings,
                links,
                entity_types,
                link_types,
            )

        links_per_element: list[list[tuple]] = [list() for _ in range(actions.size(0))]
        pairings_per_element, entities_by_batch_left_right = self._decode_pairings_to_entities(
            input_ids,
            actions,
            pairings,
            entity_types,
            use_left=False,
        )
        # keys for entities_by_batch_left_right are
        #  batch_idx, left idx, right idx

        if self.is_feature_ner_only or links is None:
            return pairings_per_element, links_per_element

        r_indices = batched_index_gen(actions.bitwise_and(0b10).ne(0))
        for (element, index, other_br, link_type) in links.nonzero():
            head_span = entities_by_batch_left_right.get((element.item(), index.item()), None)
            if head_span is None:
                continue

            if other_br >= r_indices.size(1):
                warnings.warn("Encountered invalid index for other_br, verify inputs/outputs")
                continue
            tail_span = entities_by_batch_left_right.get((element.item(), r_indices[element, other_br].item()), None)
            if tail_span is None:
                continue

            link_type = link_type.item()
            if self.is_feature_behave_like_plmarker:
                raise
                if link_type in {}:
                    raise

            links_per_element[element.item()].append((
                head_span,
                link_type,
                tail_span
            ))
        return pairings_per_element, links_per_element

    def _decode_pairings_to_entities(
            self,
            input_ids,
            actions: Tensor,
            pairings: Tensor,
            names,
            *,
            use_left=True
    ):
        pairings_per_element: list[list[tuple]] = [list() for _ in range(pairings.size(0))]
        if self.is_feature_extra_lr_class:
            pairings = pairings[..., :-1]

        def decode_tokens(tokens: Tensor):
            if not self.is_feature_perf_optimized:
                return self.tokenizer.decode(tokens)
            return tuple(tokens.tolist())

        l_indices = batched_index_gen(actions.bitwise_and(0b01).ne(0))
        entities_by_batch_left_right = {}
        for (batch_idx, seq_idx, l_idx, pairing_type) in pairings.nonzero():
            pairing_l_index = l_indices[batch_idx, l_idx]
            pairing_input_ids = input_ids[batch_idx, pairing_l_index:seq_idx + 1]

            pairings_per_element[batch_idx.item()].append(span := (
                tuple(pairing_input_ids.tolist()),
                pairing_type.item(),
                decode_tokens(pairing_input_ids),
                names[pairing_type.item()]
            ))
            # fast-path access to the spans for rr_pair_flag
            entity = (batch_idx.item(), pairing_l_index.item(), seq_idx.item()) if use_left else \
                (batch_idx.item(), seq_idx.item())
            entities_by_batch_left_right[entity] = span
        return pairings_per_element, entities_by_batch_left_right

    @staticmethod
    def nested_train_forward_denom_numer(
            lr_score: Tensor,
            kept_lr_pair_flag: Tensor,
            is_after_or_at_l: Tensor
    ) -> tuple[Tensor, Tensor]:
        mask = kept_lr_pair_flag.any(dim=-1)
        denom = torch.logsumexp(
            lr_score, dim=3, keepdim=False
        ) * is_after_or_at_l.any(dim=2, keepdim=True) * mask

        f_min = NEG_INF
        numer = torch.logsumexp(
            lr_score + (~kept_lr_pair_flag * f_min),
            dim=3, keepdim=False
        ) * is_after_or_at_l.any(dim=2, keepdim=True) * mask
        return denom, numer

    def nested_train_forward_rr_loss(
            self,
            actions: Tensor,
            hidden_state: Tensor,
            rr_pair_flag: Tensor,
    ) -> Tensor:
        if self.is_feature_ner_only:
            return hidden_state.new_zeros((1,))
        if rr_pair_flag.size(3) == 0:
            return hidden_state.new_zeros((1,))
        assert 0 not in rr_pair_flag.shape, rr_pair_flag.shape
        bs, seq_len, dim = hidden_state.shape
        device = hidden_state.device
        # (bs, seq_len)
        is_pad = torch.eq(actions, 4)
        is_l = torch.ne(actions & (1 << 0), 0)
        is_r = torch.ne(actions & (1 << 1), 0)

        # assert is_r.sum() > 0
        l_indices = batched_index_gen(is_l, min_size=1)
        l_indices_mask = torch.ge(l_indices, 0)
        # (bs, num_r = is_r.sum(dim=-1).max())
        num_r = is_r.sum(dim=-1).max()
        num_l = is_l.sum(dim=-1).max()
        r_indices = batched_index_gen(is_r, min_size=1)
        r_indices_mask = torch.ge(r_indices, 0)

        assert self.is_feature_nest_depth_gt_1

        # find all right hidden states:
        #  shape: (bs, num_r)
        r_other_indices = torch.where(
            torch.ge(r_indices, 0),
            r_indices,
            0
        ).unsqueeze(-1).expand(-1, -1, dim)

        l_before_r_indices = r_indices.unsqueeze(-1) - l_indices.unsqueeze(1)
        l_before_r_indices.masked_fill_((l_indices == -1).unsqueeze(1), -1)
        l_before_r_mask = l_before_r_indices >= 0
        l_before_r_indices = l_before_r_indices + (~l_before_r_mask * 10000)

        top_k = min(self.max_nest_depth, l_before_r_indices.size(-1))
        top_k_l_before_r_indices, top_k_indices = torch.topk(l_before_r_indices, k=top_k, largest=False)
        top_k_l_before_r_mask = torch.gather(l_before_r_mask, dim=2, index=top_k_indices)
        top_k_l_before_r_indices.masked_fill_(~top_k_l_before_r_mask, -1)

        # (bs, num_r, nest_depth)
        top_k_l_before_r_indices = torch.where(
            top_k_l_before_r_mask,
            -(top_k_l_before_r_indices - r_indices.unsqueeze(-1)),
            0
        )
        # prepare top_k_l_before_r_indices for hidden_state gather
        # (bs, num_r, nest_depth, hidden)
        top_k_l_before_r_indices = top_k_l_before_r_indices.unsqueeze_(-1).expand(-1, -1, -1, dim)
        # prepare hidden_state for gather with indices from top_k_l_before_r
        top_k_hidden_state = hidden_state.unsqueeze(2).expand(-1, -1, top_k, -1)
        # gather the 'nest_depth' closest hidden states for each right bracket in the input from 'hidden_states'
        # (bs, num_r, nest_depth, hidden)
        top_k_hidden_state = torch.gather(top_k_hidden_state, dim=1, index=top_k_l_before_r_indices)
        top_k_hidden_state.masked_fill_(~top_k_l_before_r_mask.unsqueeze(-1), 0)

        # prepare the mask for dim = 4 (left brackets for right brackets)
        top_k_l_before_r_mask = top_k_l_before_r_mask.unsqueeze_(1).unsqueeze_(1).expand(-1, seq_len, top_k, -1, -1)
        # prepare the top_k hidden states for the final combined hidden states
        top_k_hidden_state = top_k_hidden_state.unsqueeze_(1).unsqueeze_(1).expand(-1, seq_len, top_k, -1, -1, -1)

        # find their corresponding hidden states:
        #  shape will be (bs, num_r, hidden)
        r_other_gather_inp = hidden_state

        r_other_hidden_state = torch.gather(
            r_other_gather_inp,
            dim=1,
            index=r_other_indices)
        r_other_hidden_state = r_other_hidden_state * r_indices_mask.unsqueeze(-1)

        # prepare the right hidden states for the final combined hidden states
        r_other_hidden_state.unsqueeze_(1).unsqueeze_(1).unsqueeze_(-2)
        # shape: (bs, seq_len, nest_depth, num_r, nest_depth, hidden)
        r_other_hidden_state = r_other_hidden_state.expand(-1, seq_len, top_k, -1, top_k, -1)

        # (bs, seq_len, nest_depth, num_r, nest_depth, hidden)
        r_other_hidden_state = torch.add(r_other_hidden_state, top_k_hidden_state)

        # now: find the 'nest_depth' closest left brackets for each position in the sequence
        #  shape: (bs, seq_len, num_l)
        l_before_pos_indices = (actions >= 0).cumsum(dim=-1) - 1
        l_before_pos_indices = l_before_pos_indices.unsqueeze_(-1) - l_indices.unsqueeze(1)
        l_before_pos_mask = l_before_pos_indices >= 0
        l_before_pos_mask = l_before_pos_mask & (l_indices >= 0).unsqueeze(1) & (~is_pad).unsqueeze(-1)

        #  shape: (bs, seq_len, num_l => nest_depth)
        l_before_pos_indices = l_before_pos_indices + (~l_before_pos_mask * 10000)
        l_before_pos_top_k, l_before_pos_top_k_indices = torch.topk(l_before_pos_indices, k=top_k, largest=False)
        l_before_pos_top_k = -(l_before_pos_top_k - ((~is_pad).cumsum(dim=-1) - 1).unsqueeze_(-1))
        l_before_pos_top_k_mask = torch.gather(l_before_pos_mask, dim=2, index=l_before_pos_top_k_indices)
        l_before_pos_top_k_indices.masked_fill_(~l_before_pos_top_k_mask, 0).unsqueeze_(-1)
        #                                                              bs, seq_len, nest_depth, num_r, nest_depth, hidden
        # l_before_pos_top_k_indices = l_before_pos_top_k_indices.expand(-1, -1, -1, )

        # update rr_pair_flag, because we have selected the top_k indices in dim = 2
        orig_rr_pair_flag = rr_pair_flag  # keep a copy of the original one
        # (bs, seq_len, nest_depth, 1, 1, num_links)
        l_before_pos_top_k_indices = l_before_pos_top_k_indices.unsqueeze_(-1).unsqueeze_(-1)
        l_before_pos_top_k_indices = l_before_pos_top_k_indices.expand(-1, -1, -1, num_r, num_l, rr_pair_flag.size(-1))
        rr_pair_flag = torch.gather(rr_pair_flag, 2, l_before_pos_top_k_indices)
        if rr_pair_flag.size(3) == 0:
            return hidden_state.new_zeros((1,))

        # prepare the hidden states for dim = 2
        # l_before_pos_top_k.masked_fill_(~l_before_pos_top_k_mask, 0).unsqueeze_(-1)  # pytorch PR? doesnt work with compile
        l_before_pos_top_k = l_before_pos_top_k.masked_fill(~l_before_pos_top_k_mask, 0).unsqueeze(-1)
        l_before_pos_top_k = l_before_pos_top_k.expand(-1, -1, -1, dim)
        l_before_pos_hidden_state = hidden_state.unsqueeze(2).expand(-1, -1, top_k, -1)
        l_before_pos_hidden_state = torch.gather(l_before_pos_hidden_state, 1, l_before_pos_top_k)
        # (bs, seq_len, nest_depth, hidden) -> (bs, seq_len, nest_depth, 1, 1, hidden)
        l_before_pos_hidden_state = l_before_pos_hidden_state.unsqueeze(-2).unsqueeze(-2)
        # prepare the mask for dim = 2 (left brackets for each position in dim 1 (seq_len))
        l_before_pos_top_k_mask.unsqueeze_(-1).unsqueeze_(-1)
        l_before_pos_top_k_mask = l_before_pos_top_k_mask.expand(-1, -1, -1, num_r, top_k)

        # combine l_before_pos_hidden_state with hidden_state
        # shape: (bs, seq_len, num_r, hidden)
        r_hidden_state = hidden_state.unsqueeze(2).unsqueeze(-2).unsqueeze(2)  # .expand(-1, -1, num_r, -1)
        r_hidden_state = r_hidden_state + l_before_pos_hidden_state
        # we can save an unsqueeze + expand if we add instead of concatenating and make use of shape broadcasting
        r_hidden_state = r_hidden_state.expand(-1, -1, -1, num_r, top_k, -1)

        # update rr_pair_flag because we have selected the top k in dim = 4
        top_k_indices = top_k_indices.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        top_k_indices = top_k_indices.expand(-1, seq_len, top_k, -1, -1, rr_pair_flag.size(-1))
        rr_pair_flag = torch.gather(rr_pair_flag, 4, top_k_indices)

        # cat both hidden states together
        #  shape: (bs, seq_len, num_r, 2 * hidden)
        rr_hidden_state = torch.cat((r_other_hidden_state, r_hidden_state), dim=-1)

        # calculate rr_score
        #  shape: (bs, seq_len, num_r, num_links)
        rr_score = self.rr_score(rr_hidden_state)

        rr_mask_inp = ~is_pad
        if self.is_feature_in_development:
            rr_mask_inp = is_r
        # calculate a mask for rr_score
        #  shape: (bs, seq_len, num_r)
        rr_mask = r_indices_mask.unsqueeze(1)
        rr_mask = rr_mask.expand(-1, seq_len, -1)
        rr_mask = rr_mask & rr_mask_inp.unsqueeze(-1)
        rr_mask = rr_mask.unsqueeze_(2).unsqueeze_(-1) & top_k_l_before_r_mask & l_before_pos_top_k_mask

        return self.rr_loss(rr_score[rr_mask], rr_pair_flag[rr_mask].float(), reduction="none")

    def nested_inference_forward(
            self,
            attention_mask: Tensor,
            hidden_state: Tensor,
            ere_hidden_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # assert not self.is_feature_frozen_encoder
        assert self.is_feature_extra_lr_class
        assert self.is_feature_nest_depth_gt_1

        batch_size, seq_len, hidden = hidden_state.size()
        logits = self.is_l(hidden_state).squeeze(-1)  # (bs, seq_len)
        indices = torch.arange(seq_len, device=hidden_state.device).unsqueeze(0).expand(batch_size, -1)
        nest_depth_hidden_state = hidden_state.unsqueeze(2).expand(-1, -1, self.max_nest_depth, -1)

        is_l = torch.gt(logits, 0)  # (bs, seq_len)
        is_l_pos = batched_index_gen(is_l, min_size=self.max_nest_depth)
        is_l_mask = is_l_pos >= 0

        l_at_pos = indices.unsqueeze(-1) - is_l_pos.unsqueeze(1)
        l_at_pos.masked_fill_(~is_l_mask.unsqueeze(1), -1)
        l_at_pos_mask = l_at_pos >= 0
        l_at_pos_mask = l_at_pos_mask & attention_mask.unsqueeze(-1).bool()
        l_at_pos.masked_fill_(~l_at_pos_mask, 10000)

        # (bs, seq_len, nest_depth)
        top_k_l_at_pos_indices: Tensor
        top_k_l_at_pos_positions, top_k_l_at_pos_indices = torch.topk(l_at_pos, k=self.max_nest_depth, largest=False)
        top_k_l_at_pos_positions = torch.gather(
            is_l_pos.unsqueeze(1).expand(-1, seq_len, -1),
            dim=2, index=top_k_l_at_pos_indices
        )
        top_k_l_at_pos_mask = torch.gather(l_at_pos_mask, 2, top_k_l_at_pos_indices)
        top_k_l_at_pos_positions.masked_fill_(~top_k_l_at_pos_mask, 0)

        top_k_l_at_pos_hidden_positions = top_k_l_at_pos_positions.unsqueeze(-1).expand(-1, -1, -1, hidden)
        # convert 0..num_l to indices
        # top_k_l_at_pos_hidden =
        top_k_l_at_pos_hidden = torch.gather(
            nest_depth_hidden_state,
            dim=1, index=top_k_l_at_pos_hidden_positions
        )  # (bs, seq_len, nest_depth, hidden)
        lr_hidden_state = torch.cat((
            top_k_l_at_pos_hidden,
            nest_depth_hidden_state,
        ), dim=-1)  # (bs, seq_len, nest_depth, 2 * hidden)

        # (bs, seq_len, nest_depth, num_types)
        lr_logits = self.lr_score(lr_hidden_state)
        lr_logits.masked_fill_(~top_k_l_at_pos_mask.unsqueeze(-1), float("-inf"))
        lr_logits.masked_fill_(~attention_mask.unsqueeze(-1).unsqueeze(-1).bool(), float("-inf"))

        # (bs, seq_len, nest_depth)
        last_class_denominator = lr_logits[..., [-1]]
        denominator = lr_logits[..., :-1] > last_class_denominator
        denominator = denominator.any(dim=-1)

        # (bs, seq_len)
        is_r = denominator.any(dim=-1)
        actions = (is_l * (1 << 0) + is_r * (1 << 1)) * attention_mask

        num_l = is_l_pos.size(-1)
        lr_pair_flag = torch.zeros((batch_size, seq_len, num_l, self.num_types), device=hidden_state.device, dtype=torch.bool)
        lr_pair_flag[..., -1] = True
        lr_argmax = lr_logits.argmax(dim=-1)
        lr_argmax.masked_fill_(~top_k_l_at_pos_mask, self.num_types - 1)
        one_hot_type_choice = F.one_hot(lr_argmax).bool()
        scatter_indices = top_k_l_at_pos_indices.unsqueeze(-1).expand(-1, -1, -1, self.num_types)
        lr_pair_flag = torch.scatter(lr_pair_flag, dim=2, index=scatter_indices, src=one_hot_type_choice)

        is_r_pos = batched_index_gen(is_r, min_size=1)  # (bs, num_r)
        is_r_pos_mask = torch.ge(is_r_pos, 0)
        is_r_pos.masked_fill_(~is_r_pos_mask, 0)
        num_r = is_r_pos.size(-1)
        rr_pair_flag = torch.zeros((batch_size, seq_len, num_l, num_r, num_l, self.num_links), device=hidden_state.device, dtype=torch.bool)

        if self.is_feature_ner_only:
            return actions, lr_pair_flag, rr_pair_flag

        nest_depth_hidden_state = ere_hidden_state.unsqueeze(2).expand(-1, -1, self.max_nest_depth, -1)
        top_k_l_at_pos_hidden = torch.gather(
            nest_depth_hidden_state,
            dim=1, index=top_k_l_at_pos_hidden_positions
        )  # (bs, seq_len, nest_depth, hidden)

        # (bs, num_r) => (bs, num_r, nest_depth, hidden)
        is_r_pos_indices = is_r_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.max_nest_depth, hidden)
        is_r_pos_hidden = torch.gather(
            nest_depth_hidden_state,
            dim=1, index=is_r_pos_indices,
        )
        is_r_pos_l_hidden = torch.gather(
            top_k_l_at_pos_hidden,
            dim=1, index=is_r_pos_indices
        )
        # (bs, num_r, nest_depth, hidden)
        is_r_pos_hidden = is_r_pos_hidden + is_r_pos_l_hidden

        # a little more overhead but batched computation => profit at the cost of memory requirement
        is_rr_pos_hidden = is_r_pos_hidden.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, num_r, nest_depth, hidden)
        is_rr_pos_hidden = is_rr_pos_hidden.expand(-1, num_r, self.max_nest_depth, -1, -1, -1)
        is_rr_pos_mask = is_r_pos_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)  # (bs, 1, 1, num_r, 1)
        is_rr_pos_mask = is_rr_pos_mask.expand(-1, num_r, self.max_nest_depth, -1, self.max_nest_depth)

        is_rr_global_hidden = is_r_pos_hidden.unsqueeze(3).unsqueeze(3)  # (bs, num_r, nest_depth, 1, 1, hidden)
        is_rr_global_hidden = is_rr_global_hidden.expand(-1, -1, -1, num_r, self.max_nest_depth, -1)  # (bs, num_r, nest_depth, num_r, nest_depth, hidden)
        is_rr_global_mask = is_r_pos_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (bs, num_r, 1, 1, 1)
        is_rr_global_mask = is_rr_global_mask.expand(-1, -1, self.max_nest_depth, num_r, self.max_nest_depth)  # (bs, num_r, num_r, nest_depth)

        is_rr_hidden = torch.cat((is_rr_pos_hidden, is_rr_global_hidden), dim=-1)
        is_rr_mask = is_rr_pos_mask & is_rr_global_mask
        is_rr_logits = self.rr_score(is_rr_hidden)
        is_rr_logits.masked_fill_(~is_rr_mask.unsqueeze(-1), float("-inf"))
        is_rr_score = is_rr_logits.sigmoid()
        is_rr = is_rr_score > self.threshold
        if self.is_feature_extra_rr_class:
            is_rr = is_rr_logits >= is_rr_logits.amax(dim=-1, keepdim=True)
            is_rr = is_rr & (is_rr_logits.argmax(dim=-1, keepdim=True) != is_rr_logits.size(-1))
            is_rr = is_rr[..., :-1]
            assert False

        for (batch, head, head_nest, tail, tail_nest, tp) in is_rr.nonzero():
            head_pos = is_r_pos[batch, head]
            tail_pos = is_r_pos[batch, tail]

            head_l_pos = top_k_l_at_pos_positions[batch, head_pos, head_nest]
            assert head_l_pos <= head_pos
            head_l_index = top_k_l_at_pos_indices[batch, head_pos, head_nest]
            tail_l_pos = top_k_l_at_pos_positions[batch, tail_pos, tail_nest]
            assert tail_l_pos <= tail_pos
            tail_l_index = top_k_l_at_pos_indices[batch, tail_pos, tail_nest]

            if tp >= self.num_links:
                rr_pair_flag[batch, tail_pos, tail_l_index, head, head_l_index, tp % self.num_links] = 1
                continue
            rr_pair_flag[batch, head_pos, head_l_index, tail, tail_l_index, tp] = True
            pass
        return actions, lr_pair_flag, rr_pair_flag

    def nested_decode_actions_and_pairings(
            self,
            input_ids: Tensor,
            actions: Tensor,
            pairings: Tensor,
            links: Tensor,
            lr_typing_names: list[str],
            rr_typing_names: list[str],
    ):
        links_per_element: list[list[tuple]] = [list() for _ in range(actions.size(0))]
        pairings_per_element, entities_by_batch_left_right = self._decode_pairings_to_entities(
            input_ids,
            actions,
            pairings,
            lr_typing_names,
            use_left=True
        )

        if self.is_feature_ner_only:
            return pairings_per_element, links_per_element

        l_indices = batched_index_gen(actions.bitwise_and(0b01).ne(0))
        r_indices = batched_index_gen(actions.bitwise_and(0b10).ne(0))
        for (batch_idx, seq_idx, seq_l_idx, r_idx, r_l_idx, link_type) in links.nonzero():
            if seq_l_idx >= l_indices.size(1):
                entities_with_batch_id = {v for (b, _, _), v in entities_by_batch_left_right.items() if b == batch_idx}
                warnings.warn(f"Encountered invalid index for other left bracket, verify inputs/outputs: "
                              f"{entities_with_batch_id}, {seq_idx, seq_l_idx, r_idx, r_l_idx}")
                continue
            head_l_index = l_indices[batch_idx, seq_l_idx]
            head_key = (batch_idx.item(), head_l_index.item(), seq_idx.item())
            head_entity = entities_by_batch_left_right.get(head_key, None)
            if head_entity is None:
                continue
            if r_l_idx >= l_indices.size(1):
                entities_with_batch_id = {v for (b, _, _), v in entities_by_batch_left_right.items() if b == batch_idx}
                warnings.warn(f"Encountered invalid index for other left bracket, verify inputs/outputs: "
                              f"{entities_with_batch_id}, {seq_idx, seq_l_idx, r_idx, r_l_idx}")
                continue
            if r_idx >= r_indices.size(1):
                entities_with_batch_id = {v for (b, _, _), v in entities_by_batch_left_right.items() if b == batch_idx}
                warnings.warn(f"Encountered invalid index for other right bracket, verify inputs/outputs: "
                              f"{entities_with_batch_id}, {seq_idx, seq_l_idx, r_idx, r_l_idx}")
                continue
            tail_l_index = l_indices[batch_idx, r_l_idx]
            tail_r_index = r_indices[batch_idx, r_idx]
            tail_key = (batch_idx.item(), tail_l_index.item(), tail_r_index.item())
            tail_entity = entities_by_batch_left_right.get(tail_key, None)
            if tail_entity is None:
                continue
            link_type = link_type.item()

            links_per_element[batch_idx.item()].append((
                head_entity,
                link_type,
                tail_entity
            ))

        return pairings_per_element, links_per_element
