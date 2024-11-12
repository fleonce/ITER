import warnings
from typing import Optional

import torch
from torchcheck import batched_index_padded

from torch import Tensor


def _decode_pairings_to_entities(
    self,
    input_ids,
    actions: Tensor,
    pairings: Tensor,
    names,
    *,
    use_left=True
):
    if names is None:
        raise ValueError("names cannot be None")

    pairings_per_element: list[list[tuple]] = [list() for _ in range(pairings.size(0))]
    if self.is_feature_extra_lr_class:
        pairings = pairings[..., :-1]

    def decode_tokens(tokens: Tensor):
        if not self.is_feature_perf_optimized:
            return self.tokenizer.decode(tokens)
        return tuple(tokens.tolist())

    l_indices = batched_index_padded(actions.bitwise_and(0b01).ne(0))
    entities_by_batch_left_right = {}
    for (batch_idx, seq_idx, l_idx, pairing_type) in pairings.nonzero():
        pairing_l_index = l_indices[batch_idx, l_idx]
        pairing_input_ids = input_ids[batch_idx, pairing_l_index:seq_idx + 1]
        pairing_repr = decode_tokens(pairing_input_ids)

        pairings_per_element[batch_idx.item()].append(
            span := (
                tuple(pairing_input_ids.tolist()),
                pairing_type.item(),
                pairing_repr,
                names[pairing_type.item()]
            )
        )
        # fast-path access to the spans for rr_pair_flag
        entity = (batch_idx.item(), pairing_l_index.item(), seq_idx.item()) if use_left else \
            (batch_idx.item(), seq_idx.item())
        entities_by_batch_left_right[entity] = span
    return pairings_per_element, entities_by_batch_left_right


def decode_actions_and_pairings(
    self,
    input_ids: Tensor,
    actions: Tensor,
    pairings: Tensor,
    links: Optional[Tensor],
    link_probabilities: Optional[Tensor],
    entity_types: list[str],
    link_types: list[str],
):
    if entity_types is None:
        raise ValueError("entity_types cannot be None")

    if self.is_feature_nest_depth_gt_1 and self.is_feature_extra_lr_class:
        return nested_decode_actions_and_pairings(
            self,
            input_ids,
            actions,
            pairings,
            links,
            link_probabilities,
            entity_types,
            link_types,
        )

    links_per_element: list[list[tuple]] = [list() for _ in range(actions.size(0))]
    pairings_per_element, entities_by_batch_left_right = _decode_pairings_to_entities(
        self,
        input_ids,
        actions,
        pairings,
        entity_types,
        use_left=False,
    )

    if self.is_feature_ner_only or links is None:
        return pairings_per_element, links_per_element

    r_indices = batched_index_padded(actions.bitwise_and(0b10).ne(0))
    one = links.new_ones(tuple(), dtype=torch.float)
    for indices in links.nonzero():
        link_probability = one
        if link_probabilities is not None:
            link_probability = link_probabilities[indices.unbind()]
        element, index, other_br, link_type = indices.unbind()
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

        links_per_element[element.item()].append(
            (
                head_span,
                link_type,
                tail_span,
                link_probability.item(),
            )
        )
    return pairings_per_element, links_per_element


def nested_decode_actions_and_pairings(
    self,
    input_ids: Tensor,
    actions: Tensor,
    pairings: Tensor,
    links: Optional[Tensor],
    link_probabilities: Optional[Tensor],
    lr_typing_names: list[str],
    rr_typing_names: list[str],
):
    links_per_element: list[list[tuple]] = [list() for _ in range(actions.size(0))]
    pairings_per_element, entities_by_batch_left_right = _decode_pairings_to_entities(
        self,
        input_ids,
        actions,
        pairings,
        lr_typing_names,
        use_left=True
    )

    if self.is_feature_ner_only:
        return pairings_per_element, links_per_element

    if links is None:
        raise ValueError("links is None")

    l_indices = batched_index_padded(actions.bitwise_and(0b01).ne(0))
    r_indices = batched_index_padded(actions.bitwise_and(0b10).ne(0))
    one = links.new_ones(tuple(), dtype=torch.float)
    for indices in links.nonzero():
        link_probability = one
        if link_probabilities is not None:
            link_probability = link_probabilities[indices.unbind()]
        (batch_idx, seq_idx, seq_l_idx, r_idx, r_l_idx, link_type) = indices
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
            tail_entity,
            link_probability.item()
        ))

    return pairings_per_element, links_per_element
