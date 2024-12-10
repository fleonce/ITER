import dataclasses
import os
import warnings
from collections import OrderedDict, defaultdict
from json import load as load_json
from logging import Logger
from pathlib import Path

import torch
from safetensors_dataset import SafetensorsDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, AutoConfig, PretrainedConfig

from iter.datasets.training import NameableConfig
from iter.modeling_features import FeaturesMixin


def convert_word_ids(word_ids: list[int | None]) -> torch.Tensor:
    word_ids = [w if w is not None else -1 for w in word_ids]
    if word_ids[0] < 0:
        word_ids[0] = -100
    word_ids = torch.tensor(word_ids).long()
    return torch.where(word_ids == -1, word_ids.max() + 1, word_ids).long()


class ITERDataset:
    def __getitem__(self, item: str) -> SafetensorsDataset: ...

    def setup_dataset(self): ...
    def setup_hpo(self): ...
    def list_splits(self, logger: Logger): ...

@dataclasses.dataclass
class CoNLL(FeaturesMixin, NameableConfig, ITERDataset):
    section_name = "dataset"
    with_filepath = True

    name: str
    features: int
    entity_nest_depth: int
    splits: dict[str, str]
    data_dir: str
    tokenizer: PreTrainedTokenizerBase = None
    tokenizer_config: PretrainedConfig = None
    reinitialize_dataset: bool = False
    memory_only: bool = False

    file_path: str = None
    max_length: int = 512
    dont_shorten_tokenizer_name: bool = False
    use_task_prefix: bool = False
    hpo: bool = False
    data: dict[str, SafetensorsDataset] = dataclasses.field(default_factory=dict)
    _link_types: list[str] = None
    _entity_types: list[str] = None

    def __post_init__(self):
        if isinstance(self.features, list):
            features = sum([2 ** f for f in self.features])
            object.__setattr__(self, "features", features)
        if self.tokenizer is not None:
            config = AutoConfig.from_pretrained(self.tokenizer.name_or_path)
            object.__setattr__(self, "tokenizer_config", config)

    def __getitem__(self, item: str) -> SafetensorsDataset:
        return self.data[item]

    @property
    def entity_types(self) -> list[str]:
        self.load_types()
        return self._entity_types

    @property
    def link_types(self) -> list[str]:
        self.load_types()
        return self._link_types

    @property
    def symmetric_link_types(self) -> set[int]:
        self.load_types()
        return self._symmetric_link_types

    def load_types(self):
        if self._link_types is not None:
            return
        with open(self.split_path(self.splits["types"])) as f:
            json_blob = load_json(f)
        self._entity_types = []
        self._entity_types.extend(json_blob["entities"].keys())
        self._link_types = []
        self._link_types.extend(json_blob["relations"].keys())
        self._symmetric_link_types = set()
        self._symmetric_link_types.update(k for k, v in json_blob["relations"].items() if v.get("symmetric", False))
        self._symmetric_link_types = {self._link_types.index(s) for s in self.symmetric_link_types}

    @property
    def num_types(self) -> int:
        return len(self.entity_types) + int(self.is_feature_extra_lr_class)

    @property
    def num_real_types(self) -> int:
        return len(self.entity_types)

    @property
    def num_links(self) -> int:
        return len(self.link_types) + int(self.is_feature_extra_rr_class)

    @property
    def num_real_links(self) -> int:
        return len(self.link_types)

    def setup_hpo(self):
        self.splits["test"] = self.splits["eval"]
        self.hpo = True

    def is_hpo_dataset(self):
        return self.hpo

    def setup_dataset(self):
        for split, path in self.splits.items():
            if split == "types":
                continue

            if (
                not os.path.exists(self.split_path(path, final=True))
                or self.reinitialize_dataset
            ):
                self.setup_split(split, path)
            self.load_split(split, path)

    def list_splits(self, logger):
        info = [f"Loaded dataset {self.name} with splits"]
        for split, path in self.splits.items():
            if split == "types":
                continue
            info.append(f"'{split}': {len(self.data[split])},")
        logger.info(" ".join(info))

    def split_path(self, path, final=False):
        filename = path
        if final:
            filename = Path(filename)
            filename = filename.name[:-len(filename.suffix)]
            if self.features != 0:
                filename = f"{filename}.feat{self.features:b}"
            if self.tokenizer_config is not None:
                if self.dont_shorten_tokenizer_name:
                    tokenizer_name = self.tokenizer.name_or_path.replace("/", "_")
                else:
                    if self.tokenizer.name_or_path.startswith("google/t5-v1"):
                        raise ValueError("Using T5 v1.1 with the default T5 tokenizer will not work")
                    tokenizer_name = self.tokenizer_config.model_type
                if tokenizer_name != "t5":
                    filename = f"{filename}.{tokenizer_name}"
            filename = f"{filename}.safetensors"
        return Path.cwd() / "datasets" / self.data_dir / filename

    def load_split(self, name, path):
        if self.memory_only:
            if name not in self.data:
                raise ValueError(f"Cannot load split {name} when {self.memory_only=}")
            return
        dataset = SafetensorsDataset.load_from_file(self.split_path(path, final=True))
        self.data[name] = dataset

    def open_split(self, _name, path):
        with open(self.split_path(path, final=False)) as f:
            return load_json(f)

    def setup_split(self, name, path):
        data = self.open_split(name, path)
        dataset = self.setup_examples(name, data, self.split_path(path))
        dataset = SafetensorsDataset(dataset)
        if not self.memory_only:
            dataset.save_to_file(self.split_path(path, final=True))
        self.data[name] = dataset

    def setup_examples(self, split, json, out_filename: Path):
        output = {
            'input_ids': [],
            'actions': [],
            'lr_pair_flag': [],
            'rr_pair_flag': [],
        }

        def append_to_map(m: dict):
            for k, v in m.items():
                output[k].append(v)

        infos = OrderedDict(skipped=0)
        for item in (tq := tqdm(json, desc=out_filename.name)):
            example = self.setup_example(split, item)
            if not example:
                infos["skipped"] += 1
                tq.set_postfix(infos)
                continue
            append_to_map(example)

        def map_without_empty(m: dict):
            return {k: v for k, v in m.items() if len(v) > 0}

        output = map_without_empty(output)
        return output

    def setup_example(self, split, item):
        # % if self._is_skip_example(filename, item):
        # %    return False
        tokens = item["tokens"]
        if self.is_feature_use_extended_context:
            warnings.warn(f"Dataset {self.name} has flag 'use_extended_context' enabled")
            tokens = item["extended"]
        encodings = self.tokenizer(
            tokens, is_split_into_words=True, return_tensors="pt", return_offsets_mapping=True)
        input_ids = encodings["input_ids"].squeeze(0)
        if input_ids.size(-1) > self.tokenizer.model_max_length:
            warnings.warn(f"Skipping examples because the tokenizer/model is only configured for "
                          f"{self.tokenizer.model_max_length} tokens, but we got {input_ids.size(-1)} tokens")
            return False  # skip examples longer than 512 tokens
        word_ids = convert_word_ids(encodings.word_ids())
        entities = item["entities"]
        links = item["relations"] if "relations" in item else []

        rights_at_positions: dict[int, set] = defaultdict(set)
        lefts_at_positions: dict[int, set] = defaultdict(set)
        actions = torch.full_like(input_ids, 0)
        seq_len = actions.size(-1)
        links_between_entities: dict[tuple, set] = defaultdict(set)

        all_rights = set()
        all_lefts = set()
        filtered_entities = []
        skip_example = False
        if len(entities) == 0:
            if not self.is_feature_empty_examples:
                return False
            lr_pair_flag = torch.zeros((seq_len, 1, self.num_types), dtype=torch.bool)
            if self.is_feature_extra_lr_class:
                lr_pair_flag[..., -1] = 1
            rr_pair_flag = torch.zeros((seq_len, 1, self.num_links), dtype=torch.bool)
            if self.is_feature_nest_depth_gt_1:
                rr_pair_flag = torch.zeros((seq_len, 1, 1, 1, self.num_links), dtype=torch.bool)
            if self.is_feature_extra_rr_class:
                rr_pair_flag[..., -1] = 1

            output = {
                'input_ids': input_ids,
                'actions': actions,
                'lr_pair_flag': lr_pair_flag.to_sparse(),
                'rr_pair_flag': rr_pair_flag.to_sparse()
            }

            if self.is_feature_ner_only:
                output.pop('rr_pair_flag')
            warnings.warn(f"Dataset {self.name} contains examples where there are no entities, be aware")
            return output

        if self.is_feature_use_extended_context:
            sent_start = tokens.index('<extra_id_22>')
            for entity in entities:
                entity['start'] += sent_start
                entity['end'] += sent_start
                pass

        checked_entities = dict()
        for entity in entities:
            start_word, end_word = entity["start"], entity["end"] - 1
            if (start_word, end_word) in checked_entities:
                warnings.warn("There exist training examples where the same entity occurs multiple times in the dataset")
                filtered_entities.append(checked_entities[(start_word, end_word)])
                continue
            if start_word in all_lefts and end_word in all_rights:
                warnings.warn("There exist training examples where the boundaries of an entity are already registered")

            checked_entities[(start_word, end_word)] = entity
            all_lefts.add(entity["start"])
            all_rights.add(entity["end"])
            filtered_entities.append(entity)
        all_rights = list(all_rights)
        all_rights.sort()
        all_lefts = list(all_lefts)
        all_lefts.sort()

        # starts
        position_entities = []
        skip_example = False
        for entity in filtered_entities:
            start_word = entity["start"]
            end_word = entity["end"] - 1
            start_pos = all_lefts.index(entity["start"])
            end_pos = all_rights.index(entity["end"])
            entity = (start_pos, entity["type"], end_pos)
            try:
                start_token_pos = torch.eq(word_ids, start_word).nonzero().min()
            except:
                print(word_ids.unique(), start_word, entity, tokens[start_word-1:start_word+2])
                return False
                raise
            end_token_pos = torch.eq(word_ids, end_word).nonzero().max()

            position_entities.append(entity)
            lefts_at_positions[start_pos].add(entity)
            actions[start_token_pos] |= 0b01
            actions[start_token_pos] &= 0b011

            rights_at_positions[end_pos].add(entity)
            actions[end_token_pos] |= 0b10
            actions[end_token_pos] &= 0b011
        if skip_example:
            return False
        if len(filtered_entities) == 0:
            assert False
        if len(filtered_entities) < len(entities):
            raise ValueError(f"Filtered entities")

        used_links = set()
        for link in links:
            link_type = self.link_types.index(link["type"])
            link_start = position_entities[link["head"]]
            link_end = position_entities[link["tail"]]
            if link_type in self.symmetric_link_types and self.is_feature_train_symrels_both_directions:
                used_links.add((link_end, link_start, link_type))
                links_between_entities[link_end + link_start].add(link_type)
            if link_type not in self.symmetric_link_types and self.is_feature_negsample_link_types:
                inv_link_type = link_type + len(self.link_types)
                used_links.add((link_end, link_start, inv_link_type))
                links_between_entities[link_end + link_start].add(inv_link_type)
            used_links.add((link_start, link_end, link_type))
            links_between_entities[link_start + link_end].add(link_type)

        if not self.is_feature_ner_only and len(links_between_entities) == 0:
            warnings.warn(f"Dataset {self.name} contains examples with no relations")
            if not self.is_feature_examples_without_links:
                warnings.warn(f"Dataset {self.name} is configured to SKIP examples without relations inside")
                return False

        a_is_l = torch.ne(torch.bitwise_and(actions, 0b01), 0)
        a_is_r = torch.ne(torch.bitwise_and(actions, 0b10), 0)
        if self.entity_nest_depth == 1:
            condition = a_is_l.sum() == a_is_r.sum()
            assert condition

        lr_pair_flag = torch.zeros((seq_len, max(1, a_is_l.sum().item()), self.num_types), dtype=torch.bool)
        if self.is_feature_extra_lr_class:
            lr_pair_flag[..., -1] = 1

        num_links = self.num_real_links + self.is_feature_extra_rr_class
        rr_pair_flag = torch.zeros((seq_len, max(1, a_is_r.sum().item()), num_links), dtype=torch.bool)
        if self.is_feature_nest_depth_gt_1:
            assert self.entity_nest_depth > 1
            num_l = a_is_l.sum().clamp_min(1)
            num_r = a_is_r.sum().clamp_min(1)
            rr_pair_flag = torch.zeros((seq_len, num_l, num_r, num_l, num_links), dtype=torch.bool)
        if self.is_feature_extra_rr_class:
            rr_pair_flag[..., -1] = 1
        if self.is_feature_negsample_link_types:
            repeat_dim = 2
            if self.is_feature_nest_depth_gt_1:
                repeat_dim = 4
            repeat_ints = torch.ones(rr_pair_flag.dim(), dtype=torch.int)
            repeat_ints[repeat_dim] = 2
            rr_pair_flag = rr_pair_flag.repeat(repeat_ints.tolist())

        indices = torch.arange(seq_len)
        is_l_indices = indices[a_is_l]
        is_r_indices = indices[a_is_r]
        for i, entities in lefts_at_positions.items():
            idx = is_l_indices[i]

            entities = [
                (
                    start_pos,
                    is_l_indices[start_pos],
                    entity_type,
                    end_pos,
                    is_r_indices[end_pos]
                )
                for (start_pos, entity_type, end_pos) in entities
            ]
            entities_with_lefts = list(entities)
            entities_with_lefts.sort(key=lambda x: x[4] - idx, reverse=False)  # sort by closest ] (right br)
            orig_entities_with_lefts = entities_with_lefts
            entities_with_lefts = entities_with_lefts[:self.entity_nest_depth]
            if len(entities_with_lefts) < len(orig_entities_with_lefts) and self.entity_nest_depth < len(
                    orig_entities_with_lefts):
                #raise ValueError(f"{self.entity_nest_depth=} but multiple entities with the same left bracket"
                #                 f"({len(orig_entities_with_lefts)})")
                pass
            assert len(entities_with_lefts) > 0, f"{entities_with_lefts=}"

            for (left_pos, _, entity_type, _, right_pos) in entities_with_lefts:
                type_ids = self.entity_types.index(entity_type)
                lr_pair_flag[right_pos, left_pos, type_ids] = 1
                if self.is_feature_extra_lr_class and lr_pair_flag[right_pos, left_pos, -1] == 1:
                    lr_pair_flag[right_pos, left_pos, -1] = 0
                # if self.ablation == 2 and lr_pair_flag[right_pos, left_pos, -1] == 1:
                #     lr_pair_flag[right_pos, left_pos, -1] = False

        assert (
                self.is_feature_extra_lr_class
                or self.is_feature_nest_depth_gt_1
                or (lr_pair_flag.sum() == is_r_indices.size(-1))
        ), f"{lr_pair_flag.sum()=} vs. {is_r_indices.size(-1)=}"

        for head_entity in position_entities:
            for tail_entity in position_entities:
                head_idx = is_r_indices[head_entity[2]]
                head_l_pos = head_entity[0]
                tail_pos = tail_entity[2]
                tail_l_pos = tail_entity[0]
                link_types = links_between_entities.get(head_entity + tail_entity, None)
                if link_types:
                    for link_type in link_types:
                        if self.is_feature_nest_depth_gt_1:
                            rr_pair_flag[head_idx, head_l_pos, tail_pos, tail_l_pos, link_type] = True
                            if self.is_feature_extra_rr_class:
                                rr_pair_flag[head_idx, head_l_pos, tail_pos, tail_l_pos, -1] = False
                            continue
                        rr_pair_flag[head_idx, tail_pos, link_type] = True
                        if self.is_feature_extra_rr_class:
                            rr_pair_flag[head_idx, tail_pos, -1] = False

        def cut_if_feature(feat_enabled: bool):
            def decorator(func):
                def inner(flag: torch.Tensor):
                    if feat_enabled:
                        flag = flag[..., :-1]
                    func(flag)

                return inner

            return decorator

        def run_if_feature(feat_enabled: bool):
            def decorator(func):
                def inner(flag: torch.Tensor):
                    if feat_enabled:
                        func(flag)

                return inner

            return decorator

        @cut_if_feature(self.is_feature_extra_lr_class)
        def check_lr_pair_flag(flag: torch.Tensor):
            if not self.is_feature_nest_depth_gt_1:
                # assert a_is_l.sum() == len(position_entities), position_entities
                # assert a_is_r.sum() == len(position_entities), (position_entities, item)
                assert a_is_l.sum() == a_is_r.sum()
                assert a_is_l.sum() == lr_pair_flag.amax(dim=-1).sum()
            else:
                assert torch.all(flag[actions == 2].amax(dim=(1, 2))), (
                position_entities, actions, flag[actions == 2].any(dim=-1))
                assert a_is_r.sum() == flag.amax(dim=(1, 2)).sum(), position_entities

        check_lr_pair_flag(lr_pair_flag)

        if self.use_task_prefix:
            num_l, num_r = lr_pair_flag.size(1), rr_pair_flag.size(1 if not self.is_feature_nest_depth_gt_1 else 2)
            task_prefix = self.tokenizer.encode("relation extraction: ", return_tensors="pt", add_special_tokens=True)
            task_prefix = task_prefix.squeeze()
            task_len = task_prefix.size(0)
            task_lr_pair_flag = torch.zeros((task_len, num_l, self.num_types), dtype=torch.bool)
            task_rr_pair_flag = torch.zeros((task_len, num_r, self.num_links), dtype=torch.bool)
            if self.is_feature_nest_depth_gt_1:
                task_rr_pair_flag = torch.zeros((task_len, num_l, num_r, num_l, self.num_links), dtype=torch.bool)

            input_ids = torch.cat((task_prefix, input_ids), dim=-1)
            actions = torch.cat((torch.zeros_like(task_prefix), actions), dim=-1)
            lr_pair_flag = torch.cat((task_lr_pair_flag, lr_pair_flag), dim=0)
            if self.is_feature_extra_lr_class:
                lr_pair_flag[..., -1] = 1
            rr_pair_flag = torch.cat((task_rr_pair_flag, rr_pair_flag), dim=0)
            if self.is_feature_extra_rr_class:
                rr_pair_flag[..., -1] = 1

        output = {
            'input_ids': input_ids,
            'actions': actions,
            'lr_pair_flag': lr_pair_flag.to_sparse(),
            'rr_pair_flag': rr_pair_flag.to_sparse()
        }

        if self.is_feature_ner_only:
            output.pop('rr_pair_flag')

        return output

