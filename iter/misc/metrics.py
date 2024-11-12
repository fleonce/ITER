import dataclasses
import itertools
from dataclasses import dataclass

import typing_extensions
import warnings
from collections import defaultdict, OrderedDict
from typing import Mapping, Optional, Union, Literal, overload, Iterable, TypeVar, MutableMapping

import torch
from torch import Tensor


_entity_type = tuple[
    list,  # tokens
    int,  # type id
    str,  # text
    str,  # type str
]
_link_type = tuple[
    _entity_type,  # entity
    int,           # link id
    _entity_type,  # entity
]
_symmetric_link_type = frozenset[_entity_type | int]
_per_class_type = Optional[Union[Tensor, Mapping[int, tuple[float, float, float]]]]


def average_metrics(metrics: "list[Metrics]") -> Mapping[str, tuple[float, float]]:
    outcomes = defaultdict(list)
    for metric in metrics:
        for key, value in metric.to_dict().items():
            outcomes[key].append(value)
    tensor_outcomes = {k: torch.tensor(v) for k, v in outcomes.items()}
    std_mean_tensor_outcomes = {k: std_mean_or_zero(v) for k, v in tensor_outcomes.items()}
    std_mean_outcomes = {k: (mean.item(), std.item()) for k, (std, mean) in std_mean_tensor_outcomes.items()}
    return std_mean_outcomes


def average_into_metrics(metrics: "list[Metrics]") -> "Metrics":
    outcomes = average_metrics(metrics)
    outcome_means = {k: mean for k, (mean, std) in outcomes.items()}
    return Metrics.from_dict(outcome_means)



def std_mean_or_zero(t: torch.Tensor):
    if t.numel() <= 1:
        return torch.tensor(0.), t  # 0 std, mean = t
    if not t.is_floating_point():
        t = t.to(torch.float64)
    return torch.std_mean(t)


def format_average_metrics(best: int, metrics: "list[Metrics]", bound_metrics: "list[Metrics]"):
    outcomes = average_metrics(metrics)
    best_outcome = average_metrics([metrics[best],])
    bound_outcomes = average_metrics(bound_metrics)
    best_bound_outcome = average_metrics([bound_metrics[best],])

    all_outcomes = (outcomes, best_outcome, bound_outcomes, best_bound_outcome)
    outcomes_of_all = {k: tuple(o.get(k, (0.0, 0.0)) for o in all_outcomes) for k in outcomes.keys()}  # if all(k in o for o in all_outcomes)}
    tuple_outcomes = {k: tuple(itertools.chain.from_iterable(v)) for k, v in outcomes_of_all.items()}
    return OrderedDict({
        k: f"{mean:.6f} ± {std_dev:.4f} / "
           f"{bound_mean:.6f} ± {bound_std:.4f} "
           f"(best is {best:.6f} / {bound_best:.6f})"
        for k, (mean, std_dev, best, _, bound_mean, bound_std, bound_best, _) in tuple_outcomes.items()
    })


@dataclasses.dataclass
class Metrics:
    ner_pr: float
    ner_rec: float
    ner_f1: float
    ere_pr: float
    ere_rec: float
    ere_f1: float

    macro_ner_pr: float
    macro_ner_rec: float
    macro_ner_f1: float
    macro_ere_pr: float
    macro_ere_rec: float
    macro_ere_f1: float

    ner_per_class: _per_class_type = None
    ere_per_class: _per_class_type = None
    ner_per_class_names: Optional[list[str]] = None
    ere_per_class_names: Optional[list[str]] = None

    samples_per_second: float = 0.
    samples_total_num: int = 0
    samples_total_time: float = 0.

    loss: Optional[float] = None
    measurements: Optional[tuple[Tensor, ...]] = None


    def __repr__(self):
        string = (
            f"NER :: pr={self.ner_pr:.6f} rec={self.ner_rec:.6f} f1={self.ner_f1:.6f} "
            f"ERE :: pr={self.ere_pr:.6f} rec={self.ere_rec:.6f} f1={self.ere_f1:.6f} "
            f"PERF :: {self.samples_per_second}/s"
        )
        if self.loss is not None:
            string = string + f" LOSS :: {self.loss:.4f}"
        return string

    def clear_per_class(self):
        self.ere_per_class = self.ner_per_class = None

    @classmethod
    def from_dict(cls, inp: dict):
        return cls(
            ner_pr=inp.get("ner_pr", 0.),
            ner_rec=inp.get("ner_rec", 0.),
            ner_f1=inp.get("ner_f1", 0.),
            ere_pr=inp.get("ner_pr", 0.),
            ere_rec=inp.get("ner_rec", 0.),
            ere_f1=inp.get("ner_f1", 0.),
            macro_ner_pr=inp.get("macro_ner_pr", 0.),
            macro_ner_rec=inp.get("macro_ner_rec", 0.),
            macro_ner_f1=inp.get("macro_ner_f1", 0.),
            macro_ere_pr=inp.get("macro_ere_pr", 0.),
            macro_ere_rec=inp.get("macro_ere_rec", 0.),
            macro_ere_f1=inp.get("macro_ere_f1", 0.),
            samples_per_second=inp.get("samples_per_s", 0.),
            samples_total_num=inp.get("samples_total_num", 0),
            samples_total_time=inp.get("samples_total_s", 0.)
        )

    def to_dict(self):
        perf_metrics = {
            "samples_per_s": self.samples_per_second,
            "samples_total_num": self.samples_total_num,
            "samples_total_s": self.samples_total_time,
        }

        def join_dicts(b: dict, *dicts: dict):
            for d in dicts:
                b.update(d)
            return b

        per_class_metrics = dict()
        if (
            self.ner_per_class is not None
            and self.ner_per_class_names is not None
            and -1 not in self.ner_per_class
        ):
            per_class_metrics = join_dicts(dict(), *[
                {
                    k.lower() + "_pr": float(self.ner_per_class[i][0]),
                    k.lower() + "_rec": float(self.ner_per_class[i][1]),
                    k.lower() + "_f1": float(self.ner_per_class[i][2]),
                } for i, k in enumerate(self.ner_per_class_names)
            ])
        elif self.ner_per_class is not None and -1 not in self.ner_per_class:
            warnings.warn("Cannot save ner_per_class to dict as selfner_per_class_names is None")

        return {
            "f1": self.ere_f1,
            "pr": self.ere_pr,
            "rec": self.ere_rec,
            "ner_f1": self.ner_f1,
            "ner_pr": self.ner_pr,
            "ner_rec": self.ner_rec,
            "macro_f1": self.macro_ere_f1,
            "macro_pr": self.macro_ere_pr,
            "macro_rec": self.macro_ere_rec,
            "macro_ner_f1": self.macro_ner_f1,
            "macro_ner_pr": self.macro_ner_pr,
            "macro_ner_rec": self.macro_ner_rec,
            **perf_metrics,
            **per_class_metrics,
        }

    def __invert__(self):
        return Metrics(**{k: 1 - v if isinstance(v, float) else v for k, v in self.__dict__.items()})


def calculate_f1(metrics: "MetricsDict | list[MetricsDict]", average: str = "micro"):
    if average == "micro":
        if isinstance(metrics, list):
            raise ValueError("Cannot supply a list of metrics for micro calculation")
        return calculate_f1_micro(metrics)
    elif average == "macro":
        if not isinstance(metrics, list):
            raise ValueError("Cannot supply a single metric for macro calculation")
        outputs = torch.tensor([calculate_f1_micro(metric) for metric in metrics]).mean(dim=0)  # torch.float64
        if outputs.dim() == 0:
            return 0., 0., 0.
        return outputs[0].item(), outputs[1].item(), outputs[2].item()
    else:
        raise ValueError(average)


def calculate_f1_micro(metrics: "MetricsDict"):
    tp = torch.tensor(metrics.tp)
    fn = torch.tensor(metrics.fn)
    fp = torch.tensor(metrics.fp)
    zero = torch.tensor(0.)
    precision = tp / (tp + fp) if (tp + fp) > 0 else zero
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else zero
    return precision.item(), recall.item(), f1.item()


from typing import Hashable

_HT = TypeVar('_HT', bound=Hashable)

def _tp_fp_fn_from_sets(out: set[_HT], targets: set[_HT]) -> "MetricsDict":
    return MetricsDict(
        tp=len(set(out) & set(targets)),
        fn=len(set(targets) - set(out)),
        fp=len(set(out) - set(targets)),
    )

@overload
def _non_batched_metrics_ere(
    links: list[_link_type],
    gold_links: list[_link_type],
    use_entity_tag: bool = False,
    average: Literal["micro"] = "micro",
    link_types: Optional[list[int]] = None,
    symmetric_links: Optional[set[int]] = None
) -> "MetricsDict": ...

@overload
def _non_batched_metrics_ere(
    links: list[_link_type],
    gold_links: list[_link_type],
    use_entity_tag: bool = False,
    average: Literal["macro"] = "macro",
    link_types: Optional[list[int]] = None,
    symmetric_links: Optional[set[int]] = None
) -> "dict[int, MetricsDict]": ...

def _non_batched_metrics_ere(
    links: list[_link_type],
    gold_links: list[_link_type],
    use_entity_tag: bool = False,
    average: Literal["micro", "macro"] = "micro",
    link_types: Optional[list[int]] = None,
    symmetric_links: Optional[set[int]] = None
) -> "MetricsDict | dict[int, MetricsDict]":
    if average == "macro":
        if link_types is None:
            raise ValueError("link_types cannot be None when average=\"macro\".")

        macro_metrics = {
            link_type: _non_batched_metrics_ere(
                filtered_links(links, link_type),
                filtered_links(gold_links, link_type),
                use_entity_tag=use_entity_tag,
                average="micro",
                link_types=link_types,
                symmetric_links=symmetric_links,
            )
            for link_type in link_types
        }
        return macro_metrics
    elif average == "micro":
        if not use_entity_tag:
            links = remove_entity_tag_from_links(links)
            gold_links = remove_entity_tag_from_links(gold_links)

        links = remove_extra_info_from_links(links)
        gold_links = remove_extra_info_from_links(gold_links)

        if symmetric_links:
            sym_links = replace_tuple_by_set_for_symrels(links, symmetric_relations=symmetric_links)
            sym_gold_links = replace_tuple_by_set_for_symrels(gold_links, symmetric_relations=symmetric_links)
            return _tp_fp_fn_from_sets(set(sym_links), set(sym_gold_links))

        return _tp_fp_fn_from_sets(set(links), set(gold_links))
    else:
        raise NotImplementedError(average)

def metrics_ere(
        links: list[list[_link_type]],
        gold_links: list[list[_link_type]],
        link_types: list[int],
        use_entity_tag: bool = False,
        average: Literal["micro", "macro"] = "micro",
        batched: bool = False,
        symmetric_links: Optional[set[int]] = None,
):
    if not batched:
        raise ValueError(
            "Using metrics_ere with batched=False is no longer supported, "
            "use _non_batched_metrics_ere instead!"
        )

    if average == "macro":
        assert link_types is not None
        per_batch_macro_metrics = [
            _non_batched_metrics_ere(
                links,
                gold_links,
                use_entity_tag=use_entity_tag,
                average="macro",
                link_types=link_types,
                symmetric_links=symmetric_links,
            )
            for links, gold_links in zip(links, gold_links)
        ]

        return {
            link_type: accumulate_micro_metrics(
                [metrics.get(link_type, MetricsDict.zeros()) for metrics in per_batch_macro_metrics],
            )
            for link_type in link_types
        }
    elif average == "micro":
        return accumulate_micro_metrics(
            [
                _non_batched_metrics_ere(
                    links,
                    gold_links,
                    use_entity_tag=use_entity_tag,
                    average="micro",
                    symmetric_links=symmetric_links
                )
                for links, gold_links in zip(links, gold_links)
            ]
        )
    else:
        raise NotImplementedError(average)


@overload
def _non_batched_metrics_ner(
    entities: list[_entity_type],
    gold_entities: list[_entity_type],
    entity_types: list[int],
    use_entity_tag: bool = True,
    average: Literal["micro"] = "micro",
) -> "MetricsDict": ...


@overload
def _non_batched_metrics_ner(
    entities: list[_entity_type],
    gold_entities: list[_entity_type],
    entity_types: list[int],
    use_entity_tag: bool = True,
    average: Literal["macro"] = "macro",
) -> "dict[int, MetricsDict]": ...


def _non_batched_metrics_ner(
    entities: list[_entity_type],
    gold_entities: list[_entity_type],
    entity_types: list[int],
    use_entity_tag: bool = True,
    average: Literal["micro", "macro"] = "micro",
) -> "MetricsDict | dict[int, MetricsDict]":
    if average == "macro":
        entity_type_keys = entity_types
        if not use_entity_tag:
            entities = remove_entity_tag_from_entities(entities)
            gold_entities = remove_entity_tag_from_entities(gold_entities)
            entity_type_keys = [-1]

        macro_metrics = {
            entity_type: _non_batched_metrics_ner(
                filtered_entities(entities, entity_type),
                filtered_entities(gold_entities, entity_type),
                entity_types=entity_types,
                average="micro",
            )
            for entity_type in entity_type_keys
        }
        return macro_metrics
    elif average == "micro":
        if not use_entity_tag:
            entities = remove_entity_tag_from_entities(entities)
            gold_entities = remove_entity_tag_from_entities(gold_entities)

        return MetricsDict(
            tp=len(set(entities) & set(gold_entities)),
            fn=len(set(gold_entities) - set(entities)),
            fp=len(set(entities) - set(gold_entities)),
        )
    else:
        raise NotImplementedError(average)


def metrics_ner(
        entities: list[list[_entity_type]],
        gold_entities: list[list[_entity_type]],
        entity_types: list[int],
        use_entity_tag: bool = True,
        average = "micro",
        batched: bool = False,
):
    if not batched:
        raise ValueError(
            "Using metrics_ner with batched=False is no longer supported, "
            "use _non_batched_metrics_ner instead!"
        )

    if average == "macro":
        assert entity_types is not None
        per_batch_macro_metrics = [
            _non_batched_metrics_ner(
                entities, gold_entities,
                average="macro",
                entity_types=entity_types,
                use_entity_tag=use_entity_tag
            )
            for entities, gold_entities in zip(entities, gold_entities)
        ]
        entity_type_keys = next(iter(per_batch_macro_metrics)).keys()

        return {
            entity_type: accumulate_micro_metrics(
                [metrics.get(entity_type, MetricsDict.zeros()) for metrics in per_batch_macro_metrics],
            )
            for entity_type in entity_type_keys
        }
    elif average == "micro":
        return accumulate_micro_metrics(
            [
                _non_batched_metrics_ner(
                    entities,
                    gold_entities,
                    entity_types,
                    use_entity_tag=use_entity_tag,
                    average="micro"
                )
                for entities, gold_entities in zip(entities, gold_entities)
            ]
        )
    else:
        raise NotImplementedError(average)


@dataclass
class MetricsDict:
    @classmethod
    def zeros(cls) -> "MetricsDict":
        return cls(0, 0, 0)

    tp: int
    fp: int
    fn: int

    def __add__(self, other: "MetricsDict") -> "MetricsDict":
        if not isinstance(other, MetricsDict):
            raise ValueError("other")

        return MetricsDict(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
        )

    def __iadd__(self, other: "MetricsDict") -> typing_extensions.Self:
        if not isinstance(other, MetricsDict):
            raise ValueError("other")

        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self


def accumulate_micro_metrics(
    metrics: list[MetricsDict],
) -> MetricsDict:
    output = MetricsDict.zeros()
    if len(metrics) == 0:
        return output

    for metric_dict in metrics:
        output += metric_dict
    return output

def accumulate_macro_metrics(
    metrics: list[Mapping[int, MetricsDict]],
    num_classes: int,
) -> MutableMapping[int, MetricsDict]:
    classes: Iterable[int] = range(num_classes)
    if num_classes == 0:
        classes = {-1}

    outcome: dict[int, MetricsDict]
    outcome = {cls: MetricsDict.zeros() for cls in classes}
#    if num_classes == 0:
#        return outcome

    metrics_by_cls: Mapping[int, list[MetricsDict]]
    metrics_by_cls = {
        cls: [metric_dict[cls] for metric_dict in metrics] for cls in outcome.keys()
    }

    return {
        cls: accumulate_micro_metrics(metrics) for cls, metrics in metrics_by_cls.items()
    }


def filtered_entities(entities: list[_entity_type], entity_type: int):
    return [entity for entity in entities if entity[1] == entity_type]


def filtered_links(links: list[_link_type], link_type: int):
    return [link for link in links if link[1] == link_type]

def replace_tuple_by_set_for_symrels(
    links: list[_link_type],
    *,
    symmetric_relations: set[int]
) -> list[_link_type | _symmetric_link_type]:
    return [
        frozenset({head, link_type, tail})
        if link_type in symmetric_relations else
        (head, link_type, tail)
        for head, link_type, tail in links
    ]


def remove_entity_tag_from_links(links: list[_link_type]) -> list[_link_type]:
    return [
        (
            remove_entity_tag_from_entity(link[0]),
            link[1],
            remove_entity_tag_from_entity(link[2])
        )
        for link in links
    ]

def remove_extra_info_from_links(links: list[_link_type]) -> list[_link_type]:
    return [link[:3] for link in links]


def remove_entity_tag_from_entities(entities: list[_entity_type]):
    return [remove_entity_tag_from_entity(ent) for ent in entities]


def remove_entity_tag_from_entity(entity: _entity_type) -> _entity_type:
    return entity[0], -1, entity[2], "<ignored>"
