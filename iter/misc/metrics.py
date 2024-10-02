import dataclasses
import itertools
from collections import defaultdict, OrderedDict
from typing import Mapping, Optional

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


def average_metrics(metrics: "list[Metrics]") -> Mapping[str, tuple[float, float]]:
    outcomes = defaultdict(list)
    for metric in metrics:
        for key, value in metric.to_dict().items():
            outcomes[key].append(value)
    outcomes = {k: torch.tensor(v) for k, v in outcomes.items()}
    outcomes = {k: std_mean_or_zero(v) for k, v in outcomes.items()}
    outcomes = {k: (mean.item(), std.item()) for k, (std, mean) in outcomes.items()}
    return outcomes


def average_into_metrics(metrics: "list[Metrics]") -> "Metrics":
    outcomes = average_metrics(metrics)
    outcomes = {k: mean for k, (mean, std) in outcomes.items()}
    return Metrics.from_dict(outcomes)



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
    outcomes_of_all = {k: tuple(itertools.chain.from_iterable(v)) for k, v in outcomes_of_all.items()}
    return OrderedDict({
        k: f"{mean:.6f} ± {std_dev:.4f} / "
           f"{bound_mean:.6f} ± {bound_std:.4f} "
           f"(best is {best:.6f} / {bound_best:.6f})"
        for k, (mean, std_dev, best, _, bound_mean, bound_std, bound_best, _) in outcomes_of_all.items()
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

    ner_per_class: "dict[int, tuple[float, float, float]]" = None
    ere_per_class: "dict[int, tuple[float, float, float]]" = None
    ner_per_class_names: list[str] = None
    ere_per_class_names: list[str] = None

    samples_per_second: float = 0.
    samples_total_num: int = 0.
    samples_total_time: float = 0.

    loss: float = 0.
    measurements: Optional[tuple[Tensor, ...]] = None


    def __repr__(self):
        return (
            f"NER :: pr={self.ner_pr:.6f} rec={self.ner_rec:.6f} f1={self.ner_f1:.6f} "
            f"ERE :: pr={self.ere_pr:.6f} rec={self.ere_rec:.6f} f1={self.ere_f1:.6f} "
            f"PERF :: {self.samples_per_second}/s"
        )

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

        per_class_metrics = join_dicts(dict(), *[
            {
                k.lower() + "_pr": self.ner_per_class[i][0],
                k.lower() + "_rec": self.ner_per_class[i][1],
                k.lower() + "_f1": self.ner_per_class[i][2],
            } for i, k in enumerate(self.ner_per_class_names)
        ])

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


def calculate_f1(metrics: dict | list = None, average: str = "micro"):
    if average == "micro":
        return calculate_f1_micro(metrics)
    elif average == "macro":
        outputs = torch.tensor([calculate_f1_micro(metric) for metric in metrics]).mean(dim=0)  # torch.float64
        if outputs.dim() == 0:
            return 0., 0., 0.
        return outputs[0].item(), outputs[1].item(), outputs[2].item()
    else:
        raise ValueError(average)


def calculate_f1_micro(metrics: dict = None):
    tp = torch.tensor(metrics["tp"])
    fn = torch.tensor(metrics["fn"])
    fp = torch.tensor(metrics["fp"])
    zero = torch.tensor(0.)
    precision = tp / (tp + fp) if (tp + fp) > 0 else zero
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else zero
    return precision.item(), recall.item(), f1.item()


def metrics_ere(
        links: list[_link_type] | list[list[_entity_type]],
        gold_links: list[_link_type] | list[list[_entity_type]],
        use_entity_tag: bool = False,
        average: str = "micro",
        batched: bool = False,
        link_types: list[str | int] = None,
        symmetric_links: set[str] = None,
):
    if average == "macro":
        assert link_types is not None
        if batched:
            per_batch_macro_metrics = [
                metrics_ere(
                    links, gold_links,
                    use_entity_tag=use_entity_tag,
                    batched=False,
                    average="macro",
                    link_types=link_types,
                    symmetric_links=symmetric_links,
                )
                for links, gold_links in zip(links, gold_links)
            ]

            return {
                link_type: accumulate_metrics(
                    [metrics.get(link_type, {}) for metrics in per_batch_macro_metrics],
                    average="micro",
                )
                for link_type in link_types
            }
        macro_metrics = {
            link_type: metrics_ere(
                filtered_links(links, link_type),
                filtered_links(gold_links, link_type),
                use_entity_tag=use_entity_tag,
                batched=False,
                average="micro",
                link_types=link_types,
                symmetric_links=symmetric_links,
            )
            for link_type in link_types
        }
        return macro_metrics

    if batched:
        return accumulate_metrics([
            metrics_ere(
                links,
                gold_links,
                use_entity_tag=use_entity_tag,
                batched=False,
                average=average,
                symmetric_links=symmetric_links
            )
            for links, gold_links in zip(links, gold_links)
        ], average, num_classes=len(link_types))

    if not use_entity_tag:
        links = remove_entity_tag_from_links(links)
        gold_links = remove_entity_tag_from_links(gold_links)

    links = remove_extra_info_from_links(links)
    gold_links = remove_extra_info_from_links(gold_links)

    if symmetric_links:
        links = replace_tuple_by_set_for_symrels(links, symmetric_relations=symmetric_links)
        gold_links = replace_tuple_by_set_for_symrels(gold_links, symmetric_relations=symmetric_links)

    return {
        "tp": len(set(links) & set(gold_links)),
        "fn": len(set(gold_links) - set(links)),
        "fp": len(set(links) - set(gold_links)),
    }


def metrics_ner(
        entities: list[_entity_type] | list[list[_entity_type]],
        gold_entities: list[_entity_type] | list[list[_entity_type]],
        use_entity_tag: bool = True,
        average = "micro",
        batched: bool = False,
        entity_types: list[str | int] = None
):
    if average == "macro":
        assert entity_types is not None
        if batched:
            per_batch_macro_metrics = [
                metrics_ner(
                    entities, gold_entities,
                    batched=False,
                    average="macro",
                    entity_types=entity_types,
                    use_entity_tag=use_entity_tag
                )
                for entities, gold_entities in zip(entities, gold_entities)
            ]
            entity_type_keys = next(iter(per_batch_macro_metrics)).keys()

            return {
                entity_type: accumulate_metrics(
                    [metrics.get(entity_type, {}) for metrics in per_batch_macro_metrics],
                    average="micro",
                    num_classes=len(entity_types)
                )
                for entity_type in entity_type_keys
            }

        entity_type_keys = entity_types
        if not use_entity_tag:
            entities = remove_entity_tag_from_entities(entities)
            gold_entities = remove_entity_tag_from_entities(gold_entities)
            entity_type_keys = [-1]

        macro_metrics = {
            entity_type: metrics_ner(
                filtered_entities(entities, entity_type),
                filtered_entities(gold_entities, entity_type),
                batched=False,
                average="micro",
            )
            for entity_type in entity_type_keys
        }
        return macro_metrics

    if batched:
        return accumulate_metrics([
            metrics_ner(entities, gold_entities, batched=False)
            for entities, gold_entities in zip(entities, gold_entities)
        ], average, num_classes=len(entity_types))

    if not use_entity_tag:
        entities = remove_entity_tag_from_entities(entities)
        gold_entities = remove_entity_tag_from_entities(gold_entities)

    return {
        "tp": len(set(entities) & set(gold_entities)),
        "fn": len(set(gold_entities) - set(entities)),
        "fp": len(set(entities) - set(gold_entities))
    }


def accumulate_metrics(
        metrics: list[dict] | list[dict[int, dict]],
        average: str,
        num_classes: int = None
) -> dict:
    if len(metrics) == 0:
        if average == "macro":
            if num_classes == 0:
                return {-1: {"tp": 0, "fp": 0, "fn": 0}}
            return {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in range(num_classes)}
        return {"tp": 0, "fp": 0, "fn": 0}
    if average == "macro":
        return {a: accumulate_metrics([item[a] for item in metrics], "micro") for a in metrics[0].keys()}
    output = {k: 0 for k in metrics[0].keys()}
    for metric_dict in metrics:
        for k, v in metric_dict.items():
            output[k] += v
    return output


def filtered_entities(entities: list[_entity_type], entity_type: int):
    return [entity for entity in entities if entity[1] == entity_type]


def filtered_links(links: list[_link_type], link_type: int):
    return [link for link in links if link[1] == link_type]

def replace_tuple_by_set_for_symrels(links: list[_link_type], *, symmetric_relations: set[str]) -> list[_link_type]:
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
