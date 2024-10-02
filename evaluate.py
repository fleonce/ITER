import os
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path

import torch.cuda
from json import dumps

from iter import ITER
from iter.datasets import CoNLL
from iter.datasets.training import Hparams
from with_argparse import with_argparse, with_opt_argparse
from iter.misc.metrics import average_metrics, format_average_metrics, Metrics, average_into_metrics
from iter.misc.training import evaluate_model


@with_opt_argparse(aliases={
    "model_or_experiment": ["-m", "--model", "--experiment"]
})
def evaluate(
    model_or_experiment: Path,
    dataset: str = None,
    split: str = "test",
    n_times: int = 1,
    batch_size: int = 8,
    jsonify: bool = False,
    threshold: float = None,
    features: list[int] = None,
):
    metrics = None
    if model_or_experiment.is_dir() and (
        (model_or_experiment / "model.safetensors").exists() or (model_or_experiment / "model.safetensors.index.json").exists()
    ):
        model = ITER.from_pretrained(model_or_experiment)
        metrics, _ = perform_evaluate(model, dataset, split, n_times, batch_size, threshold, features)
        if jsonify:
            print(dumps(metrics.to_dict(), indent=2))
    elif model_or_experiment.is_dir() and (model_or_experiment / ".status").exists():
        model_paths = defaultdict(list)
        for parent, child_dirs, child_files in os.walk(model_or_experiment):
            if "config.json" in child_files:
                # we found a model
                model_path = Path(parent).relative_to(model_or_experiment)
                model_path = model_or_experiment / model_path
                model_type = model_path.parent.relative_to(model_or_experiment)
                model_type = model_type.as_posix()
                model_paths[model_type].append(model_path)
        if not model_paths:
            print(f"No models found for experiment {model_or_experiment.name} in {model_or_experiment.as_posix()}")
            exit(1)

        model_metrics = {k: list() for k in model_paths.keys()}
        model_configs = dict()
        model_hparams = dict()
        bound_model_metrics = {k: list() for k in model_paths.keys()}
        for model_type, models in model_paths.items():
            print(f"Testing model type {model_type} ({len(models)}) models ...")
            for model_name in models:
                model = ITER.from_pretrained(model_name)
                model_configs[model_type] = model.config
                metrics, bound_metrics = perform_evaluate(model, dataset, split, n_times, batch_size, threshold, features)
                model_metrics[model_type].append(metrics)
                bound_model_metrics[model_type].append(bound_metrics)
                if model_type not in model_hparams:
                    model_hparams[model_type] = Hparams.from_name(model.config.dataset)

        def best_model_according_to(tp: str, a_models: "list[Metrics]") -> int:
            hparams = model_hparams[tp]
            order_mods = list(range(len(a_models)))
            order_mods.sort(key=lambda x: a_models[x].ner_f1 if hparams.optimize_for != "ere" else a_models[x].ere_f1, reverse=True)
            return order_mods[0]

        best_models = {k: best_model_according_to(k, v) for k, v in model_metrics.items()}
        if not jsonify:
            model_metrics = OrderedDict({k: average_into_metrics(v) for k, v in model_metrics.items()})
            bound_model_metrics = OrderedDict({k: average_into_metrics(v) for k, v in bound_model_metrics.items()})
            for k, v in model_metrics.items():
                print(f"Results for architecture {k}")
                print(v)
                print(bound_model_metrics[k])
            return
        model_metrics = OrderedDict({tp: format_average_metrics(best_models[tp], v, bound_model_metrics[tp]) for tp, v in model_metrics.items()})
        for k, v in model_metrics.items():
            print(f"Results for architecture {k}")
            print(dumps(v, indent=2, ensure_ascii=False))
    else:
        print(f"No model or experiment specified in '{model_or_experiment.as_posix()}'")
        print(f"Trying to load '{model_or_experiment}' ...")
        model = ITER.from_pretrained(model_or_experiment)
        metrics, _ = perform_evaluate(model, dataset, split, n_times, batch_size, threshold, features)
        if jsonify:
            print(dumps(metrics.to_dict(), indent=2))


def perform_evaluate(
    model: ITER,
    dataset: str = None,
    split: str = "test",
    n_times: int = 1,
    batch_size: int = 8,
    threshold: float = None,
    features: list[int] = None,
):
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    model = model.to(device)
    if threshold:
        model.threshold = threshold
    tokenizer = model.tokenizer

    dataset = dataset or model.config.dataset
    dataset = CoNLL.from_name(dataset, tokenizer=tokenizer)
    if features:
        warnings.warn("Changing the model features will possibly break functionality, be aware")
        model.features = model.config.features = dataset.features = sum(2 ** bit for bit in features)
    dataset.setup_dataset()

    metrics, bound_metrics = None, None
    for _ in range(n_times):
        metrics, bound_metrics = evaluate_model(model, dataset, split, batch_size)
        print(metrics)
        print(bound_metrics)
    return metrics, bound_metrics


if __name__ == "__main__":
    evaluate()
