from pathlib import Path

import torch
from tqdm import tqdm

from iter import ITER
from iter.datasets import CoNLL
from with_argparse import with_argparse, with_opt_argparse
from iter.misc.seeding import setup_seed
from iter.misc.training import evaluate_model


@with_opt_argparse(setup_cwd=True)
def find_optimal_lambda(
    models: list[str],
    metrics: list[str],
    dataset: str,
    output_file: Path,
    seed: int = 42,
    step: float = 0.05
):
    setup_seed(seed)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    thresholds = torch.arange(0.0, 1.0, step=step)

    measurements = torch.zeros((len(models), len(metrics), thresholds.numel()))
    for model_idx, model in enumerate(tqdm(models, position=1, leave=False)):
        model: str
        model: ITER = ITER.from_pretrained(model).to(device)
        tokenizer = model.tokenizer
        dataset = dataset
        if isinstance(dataset, str):
            dataset = CoNLL.from_name(dataset, tokenizer=tokenizer)
            dataset.setup_dataset()
        for threshold_idx, threshold in enumerate(tqdm(thresholds, leave=False)):
            model.threshold = threshold
            test_metrics, _ = evaluate_model(model, dataset, "test")
            for metric_idx, metric in enumerate(metrics):
                val = getattr(test_metrics, metric)
                measurements[model_idx, metric_idx, threshold_idx] = val

    with output_file.open('wb') as f:
        torch.save({
            "models": models,
            "metrics": metrics,
            "step_size": step,
            "measurements": measurements,
        }, f)
    print(f"Saved measurements to {output_file}")


find_optimal_lambda()
