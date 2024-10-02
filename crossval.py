import glob
import json
from pathlib import Path

from with_argparse import with_argparse
from iter.misc.metrics import average_metrics

from iter.datasets import CoNLL
from iter.datasets.training import Hparams
from train import do_train as train 

@with_argparse
def main(
    transformer: str,
    dataset: list[str],
    model_path: str = None,
    log_file: Path = None,
    log_append: bool = False,
    seed: int = 42,
    use_tqdm: bool = True,
    use_bfloat16: bool = False,
    use_fsdp: bool = False,
    num_epochs: int = 0,
    verbose: bool = True,
    visualize: bool = False,
    dont_ckpt: bool = False,
    do_compile: bool = False,
    show_bound_metrics: bool = False,
    features: list[int] = None,
    root_dir: Path = Path.cwd(),
):
    train_kwargs = dict(
        transformer=transformer,
        dataset=dataset,
        model_path=model_path,
        log_file=log_file,
        log_append=log_append,
        seed=seed,
        use_tqdm=use_tqdm,
        use_bfloat16=use_bfloat16,
        use_fsdp=use_fsdp,
        num_epochs=num_epochs,
        verbose=verbose,
        visualize=visualize,
        dont_ckpt=dont_ckpt,
        do_compile=do_compile,
        show_bound_metrics=show_bound_metrics,
        features=features,

    )

    if isinstance(dataset, list) and len(dataset) == 1:
        paths = glob.glob(dataset[0], root_dir=root_dir / "cfg", recursive=False)
        paths = [Path(path) for path in paths]
        paths = [path.as_posix()[:-len(path.suffix)] for path in paths]
        paths.sort()
        print("paths",paths)
    else:
        paths = dataset

    metrics = []
    for path in paths:
        path_kwargs = dict(train_kwargs, dataset=CoNLL.from_name(path), hparams=Hparams.from_name(path))
        print(path_kwargs)
        path_metrics = train(**path_kwargs)
        metrics.append(path_metrics)
    avg_metrics = average_metrics(metrics)
    avg_metrics = {k: f"{mean:.6f} ± {std:.3f}" for k, (mean, std) in avg_metrics.items()}
    print(f"Cross validation for {len(paths)} runs")
    print(json.dumps(avg_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
