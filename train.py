import collections
import contextlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Optional, TypeVar, Sequence, cast, MutableMapping, Any, ContextManager

import torch.cuda
import torch.distributed as dist
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from tqdm import tqdm
from with_argparse import with_argparse

from iter import ITER, ITERConfig
from iter.datasets import CoNLL, ITERDataset
from iter.datasets.training import Hparams
from iter.misc.metrics import Metrics, average_metrics
from iter.misc.seeding import setup_seed
from iter.misc.util import get_commit_hash, is_clean_working_tree, get_working_tree_diff
from iter.misc.training import evaluate_model, fsdp_model_init, EvaluateProtocol, ModelInitProtocol
from iter import ITERForRelationExtraction
from iter.optimizing_iter import get_grouped_parameters, get_scheduler_lambda
from iter.tools.dataloaders import create_primary_dataloader, DataLoaderProtocol

C = TypeVar('C', bound=ITERConfig)
T = TypeVar('T', bound=ITER)


@with_argparse
def main(
    transformer: str,
    dataset: str,
    model_path: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_append: bool = False,
    seed: Optional[list[int]] = None,
    use_tqdm: bool = True,
    use_bfloat16: bool = False,
    use_fsdp: bool = False,
    num_epochs: int = 0,
    verbose: bool = True,
    dont_ckpt: bool = False,
    do_compile: bool = False,
    show_bound_metrics: bool = False,
    load_ckpt: Optional[Path] = None,
    features: Optional[list[int]] = None,
):
    hparams = Hparams.from_name(dataset)
    conll_dataset = CoNLL.from_name(dataset)

    if features:
        conll_dataset.features = sum([2 ** f for f in features])

    if conll_dataset.is_feature_ner_only and hparams.optimize_for != "ner":
        hparams.optimize_for = "ner"

    train_seed = seed or 42
    if isinstance(seed, list) and len(seed) == 1:
        train_seed = seed[0]

    result = do_train(
        transformer=transformer,
        datasets=[conll_dataset],
        hparams=hparams,
        model_path=model_path,
        log_file=log_file,
        log_append=log_append,
        seed=train_seed,
        use_tqdm=use_tqdm,
        use_bfloat16=use_bfloat16,
        use_fsdp=use_fsdp,
        num_epochs=num_epochs,
        verbose=verbose,
        dont_ckpt=dont_ckpt,
        do_compile=do_compile,
        show_bound_metrics=show_bound_metrics,
        load_ckpt=load_ckpt,
    )
    if result and isinstance(seed, list):
        formatted_avg_metrics = {k: f"{mean:.6f} ± {std:.3f}" for k, (mean, std) in result.to_dict().items()}
        print(json.dumps(formatted_avg_metrics, indent=2, ensure_ascii=False))


def save_checkpoint(
    f1: float | torch.Tensor,
    model: ITER,
    model_path: Path | str,
    hparams: Hparams,
    logger: Logger,
    rank: int,
    verbose: bool,
    use_fsdp: bool,
    disable: bool = False,
):
    if not disable:
        save_pretrained_kwargs = {}
        if use_fsdp:
            with FSDP.state_dict_type(
                model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                state_dict = model.state_dict()
                if rank == 0:
                    save_pretrained_kwargs["state_dict"] = state_dict
                    save_pretrained_kwargs["is_main_process"] = True
        if verbose:
            logger.info(
                f"Found new best checkpoint ({hparams.optimize_for}, {f1:.8f}), saving to {model_path}"
            )
        if not use_fsdp or rank == 0:
            model.save_pretrained(model_path, **save_pretrained_kwargs)
    elif verbose:
        logger.info(f"Found new best checkpoint ({hparams.optimize_for}, {f1:.8f}), but checkpointing is disabled")


def setup_model_path(
    transformer: str,
    primary_dataset: CoNLL,
    use_fsdp: bool,
) -> str:
    if use_fsdp:
        raise ValueError(
            f"For FSDP, all ranks must checkpoint from/to the same directory, "
            f"use train.py with --model_path or run_experiment with --log_ckpts"
        )
    transformer_as_path = Path(transformer)
    date = datetime.now()
    date_fmt = date.strftime('%Y-%m-%d_%H-%M-%S')
    if transformer_as_path.is_dir() and transformer_as_path.exists():
        model_path = f"models/{primary_dataset.name}/local_{transformer_as_path.name}/{date_fmt}"
    else:
        model_path = f"models/{primary_dataset.name}/{transformer}/{date_fmt}"
    return model_path


def setup_logging(
    name: str,
    model_path: str,
    log_file: Optional[Path],
    log_append: bool = False,
):
    log_handlers: list[logging.StreamHandler]
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if not log_file:
        log_file = Path(f"{model_path}/train.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_append = log_file.exists()
    if log_file and ('RANK' not in os.environ or int(os.environ['RANK']) == 0):
        log_handlers.append(logging.FileHandler(log_file, mode='w' if not log_append else 'a'))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=log_handlers,
    )
    return logging.getLogger(name)


def reproducibility_logging(
    seed: int,
    logger: Logger,
    primary_dataset: CoNLL,
    use_bfloat16: bool,
    use_fsdp: bool = False,
):
    logger.info(
        f"Python Version = {sys.version} "
        f"on {platform.system()} {platform.version()} ({platform.platform()})"
    )
    logger.info(
        f"Torch Version = {torch.__version__} "
        f"(CUDA {torch.version.cuda}) with bfloat16 = {use_bfloat16} "
        f"(Git commit = {torch.version.git_version})"
    )
    if is_clean_working_tree():
        logger.info(f"Git commit = {get_commit_hash()} on {socket.gethostname()} (tree is clean)")
    else:
        logger.info(f"Git commit = {get_commit_hash()} on {socket.gethostname()} (tree is dirty)")
        logger.warning(get_working_tree_diff())
    logger.info(f"Seed = {seed}")
    logger.info(f"Dataset = {primary_dataset.file_path}")
    if use_fsdp:
        logger.info(f"Rank = {os.environ['RANK']}")


def train_logging(
    model: T,
    hparams: Hparams,
    logger: Logger,
    primary_dataset: CoNLL,
    load_ckpt: Optional[Path],
):
    logger.info(model)
    logger.info(hparams.to_json() + f" for dataset {primary_dataset.file_path}")
    if load_ckpt is not None:
        logger.info(f"Loaded model from {load_ckpt.as_posix()}")
    numel = sum(param.numel() for param in model.parameters())
    no_grad_numel = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    logger.info(str(numel / 1e6) + f" M params ({numel} total)")
    if no_grad_numel > 0:
        logger.info(str((numel - no_grad_numel) / 1e6) + " M activated params")
    model.list_features(logger)


def setup_precision_context(
    use_bfloat16: bool,
    use_fsdp: bool,
) -> ContextManager:
    if use_bfloat16 and not use_fsdp:
        return torch.cuda.amp.autocast(
            enabled=True, dtype=torch.bfloat16
        )
    return contextlib.nullcontext()


def write_reproducibility_checkpoint(
    model_path: str,
    model: Module,
    optimizer: Optimizer,
    hparams: Hparams,
    num_epochs: int
):
    repro_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "hparams": hparams,
        "num_epochs": num_epochs
    }

    with open(f"{model_path}/requirements.txt", "w") as f:
        subprocess.run("python -m pip freeze".split(), stdout=f)

    torch.save(repro_dict, f"{model_path}/repro.pt")


def load_checkpoint(
    model: T,
    model_class: type[T],
    model_ckpt: Path,
):
    loaded_model = model_class.from_pretrained(model_ckpt)
    with torch.no_grad():
        if loaded_model.num_types != model.num_types:
            loaded_model.resize_num_types(model.num_types)
        if (
            not loaded_model.is_feature_ner_only
            and loaded_model.num_links != model.num_links
        ):
            loaded_model.resize_num_links(model.num_links)
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict)


def do_train(
    seed: list[int] | int,
    **train_kwargs,
) -> Optional[Metrics]:
    if isinstance(seed, list):
        seeds = seed
        metrics = []
        for seed in seeds:
            seed_kwargs = dict(train_kwargs, seed=seed)
            seed_metrics = do_train_impl(**seed_kwargs)
            if seed_metrics:
                metrics.append(seed_metrics)
        avg_metrics = average_metrics(metrics)
        mean_metrics = {k: mean for k, (mean, std) in avg_metrics.items()}
        return Metrics.from_dict(mean_metrics)
    else:
        return do_train_impl(**train_kwargs, seed=seed)


def do_train_impl(
    *,
    transformer: str,
    datasets: Sequence[ITERDataset],
    hparams: Hparams,
    model_class: Optional[type[T]] = None,
    model_init: Optional[ModelInitProtocol] = None,
    config_class: Optional[type[C]] = None,
    model_path: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_append: bool = False,
    seed: int = 42,
    use_tqdm: bool = True,
    use_bfloat16: bool = False,
    use_fsdp: bool = False,
    num_epochs: int = 0,
    verbose: bool = True,
    dont_ckpt: bool = False,
    do_compile: bool = False,
    show_bound_metrics: bool = False,
    load_ckpt: Optional[Path] = None,
    evaluate_fn: EvaluateProtocol = evaluate_model,
    dataloader_fn: DataLoaderProtocol = create_primary_dataloader
) -> Optional[Metrics]:
    """
    Training implementation for ITER.

    Sets up datasets, seeding, reproducibility, model.
    Contains the main training loop, handles evaluation too.
    """
    setup_seed(seed)

    primary_dataset = cast(CoNLL, datasets[0])
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    rank, world_size = 0, 1
    if not model_path:
        model_path = setup_model_path(
            transformer,
            primary_dataset,
            use_fsdp,
        )

    logger = setup_logging(
        "train",
        model_path,
        log_file,
        log_append
    )

    # log seed, torch and cuda version, whether we use mixed precision, etc...
    reproducibility_logging(
        seed,
        logger,
        primary_dataset,
        use_bfloat16,
        use_fsdp
    )

    if config_class is None:
        config_class = ITERConfig

    config = config_class(
        transformer,
        transformer_config={"max_length": primary_dataset.max_length},
        num_types=primary_dataset.num_types,
        num_links=primary_dataset.num_links,
        features=primary_dataset.features,
        dataset=primary_dataset.name,
        max_nest_depth=primary_dataset.entity_nest_depth,
        dropout=hparams.dropout,
        activation_fn=hparams.activation_fn,
        d_ff=hparams.d_ff,
        entity_types=primary_dataset.entity_types,
        link_types=primary_dataset.link_types,
    )

    if model_class is None:
        model_class = ITER if config.is_feature_ner_only else ITERForRelationExtraction
    if model_init is None:
        model_init = model_class

    if use_fsdp:
        assert load_ckpt is None, f"Loading a checkpoint in FSDP is currently not supported"
        model, rank, world_size = fsdp_model_init(
            config,
            model_init_fn=model_init,
            model_init_kwargs=dict(),
            use_bfloat16=use_bfloat16,
            use_hsdp=False,
            use_activation_checkpointing=True,
        )
        device = f"cuda:{rank}"
    else:
        model = model_class(config)  # ITER(config)
        if load_ckpt:
            load_checkpoint(model, model_class, load_ckpt)

    if verbose:
        # log model arch, hparams, # params, model features
        train_logging(
            model,
            hparams,
            logger,
            primary_dataset,
            load_ckpt
        )

    if do_compile:
        model = torch.compile(model, fullgraph=False, dynamic=True)
    primary_dataset.tokenizer = model.tokenizer

    is_hpo = primary_dataset.is_hpo_dataset()
    features = primary_dataset.features

    for ds in datasets:
        if is_hpo:
            ds.setup_hpo()
        ds.features = features
        ds.setup_dataset()
        if verbose:
            ds.list_splits(logger)

    compute_loss_context_mngr = setup_precision_context(
        use_bfloat16,
        use_fsdp,
    )
    # calculate total train steps ahead of time
    num_samples_per_epoch = len(primary_dataset["train"])
    effective_batch_size = hparams.batch_size * hparams.gradient_accumulation
    num_steps = (hparams.max_epochs * num_samples_per_epoch) // effective_batch_size

    optimizer = torch.optim.AdamW(get_grouped_parameters(model, hparams), fused=False, weight_decay=0.1, lr=1e-4)
    lr_scheduler = get_scheduler_lambda(hparams.lr_scheduler, hparams.warmup_steps, num_steps)
    task_lr_scheduler = get_scheduler_lambda(hparams.task_lr_scheduler, hparams.task_warmup_steps, num_steps)
    lr_scheduler = LambdaLR(optimizer, [lr_scheduler, lr_scheduler, task_lr_scheduler, task_lr_scheduler])

    write_reproducibility_checkpoint(
        model_path=model_path,
        model=model,
        optimizer=optimizer,
        hparams=hparams,
        num_epochs=num_epochs
    )

    if not use_fsdp:
        model = model.to(device)

    def train_step():
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=False)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return norm

    best_f1 = 0.0
    best_f1_epoch = None
    outcomes = defaultdict(list)

    train_epochs = min(num_epochs or hparams.num_epochs, hparams.max_epochs)
    for epoch in range(train_epochs):
        model.train()

        # create dataloader based on hparams, world size & rank (when use_fsdp=True), seed
        dataloader = dataloader_fn(
            datasets,
            "train",
            True,
            hparams,
            world_size,
            rank,
            seed,
            use_fsdp
        )

        stats = torch.zeros(3, device=device)
        steps, grad_steps = 0, 0
        postfix: MutableMapping[str, Any]
        postfix = collections.OrderedDict()
        for batch in (
            tq := tqdm(
                dataloader,
                leave=False,
                desc=f"Training - Epoch {epoch}/{num_epochs or hparams.num_epochs} ({world_size} GPUs)",
                disable=not use_tqdm,
            )
        ):
            batch_size = batch.input_ids.size(0)
            with compute_loss_context_mngr:
                output = model(**batch.to(device))
                loss = output.loss
            loss.backward()
            stats[0] += loss.detach()
            stats[1] += batch_size

            grad_steps += 1
            steps += 1
            if grad_steps >= hparams.gradient_accumulation:
                stats[2] += train_step().detach()
                grad_steps = 0

            if steps % 20 == 0:
                if use_fsdp:
                    dist.all_reduce(stats, dist.ReduceOp.SUM)
                train_loss = stats[0] / stats[1]
                g_norm = stats[2] / stats[1]
                stats.zero_()
                postfix.update(loss=train_loss.item(), g_norm=g_norm.item())

            tq.set_postfix(postfix)
        if grad_steps > 0 and False:  # mimic original training, this should not be necessary
            train_step()

        torch.cuda.empty_cache()
        time.sleep(0.1)
        torch.cuda.empty_cache()

        val_metrics, val_metrics_no_tag = evaluate_fn(
            model,
            datasets,
            hparams,
            split="eval",
            use_tqdm=use_tqdm,
            use_fsdp=use_fsdp,
            rank=rank,
            world_size=world_size,
            dataloader_fn=dataloader_fn,
        )

        torch.cuda.empty_cache()
        time.sleep(0.1)
        torch.cuda.empty_cache()

        if verbose:
            if show_bound_metrics:
                logger.info(repr(val_metrics) + " " + repr(val_metrics_no_tag))
            else:
                logger.info(val_metrics)
                torch.set_printoptions(sci_mode=False)
                logger.info("Action confusion: \n" + str(val_metrics.measurements[0]))
        outcome_f1 = (val_metrics.ere_f1 if hparams.optimize_for == "ere" else val_metrics.ner_f1) \
            if hparams.metric_average == "micro" \
            else (val_metrics.macro_ere_f1 if hparams.optimize_for == "ere" else val_metrics.macro_ner_f1)
        if outcome_f1 > best_f1:
            best_f1 = outcome_f1
            best_f1_epoch = epoch
            # save a checkpoint
            save_checkpoint(
                best_f1,
                model,
                model_path,
                hparams,
                logger,
                rank,
                verbose,
                use_fsdp,
                dont_ckpt
            )
        for k, v in val_metrics.to_dict().items():
            outcomes[k].append(v)
        if (best_f1_epoch or 0) + hparams.patience <= epoch:
            logger.info(f"No improvement for {hparams.patience} epochs, aborting training in epoch {epoch}")
            break

    if best_f1_epoch is None and not dont_ckpt:
        logger.info(f"Model did not train, aborting ...")
        if use_fsdp:
            dist.barrier()
        return None

    if use_fsdp:
        logger.info(f"Waiting at save barrier")
        dist.barrier()
    if not dont_ckpt:
        del model
        torch.cuda.empty_cache()

        if use_fsdp:
            model, _, _ = fsdp_model_init(
                config,
                model_init_fn=cast(ModelInitProtocol, model_class.from_pretrained),
                model_init_kwargs={"pretrained_model_name_or_path": model_path},
                use_bfloat16=use_bfloat16,
                use_hsdp=False,
                use_activation_checkpointing=True,
            )
        else:
            model = model_class.from_pretrained(model_path)
            model = model.to(device)
    logger.info(f"TESTING")
    test_metrics, test_metrics_bounds = evaluate_fn(
        model,
        datasets,
        hparams,
        split="test",
        use_tqdm=use_tqdm,
        use_fsdp=use_fsdp,
        rank=rank,
        world_size=world_size,
        dataloader_fn=dataloader_fn,
    )
    logger.info(test_metrics)
    logger.info(test_metrics_bounds)

    metrics_path = Path(model_path) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump({
            "test_metrics": test_metrics.to_dict(),
            "test_metrics_bounds": test_metrics_bounds.to_dict(),
            "metrics": outcomes,
        }, f)

    del model
    torch.cuda.empty_cache()
    return test_metrics


if __name__ == "__main__":
    main()
