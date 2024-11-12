import os
from datetime import timedelta
from functools import partial
from time import time
from typing import Sequence, TypeVar, Protocol, Optional, cast, Mapping, ParamSpec, Generic

import packaging.version
import torch.cuda
import torch.cuda
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, \
    checkpoint_wrapper, CheckpointImpl
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import Module
from torcheval.metrics import MulticlassConfusionMatrix
from tqdm import tqdm
from transformers.models.bert.modeling_bert import BertModel, BertLayer
from transformers.models.t5.modeling_t5 import T5Block

from iter import ITER, ITERConfig
from iter.data.data_collator import Batch
from iter.datasets import ITERDataset, CoNLL
from iter.datasets.training import Hparams
from iter.generation_utils import decode_actions_and_pairings
from iter.misc.metrics import metrics_ner, calculate_f1, metrics_ere, Metrics, accumulate_macro_metrics, MetricsDict, \
    accumulate_micro_metrics
from iter.tools.dataloaders import DataLoaderProtocol, create_primary_dataloader


T = TypeVar('T', bound=ITER)
P = ParamSpec("P")

class ModelInitProtocol(Protocol[P]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> ITER: ...


class EvaluateProtocol(Protocol):
    def __call__(
        self,
        model: T,
        datasets: Sequence[ITERDataset],
        hparams: Hparams,
        split: str,
        use_tqdm: bool = True,
        use_fsdp: bool = False,
        rank: int = 0,
        world_size: int = 1,
        dataloader_fn: Optional[DataLoaderProtocol] = None
    ) -> tuple[Metrics, Metrics]: ...


def evaluate_model(
    model: ITER,
    datasets: Sequence[ITERDataset],
    hparams: Hparams,
    split: str,
    use_tqdm: bool = True,
    use_fsdp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    dataloader_fn: Optional[DataLoaderProtocol] = create_primary_dataloader,
) -> tuple[Metrics, Metrics]:
    if not dataloader_fn:
        dataloader_fn = create_primary_dataloader

    device = model.device
    primary_dataset = cast(CoNLL, datasets[0])
    model.eval()

    def call_model_decode(bt: dict):
        return decode_actions_and_pairings(
            model, bt["input_ids"], bt["actions"], bt["lr_pair_flag"], bt.get("rr_pair_flag", None), None,
            entity_types=primary_dataset.entity_types, link_types=primary_dataset.link_types
        )

    actions_confusion = MulticlassConfusionMatrix(num_classes=4, device=device)

    metrics = accumulate_macro_metrics([], primary_dataset.num_real_types)
    metrics_no_tag = accumulate_macro_metrics([], 0)
    ere_metrics = accumulate_macro_metrics([], primary_dataset.num_real_links)
    ere_metrics_no_tag = accumulate_macro_metrics([], primary_dataset.num_real_links)
    total_samples = 0
    total_time = 0.
    with torch.no_grad():
        dataloader = dataloader_fn(
            datasets,
            split,
            False,
            hparams,
            world_size,
            rank,
            0,
            use_fsdp,
        )

        batch: Batch
        for batch_idx, batch in enumerate(
            tqdm(dataloader, disable=not use_tqdm, leave=False, desc=f"Evaluating {primary_dataset.name}")
        ):
            batch = batch.to(device)
            if use_fsdp:
                # https://github.com/pytorch/pytorch/issues/82461
                _ = model(**batch)
            t_start = time()
            generation_output = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                entity_types=primary_dataset.entity_types,
                link_types=primary_dataset.link_types,
            )
            attention_mask = batch.attention_mask
            actions_confusion.update(generation_output.actions[attention_mask], batch.actions[attention_mask])

            t_end = time()
            total_time += (t_end - t_start)
            total_samples += batch["input_ids"].size(0)
            gold_decoded_pairings, gold_decoded_links = call_model_decode(batch)

            batch_metrics_kwargs = {
                'entities': generation_output.entities,
                'gold_entities': gold_decoded_pairings,
                'average': 'macro',
                'batched': True,
                'entity_types': list(range(model.num_types - int(model.is_feature_extra_lr_class))),
                'use_entity_tag': True
            }
            batch_metrics = metrics_ner(**batch_metrics_kwargs)
            metrics = accumulate_macro_metrics([metrics, batch_metrics], primary_dataset.num_real_types)
            batch_metrics_no_tag = metrics_ner(**dict(batch_metrics_kwargs, use_entity_tag=False))
            metrics_no_tag = accumulate_macro_metrics([metrics_no_tag, batch_metrics_no_tag], 0)
            if not model.is_feature_ner_only:
                ere_batch_metrics_kwargs = {
                    'links': generation_output.links,
                    'gold_links': gold_decoded_links,
                    'average': 'macro',
                    'batched': True,
                    'link_types': list(range(model.num_links - int(model.is_feature_extra_rr_class))),
                    'use_entity_tag': True,
                    'symmetric_links': primary_dataset.symmetric_link_types,
                }
                if model.is_feature_mimic_pl_marker_eval:
                    ere_batch_metrics_kwargs['symmetric_links'] = set()
                ere_batch_metrics = metrics_ere(**ere_batch_metrics_kwargs)
                ere_metrics = accumulate_macro_metrics([ere_metrics, ere_batch_metrics], primary_dataset.num_links)
                ere_batch_metrics_no_tag = metrics_ere(**dict(ere_batch_metrics_kwargs, use_entity_tag=False))
                ere_metrics_no_tag = accumulate_macro_metrics(
                    [ere_metrics_no_tag, ere_batch_metrics_no_tag],
                    primary_dataset.num_links
                )


    if use_fsdp:
        # accumulate across devices
        total_time_t = torch.tensor(total_time, dtype=torch.float64, device=rank)
        total_samples_t = torch.tensor(total_samples, dtype=torch.long, device=rank)
        dist.all_reduce(total_time_t, dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_t, dist.ReduceOp.SUM)
        total_time = float(total_time)
        total_samples = int(total_samples)

        num_classes = [primary_dataset.num_real_types, 0, primary_dataset.num_real_links, primary_dataset.num_real_links]
        all_metrics = [metrics, metrics_no_tag, ere_metrics, ere_metrics_no_tag]

        all_new_metrics: list[dict[int, MetricsDict]]
        all_new_metrics = [dict() for _ in range(len(all_metrics))]
        infos = torch.zeros((len(num_classes), max(num_classes), 3), dtype=torch.int, device=rank)

        for i, (metric, num_class) in enumerate(
            zip(
                all_metrics,
                num_classes
            )
        ):
            if num_class == 0:
                keys = [-1]
            else:
                keys = list(range(num_class))
            for key in keys:
                infos[i, key, 0] += metric[key].tp
                infos[i, key, 1] += metric[key].fp
                infos[i, key, 2] += metric[key].fn
        dist.all_reduce(infos, dist.ReduceOp.SUM)

        for i, (metric, num_class) in enumerate(
            zip(
                all_new_metrics,
                num_classes
            )
        ):
            if num_class == 0:
                keys = [-1]
            else:
                keys = list(range(num_class))
            for key in keys:
                metric[key] = MetricsDict(
                    tp=int(infos[i, key, 0]),
                    fp=int(infos[i, key, 1]),
                    fn=int(infos[i, key, 2]),
                )
        metrics, metrics_no_tag, ere_metrics, ere_metrics_no_tag = all_new_metrics

    def calculate_metrics(
        ner: Mapping[int, MetricsDict],
        ere: Mapping[int, MetricsDict],
    ) -> Metrics:
        per_class = {k: calculate_f1(v, average="micro") for k, v in ner.items()}
        m_pr, m_rec, m_f1 = calculate_f1(list(ner.values()), average="macro")
        pr, rec, f1 = calculate_f1(accumulate_micro_metrics(list(ner.values())), average="micro")

        ere_per_class = {k: calculate_f1(v, average="micro") for k, v in ere.items()}
        ere_m_pr, ere_m_rec, ere_m_f1 = calculate_f1(list(ere.values()), average="macro")
        ere_pr, ere_rec, ere_f1 = calculate_f1(accumulate_micro_metrics(list(ere.values())), average="micro")

        return Metrics(
            pr, rec, f1,
            ere_pr, ere_rec, ere_f1,
            m_pr, m_rec, m_f1,
            ere_m_pr, ere_m_rec, ere_m_f1,
            per_class,
            ere_per_class,
            samples_per_second=total_samples / total_time,
            samples_total_time=total_time,
            samples_total_num=total_samples,
            ner_per_class_names=primary_dataset.entity_types,
            ere_per_class_names=primary_dataset.link_types,
            loss=None,
            measurements=(actions_confusion.compute(),)
        )

    return calculate_metrics(metrics, ere_metrics), calculate_metrics(metrics_no_tag, ere_metrics_no_tag)


def do_setup(rank, world_size):
    if dist.is_initialized():
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=60))


def do_cleanup():
    dist.destroy_process_group()


def mixed_precision_policy(use_bfloat16: bool):
    is_bfloat16_supported = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    if is_bfloat16_supported and use_bfloat16:
        return MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    return None


def fsdp_model_init(
    config: ITERConfig,
    model_init_fn: ModelInitProtocol,
    model_init_kwargs: P.kwargs,
    use_bfloat16: bool = True,
    use_hsdp: bool = False,
    use_activation_checkpointing: bool = True,
):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    do_setup(rank, world_size)
    torch.cuda.set_device(rank)

    # initialize the model on CPU
    model_init_kwargs = model_init_kwargs or dict()
    if model_init_kwargs:
        model_init_fn = partial(model_init_fn, **model_init_kwargs)
    else:
        model_init_fn = partial(model_init_fn, config=config)
    if rank == 0:
        model = model_init_fn()
        # assert False, model.reset_parameters
    else:
        with torch.device("meta"):
            model = model_init_fn()

    sharding_strategy = ShardingStrategy.FULL_SHARD
    if use_hsdp:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD

    wrapping_block = T5Block
    if model.config.guess_model_class() == BertModel:
        wrapping_block = BertLayer
    wrapping_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={wrapping_block})

    def param_fn(module: Module):
        model.to_empty(device=torch.device("cuda"), recurse=False)

    # convert model to FSDP model
    model = cast(ITER, FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy(use_bfloat16),
        use_orig_params=False,
        device_id=rank,
        limit_all_gathers=True,
        sync_module_states=True,
        param_init_fn=param_fn
    ))

    if use_activation_checkpointing:
        non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

        def selective_activation_checkpoint(submodule):
            return isinstance(submodule, wrapping_block)

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_activation_checkpoint
        )

    return model, rank, world_size
