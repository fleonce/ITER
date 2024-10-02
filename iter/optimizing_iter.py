import math

from . import ITER
from .datasets.training import Hparams


def get_parameters(model: ITER, named=True):
    base_model_params, task_params = [], []
    for name, param in model.named_parameters():
        if "model." in name:
            base_model_params.append((name, param) if named else param)
        else:
            task_params.append((name, param) if named else param)
    return base_model_params, task_params


# partially from https://github.com/lyutyuh/ASP/blob/31ac48dfe9d85cb0b3ad22d43667104db38f2dc2/util/func.py#L399-L413
def get_scheduler_lambda(scheduler_type: str, warmup_steps: float | int, total_steps: int):
    if isinstance(warmup_steps, float):
        # convert ratio to integer
        return get_scheduler_lambda(scheduler_type, int(warmup_steps * total_steps), total_steps)
    if scheduler_type == 'linear':
        return get_scheduler_lambda('linear_with_warmup', 0, total_steps)
    elif scheduler_type == 'linear_with_warmup':
        def lambda_rule(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            return max(
                0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps))
            )

        return lambda_rule
    elif scheduler_type == 'constant':
        return lambda step: 1.0
    elif scheduler_type == 'constant_with_warmup':
        return lambda step: min(1.0, float(step) / float(max(1, warmup_steps)))
    elif scheduler_type == 'inverse_sqrt':
        return get_scheduler_lambda('inverse_sqrt_with_warmup', 0, total_steps)
    elif scheduler_type == 'inverse_sqrt_with_warmup':
        def lambda_rule(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            timescale = warmup_steps or total_steps / 100
            shift = timescale - warmup_steps
            decay = 1.0 / math.sqrt((step + shift) / timescale)
            return decay
        return lambda_rule
    else:
        raise ValueError(f'Unknown scheduler type {scheduler_type}')


def get_grouped_parameters(model: ITER, hparams: Hparams):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    base_model_params, task_params = get_parameters(model, named=True)
    weight_decay = hparams.weight_decay  # hparams.get("weight_decay", 0.1)
    task_weight_decay = hparams.task_weight_decay  # self.hparams.get("task_weight_decay", 0.1)
    # no weight decay on bias and LayerNorm weight
    # see official pytorch implementation:
    # https://github.com/huggingface/transformers/blob/6667b0d7bf0a135b9a85d2d21b7eb25ec8ad58cb/src/transformers/trainer.py#L1039
    optimizer_grouped_parameters = [
        {
            # n not in no_decay, n in model_names
            "params": [p for n, p in base_model_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": hparams.lr_t5,
        },
        {
            # n in no_decay, n not in model_names
            "params": [p for n, p in base_model_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": hparams.lr_t5,
        },
        {
            # n not in no_decay, n not in model_names
            "params": [p for n, p in task_params if not any(nd in n for nd in no_decay)],
            "weight_decay": task_weight_decay,
            "lr": hparams.lr_iter,
        },
        {
            "params": [p for n, p in task_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": hparams.lr_iter,
        }
    ]
    return optimizer_grouped_parameters
