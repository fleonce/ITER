import abc
import dataclasses
import json
from pathlib import Path
from typing import TypeVar

from iter.misc import merge_update


T = TypeVar('T')


class NameableConfig(abc.ABC):
    section_name: str = None
    with_filepath: bool = False

    @classmethod
    def from_name(cls: type[T], name: str, /, setup=False, **init_kwargs) -> T:
        config = cls.load_from_name(name)
        kwargs = config[cls.section_name]
        if cls.with_filepath:
            kwargs["file_path"] = name

        kwargs = merge_update(kwargs, init_kwargs)
        conf = cls(**kwargs)
        if (
            setup
            and hasattr(conf, "setup_dataset")
        ):
            conf.setup_dataset()
        return conf

    @classmethod
    def load_from_name(cls, name: str):
        assert cls.section_name is not None
        config_path = cls.find_config(name)
        with config_path.open() as f:
            config = json.load(f)
        if "base" in config:
            baseconfig = cls.load_from_name(config["base"])
            config = merge_update(baseconfig, config)
        return config

    @classmethod
    def find_config(cls, name: str) -> Path:
        if not name.endswith(".json"):
            name = name + ".json"
        path = Path(name)
        local_path = Path().cwd() / "cfg" / name
        if path.exists():
            return path
        elif local_path.exists():
            return local_path
        raise ValueError(
            f"Cannot find config '{name}' as a path or as a local config ({local_path.as_posix()})"
        )


@dataclasses.dataclass
class Hparams(NameableConfig):
    section_name = "training"
    with_filepath = False

    patience: int
    num_epochs: int
    max_epochs: int
    lr_t5: float
    lr_iter: float
    lr_scheduler: str
    weight_decay: float
    warmup_steps: float | int
    task_weight_decay: float
    task_warmup_steps: float | int
    task_lr_scheduler: str
    activation_fn: str = "gelu"
    dropout: float = 0.3
    d_ff: int = 0

    batch_size: int = 8
    gradient_accumulation: int = 1
    metric_average: str = "micro"
    optimize_for: str = "ere"
    eval_batch_size: int = 0

    def to_json(self):
        return json.dumps(self.__dict__, indent=2)

    def get_batch_size(self, train: bool):
        if train:
            return self.batch_size
        return self.eval_batch_size or self.batch_size
