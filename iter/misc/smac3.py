from typing import MutableMapping

from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import AbstractCondition
from ConfigSpace.hyperparameters import Hyperparameter
from pyrootutils import find_root, set_root


def configspace_init(
        seed: int,
        name: str,
        hyperparameters: MutableMapping[str, Hyperparameter],
        conditions: list[AbstractCondition, ...],
):
    cs = ConfigurationSpace(name, seed=seed)
    cs.add_hyperparameters(
        hyperparameters.values()
    )
    cs.add_conditions(
        conditions
    )
    return cs


def workspace_init(func):
    def inner(*args, **kwargs):
        git_root = find_root(search_from=__file__, indicator=[".git"])
        set_root(git_root, project_root_env_var=False, dotenv=False, pythonpath=True, cwd=True)

        func(*args, **kwargs)
    return inner
