from pathlib import Path

import torch

from iter import ITER, ITERConfig
from with_argparse import with_argparse


@with_argparse
def convert(
        path: Path,
        transformer_name: str,
        model_name: str,
        activation_fn: str,
        dataset: str,
        dropout: float,
        features: int,
        max_length: int = 512,
        max_nest_depth: int = 1,
):
    # guess model name
    model_path = f"models/{dataset}-{model_name}"
    state_dict = torch.load(path, weights_only=True, map_location="cpu")
    model_state_dict = state_dict["model"]

    # guess num types and links
    num_types = num_links = ...
    if "lr_score.wo.weight" in model_state_dict:
        num_types = model_state_dict["lr_score.wo.weight"].size(0)
    if "rr_score.wo.weight" in model_state_dict:
        num_links = model_state_dict["rr_score.wo.weight"].size(0)
    assert num_types != Ellipsis and num_links != Ellipsis, (num_types, num_links)

    config = ITERConfig(
        transformer_name=transformer_name,
        transformer_config={
            "max_length": max_length,
        },
        num_links=num_links,
        num_types=num_types,
        features=features,
        dataset=dataset,
        max_nest_depth=max_nest_depth,
        activation_fn=activation_fn,
        dropout=dropout,
    )
    config.save_pretrained(model_path)

    model = ITER(config)
    model.load_state_dict(model_state_dict)

    model.save_pretrained(model_path)
    print(f"Saved model checkpoint to '{model_path}'")


if __name__ == "__main__":
    convert()
