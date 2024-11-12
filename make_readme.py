import json
import sys
from dataclasses import dataclass
from pathlib import Path
from with_argparse import with_argparse

from iter import ITERConfig


def parse_torch_version_cuda_bfloat(line: str):
    args = line.split(" ")
    torch_version_name = args[8]
    torch_cuda_version = args[10].replace("(", "").replace(")", "")
    if "+" in torch_version_name:
        torch_version_name = torch_version_name.split("+")[0]
    torch_bfloat = args[14]
    if torch_bfloat == "True":
        torch_precision = "torch.bfloat16"
    elif torch_bfloat == "False":
        torch_precision = "torch.float32"
    else:
        raise ValueError(args, torch_bfloat)

    return Torch(torch_version_name, torch_cuda_version, torch_precision)


def parse_seed(line: str) -> str:
    args = line.split(" ")
    return args[7]


def parse_gpu(line: str) -> str:
    args = line.split(" ")
    hostname = args[10]
    if hostname == "cas06100315p0r-gupixe":
        return "NVIDIA H100 SXM 80 GB GPU"
    raise NotImplementedError(hostname)


def parse_config(line: str) -> str:
    raise NotImplementedError


def remove_trailing_linefeed(line: str) -> str:
    if line.endswith("\n"):
        return line[:-len("\n")]
    return line


def find_experiment_dir(path: Path):
    while not (path / "run_experiment.log").exists():
        if path.parent is None or path.parent == path:
            raise ValueError(path)
        path = path.parent
    return path


def find_experiment_config(lines: list[str]) -> str:
    datasets = set()
    for line in lines:
        if "Dataset = " in line:
            args = line.split(" ")
            datasets.add(args[7])

    datasets = list(datasets)
    if len(datasets) > 1:
        print("Found multiple datasets in run_experiment, please input manually")
        for ds in datasets:
            print(" - " + ds)

        while True:
            inp = input(">>> Config =")
            if inp in datasets:
                return inp
            else:
                print("Dataset " + inp + " not in the list ...")
    return datasets[0]


def parse_model_class(lines: list[str]):
    for line in lines:
        if "NER only" in line:
            return "ITER"
    return "ITERForRelationExtraction"


def parse_f1_score(line: str) -> str:
    args = line.split(" ")
    index = args.index("ERE")
    f1 = float(args[index + 4].split("=")[1]) * 100
    return f"{f1:.3f}"


@dataclass
class Torch:
    version: str
    cuda: str
    precision: str


def find_model_name(experiment_dir: Path, path: Path) -> str:
    model_name = path.parent.relative_to(experiment_dir)
    return model_name.as_posix()


def translate_model_name(model_name: str) -> str:
    if model_name.startswith("google/"):
        model_name = model_name[len("google/"):]
    if "flan" in model_name:
        args = model_name.split("-")
        return "flant5-" + args[2]
    elif "t5" in model_name:
        return model_name
    raise NotImplementedError(model_name)


@with_argparse
def make_readme(path: Path, base_readme: Path = Path("MODEL.md"), datasets: Path = None):
    experiment_dir = find_experiment_dir(path)
    experiment_log_file = experiment_dir / "run_experiment.log"
    if not experiment_log_file.exists():
        raise ValueError(f"Cannot find {experiment_log_file.absolute().as_posix()}")
    log_file = path / "train.log"

    base_model_name = find_model_name(experiment_dir, path)
    model_name = translate_model_name(base_model_name)

    with base_readme.open() as f_readme:
        readme_text = f_readme.read()

    with experiment_log_file.open() as f_experiment:
        experiment_lines = f_experiment.readlines()
    experiment_lines = list(map(remove_trailing_linefeed, experiment_lines))

    with log_file.open() as f_log:
        log_lines = f_log.readlines()
    log_lines = list(map(remove_trailing_linefeed, log_lines))

    torch = parse_torch_version_cuda_bfloat(log_lines[0])
    config = find_experiment_config(experiment_lines)
    dataset = config
    if "/" in dataset:
        dataset = dataset.split("/")[0]
    gpu = parse_gpu(log_lines[1])
    seed = parse_seed(log_lines[2])
    model_class = parse_model_class(log_lines)
    f1 = parse_f1_score(log_lines[-2])

    model_config = ITERConfig.from_pretrained(path)
    if not model_config.entity_types:
        print(f"Attempting to fix entity types for {path.as_posix()} ...", file=sys.stderr)
        if datasets is None:
            raise ValueError("datasets is None")
        types_file = datasets / dataset / (dataset + "_types.json")
        with types_file.open() as f:
            types = json.load(f)
        model_config.entity_types = list(types["entities"].keys())
        if not model_config.link_types:
            model_config.link_types = list(types["relations"].keys())
        model_config.save_pretrained(path)
        print(f"Please check manually ...", file=sys.stderr)

    info = {
        "f1": f1,
        "gpu": gpu,
        "seed": seed,
        "config": config,
        "torch": torch,
        "dataset": dataset,
        "citation": "{citation}",
        "model_class": model_class,
        "model_name": model_name,
        "base_model_name": base_model_name,
        "precision_command": "" if torch.precision != "torch.bfloat16" else " --use_bfloat16"
    }
    print(readme_text.format(**info))


make_readme()
