import os
from pathlib import Path


def substitute_files(dir, temp, filename):
    for split in range(10):
        substitute = temp.replace("%split%", str(split))
        with open(dir / f"{filename}_split{split}.json", "w") as f:
            f.write(substitute)


directory = Path.cwd() / "cfg"
template = directory / "ade.template.json"
with open(template) as f:
    template = f.read()

substitute_files(directory, template, "ade")
if (directory / "ade").exists():
    for file in os.listdir(directory / "ade"):
        if file.endswith(".template.json"):
            with (directory / "ade" / file).open() as f:
                template = f.read()
            substitute_files(directory / "ade", template, file[:-len(".template.json")])
