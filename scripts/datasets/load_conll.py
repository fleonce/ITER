import json
import re
import typing
from pathlib import Path

from tqdm import tqdm
from with_argparse import with_argparse


@with_argparse
def create_dataset(
    tabs: list[int],
    file: Path,
    output_file: Path = None,
    sep: typing.Literal["space", "tab"] = "space",
    strip: bool = True,
    total_tabs: int = None,
):
    sep = " " if sep == "space" else "\t"
    if output_file is None:
        output_file = file.with_suffix(".json")

    with (
        file.open() as f,
        tqdm(desc=file.name) as tq
    ):
        documents = []
        words = []
        entities = []
        active_tag: dict[int, str | None] = {tab: None for tab in tabs}
        active_since: dict[int, int] = {tab: -1 for tab in tabs}

        def end_active_tag(tab: int, include_last:bool = False):
            nonlocal active_tag, active_since, entities
            if active_since[tab] < 0:
                return
            assert active_since[tab] >= 0
            offset = -1 if not include_last else -0
            entities.append(
                {
                    "start": active_since[tab],
                    "end": len(words) + offset,
                    "type": active_tag[tab].lower(),
                    "text": words[active_since[tab]:] if include_last else words[active_since[tab]:-1]
                }
            )
            active_since[tab] = -1
            active_tag[tab] = None

        def end_active_tags(include_last: bool = False):
            for tab in tabs:
                end_active_tag(tab, include_last)

        line_no = 0
        while line := f.readline():
            line_no += 1
            if line == "\n":
                end_active_tags(include_last=True)

                documents.append({
                    "tokens": words,
                    "entities": entities,
                    "relations": []
                })
                words = []
                entities = []
                pass
            else:
                lf = '\n'
                line = line.replace(lf, '')
                if strip:
                    line = line.strip()
                if line == 'O':
                    continue
                try:
                    tokens = line.split(sep)
                    if len(tokens) < 2:
                        raise ValueError()
                    if total_tabs is not None and len(tokens) != total_tabs:
                        raise ValueError()
                except ValueError:
                    raise ValueError(f"line {line_no} ({line.replace(lf, '')}) was split into " + '---'.join(line.split()))
                for tab in tabs:
                    tag = tokens[tab - 1]
                    word = tokens[0]
                    if len(tokens) > tab:
                        overflow_tokens = len(tokens) - tab
                        word = ' '.join(tokens[:overflow_tokens + 1])
                    words.append(word)
                    if tag == 'O':
                        if active_tag[tab] is not None:
                            end_active_tag(tab)
                        continue
                    elif tag.startswith('B'):
                        if active_tag[tab] is not None:
                            end_active_tag(tab)
                        active_tag[tab] = tag.split('-')[1]
                        active_since[tab] = len(words) - 1
                    elif tag.startswith('I'):
                        if active_tag[tab] is None:
                            tq.write(f"line {line_no}: Found {tag} when previous was not B- or I-")
                            continue
                        elif active_tag[tab] != tag.split('-')[1]:
                            tq.write(f"line {line_no}: I- tag was different from prev. tag: {tag} vs B/I-{active_tag[tab]}")
                            end_active_tag(tab)
            tq.update(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(documents, f, indent=2)


create_dataset()