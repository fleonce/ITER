from collections import defaultdict
from json import load, dump
from pathlib import Path

from tqdm import tqdm

from with_argparse import with_argparse


@with_argparse
def reformat_dataset(
    path: Path,
    files: list[str],
    tokenizer: str = "t5-small",
    use_fast: bool = True,
):
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=use_fast)
    for file in files:
        with open(path / (file + ".json")) as f:
            json_blob = load(f)
        items = []
        for item in (tq := tqdm(json_blob)):
            text = item['tokens']
            assert isinstance(text, str)
            split_text_at = defaultdict(list)
            for entity in item['entities']:
                split_text_at[entity['start']].append(entity)
                split_text_at[entity['end']].append(entity)
            split_text_at_keys = set(split_text_at.keys())
            split_text_at_keys = list(split_text_at_keys)
            split_text_at_keys.sort()
            prev_split_at = None
            words = []
            for split_at_key in split_text_at_keys:
                if prev_split_at is not None and prev_split_at == split_at_key:
                    continue
                split_text = text[prev_split_at:split_at_key] if prev_split_at is not None else text[:split_at_key]
                if not split_text:
                    continue
                words.append(split_text.strip())
                prev_split_at = split_at_key
                pass
            if prev_split_at is not None and prev_split_at < len(text):
                words.append(text[prev_split_at:].strip())
            elif prev_split_at is None:
                words.append(text.strip())

            if 0 not in split_text_at_keys:
                split_text_at_keys.insert(0, 0)
            word_entities = []
            for entity in item['entities']:
                start_word = split_text_at_keys.index(entity['start'])
                end_word = split_text_at_keys.index(entity['end'])
                word_entities.append(
                    {'start': start_word, 'end': end_word, 'type': entity['type'], 'word': entity['word']}
                )
                entity_word = entity['word']
                span_text = "".join(words[start_word:end_word])
                # if entity_word != span_text:
                #     tq.write(f"Found non-occurring word in input: {entity_word} is not on positions "
                #              f"{entity['start']} - {entity['end']} in text '{text}'")
                #     continue
                # assert span_text == entity['word']
                if span_text == " ":
                    print(split_text_at_keys)
                    print(start_word, end_word, entity['word'])
                    print(words[:start_word])
                    print(words[start_word:end_word])
                    print(words[end_word:])
                    print(entity['start'], entity['end'])
                    print(text[:entity['start']])
                    print(text[entity['start']:entity['end']])
                    print(text[entity['end']:])
                    print("a", start_word, end_word, words[start_word-1:], text[entity['start']-10:])
                    exit(1)
            if not words:
                print("a")
                continue
            items.append({
                'tokens': words,
                'entities': word_entities,
                'relations': []
            }
            )
        with open(path / ("words_" + file + ".json"), "w") as f:
            dump(items, f)
        pass


reformat_dataset()
