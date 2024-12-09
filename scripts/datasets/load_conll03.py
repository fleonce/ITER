import json
from pathlib import Path

from datasets import load_dataset, Dataset


def create_dataset(dataset: Dataset, output_filename=None):
#    mapping = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    mapping = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    outputs = []
    for elem in dataset:
        elem_tokens = elem['tokens']
        elem_tags = [mapping[tag] for tag in elem['ner_tags']]

        current_entity = []
        current_category = None
        start_index = 0
        # Iterate through the tokens and numerical NE tags
        entities = []
        begins = 0
        for index, (token, tag) in enumerate(zip(elem_tokens, elem_tags)):
            if tag == 'O':
                if current_entity:
                    assert current_category is not None
                    assert current_entity
                    entities.append({
                        'type': current_category,
                        'start': start_index,
                        'end': index,
                    })
                    current_category = None
                current_entity = []
            else:
                tag_category = tag.split('-')[1]
                tag_type = tag.split('-')[0]
                if tag_type == 'B':
                    begins += 1
                if (tag_category != current_category and current_category is not None and current_entity) or (tag_type == 'B' and current_entity):
                    assert current_entity
                    entities.append({
                        'type': current_category,
                        'start': start_index,
                        'end': index,
                    })
                    current_entity = []
                if not current_entity:
                    start_index = index
                current_entity.append(token)
                current_category = tag_category

        # check for leftover entities
        if current_entity:
            entities.append({
                'type': current_category,
                'start': start_index,
                'end': len(elem_tokens),
            })
        assert len(entities) == begins
        outputs.append({
            'tokens': elem_tokens,
            'entities': entities
        })


    output_file = Path(output_filename)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(outputs, f, indent=2)


if __name__ == '__main__':
    dataset_dict = load_dataset('conll2003')
#    tk = (AutoTokenizer.from_pretrained("t5-small"))
    create_dataset(dataset_dict["train"], output_filename="datasets/conll03/conll03_strain.json")
    create_dataset(dataset_dict["validation"], output_filename="datasets/conll03/conll03_sval.json")
    create_dataset(dataset_dict["test"], output_filename="datasets/conll03/conll03_stest.json")
