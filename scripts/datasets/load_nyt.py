import json
from json import load

with open('datasets/nyt/train.json') as f:
    json_blob = load(f)

entity_types = set()
link_types = set()
for item in json_blob:
    entity_types.update({tp for detail in item['spo_details'] for tp in {detail[2], detail[6]}})
    link_types.update({detail[3] for detail in item['spo_details']})

with open('datasets/nyt/nyt_types.json', 'w') as f:
    json.dump({
        "entities": {tp: {"short": tp, "verbose": tp} for tp in entity_types},
        "relations": {tp: {"short": tp, "verbose": tp, "symmetric": False} for tp in link_types}
    }, f)

entity_types = list(entity_types)
link_types = list(link_types)
print(len(entity_types), len(link_types))

files = ['train', 'dev', 'test']
for file in files:
    def spo_to_dict(spo):
        return {'start': spo[0], 'end': spo[1], 'type': spo[2]}
    with open(f'datasets/nyt/{file}.json') as f:
        json_blob = json.load(f)
    output = []
    for item in json_blob:
        entities = {tp for detail in item['spo_details'] for tp in {
            tuple(detail[:3]), tuple(detail[4:])
        }}
        entities = [spo_to_dict(tp) for tp in entities]
        relations = {(entities.index(spo_to_dict(detail[0:])), detail[3], entities.index(spo_to_dict(detail[4:]))) for detail in item['spo_details']}
        relations = list(relations)
        relations = [{"head": detail[0], "type": detail[1], "tail": detail[2]} for detail in relations]
        output.append({
            "tokens": item["tokens"],
            "entities": entities,
            "relations": relations
        })
    with open(f'datasets/nyt/nyt_{file}.json', 'w') as f:
        json.dump(output, f)
