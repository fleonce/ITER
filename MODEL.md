---
license: apache-2.0
base_model:
- {base_model_name}
library_name: transformers
tags:
- relation extraction
- nlp
model-index:
  - name: iter-{dataset}-{model_name}
    results:
      - task:
          type: relation-extraction
        dataset:
          name: {dataset}
          type: {dataset}
        metrics:
          - name: F1
            type: f1
            value: {f1}
---


# ITER: Iterative Transformer-based Entity Recognition and Relation Extraction

This model checkpoint is part of the collection of models published alongside our paper ITER, 
[accepted at EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.655/).<br>
To ease reproducibility and enable open research, our source code has been published on [GitHub](https://github.com/fleonce/iter).

This model achieved an F1 score of `{f1}` on dataset `{dataset}`

### Using ITER in your code

First, install ITER in your preferred environment:

```text
pip install git+https://github.com/fleonce/iter
```

To use our model, refer to the following code:
```python
from iter import {model_class}

model = {model_class}.from_pretrained("fleonce/iter-{dataset}-{model_name}")
tokenizer = model.tokenizer

encodings = tokenizer(
  "An art exhibit at the Hakawati Theatre in Arab east Jerusalem was a series of portraits of Palestinians killed in the rebellion .",
  return_tensors="pt"
)

generation_output = model.generate(
    encodings["input_ids"],
    attention_mask=encodings["attention_mask"],
)

# entities
print(generation_output.entities)

# relations between entities
print(generation_output.links)
```

### Checkpoints

We publish checkpoints for the models performing best on the following datasets:

- **ACE05**:
  1. [fleonce/iter-ace05-flant5-large](https://huggingface.co/fleonce/iter-ace05-flant5-large)
  2. [fleonce/iter-ace05-flant5-xl](https://huggingface.co/fleonce/iter-ace05-flant5-xl)
  3. [fleonce/iter-ace05-deberta-large](https://huggingface.co/fleonce/iter-ace05-deberta-large)
- **CoNLL04**:
  1. [fleonce/iter-conll04-flant5-large](https://huggingface.co/fleonce/iter-conll04-flant5-large)
  2. [fleonce/iter-conll04-flant5-xl](https://huggingface.co/fleonce/iter-conll04-flant5-xl)
  3. [fleonce/iter-conll04-deberta-large](https://huggingface.co/fleonce/iter-conll04-deberta-large)
- **ADE**:
  1. [fleonce/iter-ade-flant5-large](https://huggingface.co/fleonce/iter-ade-flant5-large)
  2. [fleonce/iter-ade-flant5-xl](https://huggingface.co/fleonce/iter-ade-flant5-xl)
  3. [fleonce/iter-ade-deberta-large](https://huggingface.co/fleonce/iter-ade-deberta-large)
- **SciERC**:
  1. [fleonce/iter-scierc-flant5-large](https://huggingface.co/fleonce/iter-scierc-flant5-large)
  2. [fleonce/iter-scierc-flant5-xl](https://huggingface.co/fleonce/iter-scierc-flant5-xl)
  3. [fleonce/iter-scierc-deberta-large](https://huggingface.co/fleonce/iter-scierc-deberta-large)
  4. [fleonce/iter-scierc-scideberta-full](https://huggingface.co/fleonce/iter-scierc-scideberta-full)
- **CoNLL03**:
  1. [fleonce/iter-conll03-flant5-large](https://huggingface.co/fleonce/iter-conll03-flant5-large)
  2. [fleonce/iter-conll03-flant5-xl](https://huggingface.co/fleonce/iter-conll03-flant5-xl)
  3. [fleonce/iter-conll03-deberta-large](https://huggingface.co/fleonce/iter-conll03-deberta-large)
- **GENIA**:
  1. [fleonce/iter-genia-flant5-large](https://huggingface.co/fleonce/iter-genia-flant5-large)
  2. [fleonce/iter-genia-flant5-xl](https://huggingface.co/fleonce/iter-genia-flant5-xl)
  3. [fleonce/iter-genia-deberta-large](https://huggingface.co/fleonce/iter-genia-deberta-large)


### Reproducibility

For each dataset, we selected the best performing checkpoint out of the 5 training runs we performed during training.
This model was trained with the following hyperparameters:

- Seed: `{seed}`
- Config: `{config}`
- PyTorch `{torch.version}` with CUDA `{torch.cuda}` and precision `{torch.precision}`
- GPU: `1 {gpu}`

Varying GPU and CUDA version as well as training precision did result in slightly different end results in our tests
for reproducibility.

To train this model, refer to the following command:
```shell
python3 train.py --dataset {config} --transformer {base_model_name}{precision_command} --seed {seed}
```

```text
@inproceedings{citation}
```
