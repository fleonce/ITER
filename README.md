## ITER: Iterative Transformer-based Entity Recognition and Relation Extraction

This repository contains the source code for our paper ITER, [accepted at EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.655/).

### Setup

To set up the repository, the following basic steps are required:


#### 1. Install ITER
```bash
python3 -m venv venv && source venv/bin/activate  # optional
pip install git+https://github.com/fleonce/iter
```

#### 2. Download the required datasets
```bash
bash scripts/datasets/load_datasets.sh
```

#### 3. Run the training script

```bash
python3 train.py --transformer t5-small --dataset {ace05,ade,conll03,conll04,genia,scierc}
```
where the transformer and dataset arguments have the following possible values:

Currently working transformer models are:
```
- t5-{small,base,large,3b,11b}
- google/t5-v1_1-{small,base,large,xl,xxl}
- google/flan-t5-{small,base,large,xl,xll}
- bert-large-cased
- microsoft/deberta-v3-{xsmall,small,base,large}
- microsoft/deberta-v2-{xlarge,xxlarge}
```

Currently supported datasets are:
```
- ace05
- ade
- conll03
- conll04
- genia
- scierc
```

#### 4. Evaluating models

To evaluate the checkpoints we provided, simply use the following command:

```bash
python3 evaluate.py --model {checkpoint}
```

#### 5. Model Checkpoints on Huggingface
### Checkpoints

We publish checkpoints for the models performing best on the following datasets:

- **ACE05**:
  1. [fleonce/iter-ace05-deberta-large](https://huggingface.co/fleonce/iter-ace05-deberta-large)
- **CoNLL04**:
  1. [fleonce/iter-conll04-deberta-large](https://huggingface.co/fleonce/iter-conll04-deberta-large)
- **ADE**:
  1. [fleonce/iter-ade-deberta-large](https://huggingface.co/fleonce/iter-ade-deberta-large)
- **SciERC**:
  1. [fleonce/iter-scierc-deberta-large](https://huggingface.co/fleonce/iter-scierc-deberta-large)
  2. [fleonce/iter-scierc-scideberta-full](https://huggingface.co/fleonce/iter-scierc-scideberta-full)
- **CoNLL03**:
  1. [fleonce/iter-conll03-deberta-large](https://huggingface.co/fleonce/iter-conll03-deberta-large)
- **GENIA**:
  1. [fleonce/iter-genia-deberta-large](https://huggingface.co/fleonce/iter-genia-deberta-large)
