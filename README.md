## ITER: Iterative Transformer-based Entity Recognition and Relation Extraction

This repository contains the source code for our paper ITER, accepted at EMNLP 2024.

### Setup

To set up the repository, the following basic steps are required:

### Open Issues 
- [ ] At the moment it seems like cuda is required for everything to work. 

#### 1. Install the project dependencies
```bash
python3 -m venv venv && source venv/bin/activate  # optional
pip install -e .
```

#### 2. Download the required datasets
```bash
# requires gdown from the requirements to be installed
# pip install gdown
bash scripts/datasets/load_datasets.sh
```

#### 3. Run the training script

```bash
python3 train.py --transformer t5-small --dataset {conll04,conll03,ade,genia,scierc}
```
where the transformer and dataset arguments have the following possible values:

Currently working transformer models are:
```
- t5-{small,base,large,3b,11b}
- google/t5-v1_1-{small,base,large,xl,xxl}
- google/flan-t5-{small,base,large,xl,xll}
- bert-large-cased
- microsoft/deberta-v3-{small,base,large}
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
python3 evaluate.py --model models/conll04/{time} [--dataset {ace05,conll03,conll04,ade,genia,scierc}]
```

#### 5. Model Checkpoints on Huggingface
- ace05
- ade
- conll03
- conll04
- genia
- scierc

#### 6. Scripts for Reproducing Results
- ace05
```bash 

```
- ade
```bash 

```
- conll03
```bash 

```
- conll04
```bash 

```
- genia
```bash 

```
- scierc
```bash 

```
```bash 

```
