# LGCM
This is the official repository for the Findings of EACL 2024 paper "Local and Global Contexts for Conversation".

## Environment
+ python==3.6.8
+ torch==1.4.0
+ transformers==3.0.2


## Usage
### Data preparation
#### Tokenizer
We use GPT2 vocabulary in our experiments. To prepare vocabulary files, please:
+ download `gpt2-vocab.json` from [here](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json), rename it to `vocab.json`, and move it to the folder `./gpt2_vocab/`
+ download `gpt2-merges.txt` from [here](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt), rename it to `merges.txt`, and move it to the folder `./gpt2_vocab/`

#### Datasets
We have trained LGCM on three public available dalogue datasets:
+ [PersonaChat](https://aclanthology.org/P18-1205/)
+ [DalyDialog](https://aclanthology.org/I17-1099/)
+ [MultiWOZ](https://aclanthology.org/D18-1547/)

After downloading raw data, please run scripts in `./prepare_data/` to preprocess data.

### Training
+ PersonaChat: `bash scripts/train_personachat.sh`
+ DailyDialog: `bash scripts/train_dailydialog.sh`
+ MultiWOZ: `bash scripts/train_multiwoz.sh`

### Evaluation
+ PersonaChat: `bash scripts/evaluate_personachat.sh`
+ DailyDialog: `bash scripts/evaluate_dailydialog.sh`
+ MultiWOZ: `bash scripts/evaluate_multiwoz.sh`