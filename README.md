# Lightweight Cross-lingual Sentence Representation Learning
Read this before implementing the lightweight crosslingual sentence embedding model training.

## Prerequisites:
- Pytorch 1.7.1
- SentencePiece

## Dataset and Preprocessing:
We uploaded the cleaned and preprocessed dataset of English-French with the size of 300,000 parallel sentences due to file size limitation.
Find the datasets in the submitted "data.zip" file.
Please set up the specific training data path in "config.py" before implementing the model training.

## Usage:
```
bash start.sh 0 UGT ckpt 0 fr
```
- Arg1: GPU card number
- Arg2: Training tasks: use "UGT" or "UGT+ALIGN+SIM"
- Arg3: path for saving checkpoints and log files
- Arg4: "0" denotes training from scratch; "True" denotes resuming training from the recent checkpoint
- Arg5: another language (default french)

## Others
We will add more details soon.
