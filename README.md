# Lightweight Cross-lingual Sentence Representation Learning
Read this before implementing the lightweight crosslingual sentence embedding model training.
We include the training code for your checking of the lightweight model training.


## Prerequisites:
- Pytorch 1.7.1
- SentencePiece

## Dataset and Preprocessing:
We uploaded the cleaned and preprocessed dataset of English-French with the size of 300,000 parallel sentences due to file size limitation.
Find the datasets in the submitted "data.zip" file.
(!!important!!) Then set the specific training data path in "config.py"

## Usage:
```
bash start.sh 0 UGT ckpt 0 fr
```
- Arg1: GPU card number
- Arg2: Training tasks: use "UGT" or "UGT+ALIGN+SIM"
- Arg3: path for saving checkpoints and log files
- Arg4: "0" denotes training from scratch; "True" denotes resuming training from the recent checkpoint
- Arg5: another language (default french)


## Note that:
Due to the data upload limitation, we only support the training of "UGT" and "UGT+ALIGN+SIM" for English-French
We will update the support for other language pairs on Github release.

Specifically,
```
bash start.sh 0 UGT ckpt 0 fr
```
or
```
bash start.sh 0 UGT+ALIGN+SIM ckpt 0 fr
```
can be implemented for now.

