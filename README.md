# Lightweight Cross-lingual Sentence Representation Learning
Code for [paper](https://aclanthology.org/2021.acl-long.226.pdf)

## Prerequisites:
```
conda env create --file myenv.yaml
```

## Dataset and Preprocessing:
We uploaded the cleaned and preprocessed dataset of English-French with the size of 300,000 parallel sentences as examples.
Find the training data samples in "data.zip".
For the whole ParaCrawl dataset, please download them from [ParaCrawl](https://opus.nlpl.eu/ParaCrawl-v5.php).

## Usage:
```
bash start.sh 0 UGT ckpt false fr
```
- $1: GPU card number
- $2: Training tasks: use "UGT" or "UGT+ALIGN+SIM"
- $3: path for saving checkpoints and log files
- $4: "false" denotes training from scratch; "true" denotes resuming training from checkpoint, which can be set up in "config.py"
- $5: another language (default: french)

To monitor the training, use
```
tail -f ckpt/UGT.out
```

## Others
We will add more details soon.

## Reference
[1] Zhuoyuan Mao, Prakhar Gupta, Chenhui Chu, Martin Jaggi, Sadao Kurohashi. 2021. [*Lightweight Cross-Lingual Sentence Representation Learning*](https://aclanthology.org/2021.acl-long.226/). ACL 2021.

```
@inproceedings{mao-etal-2021-lightweight,
    title = "Lightweight Cross-Lingual Sentence Representation Learning",
    author = "Mao, Zhuoyuan  and
      Gupta, Prakhar  and
      Chu, Chenhui  and
      Jaggi, Martin  and
      Kurohashi, Sadao",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.226",
    doi = "10.18653/v1/2021.acl-long.226",
    pages = "2902--2913",
    abstract = "Large-scale models for learning fixed-dimensional cross-lingual sentence representations like Large-scale models for learning fixed-dimensional cross-lingual sentence representations like LASER (Artetxe and Schwenk, 2019b) lead to significant improvement in performance on downstream tasks. However, further increases and modifications based on such large-scale models are usually impractical due to memory limitations. In this work, we introduce a lightweight dual-transformer architecture with just 2 layers for generating memory-efficient cross-lingual sentence representations. We explore different training tasks and observe that current cross-lingual training tasks leave a lot to be desired for this shallow architecture. To ameliorate this, we propose a novel cross-lingual language model, which combines the existing single-word masked language model with the newly proposed cross-lingual token-level reconstruction task. We further augment the training task by the introduction of two computationally-lite sentence-level contrastive learning tasks to enhance the alignment of cross-lingual sentence representation space, which compensates for the learning bottleneck of the lightweight transformer for generative tasks. Our comparisons with competing models on cross-lingual sentence retrieval and multilingual document classification confirm the effectiveness of the newly proposed training tasks for a shallow model.",
}
```
