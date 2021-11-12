# RoBERTa results reproduction
Machine Learning exam at University of Study of Florence, Italy.

RoBERTa paper: https://arxiv.org/pdf/1907.11692.pdf \
fairseq repository on GitHUb: https://github.com/pytorch/fairseq \
GLUE Benchmark: https://gluebenchmark.com \
SuperGLUE Benchmark (for WSC): https://super.gluebenchmark.com/
RACE Benchmark: http://www.qizhexie.com/data/RACE_leaderboard.html

## Introduction
Since NLP tasks are gaining more and more importance in the AI universe nowdays, I decided to focus on bidirectional transformers for my Machine Learning exam, in particular on Google AI [BERT](https://arxiv.org/pdf/1810.04805.pdf) and one of its optimizations, which is Facebook AI [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf). Transformers are deep learning models which use the **attention mechanism** in order to understand sentences meaning giving to each part of them a different significance weight. Humans do not read only in one direction, but they take care also of already read words to completetly understand a sentence: bidirectional transformers try to do exactly the same.

RoBERTa reached some state of the art results in 2019 when was published. [fairseq](https://github.com/pytorch/fairseq) repository contains everything to reproduce such results, pre-trained models and a good documentation; however, some work could be trivial. In this repository I would like to gather my experience and everything I found useful to reproduce:
- [GLUE Benchmark results](https://gluebenchmark.com/submission/JuLiHrAkS9VSQRh1W6TJ9V9SOu23/-Lk5ZrckAabWVeQBoxrA)
- [SuperGLUE Benchmark results (WSC)](https://gluebenchmark.com/submission/JuLiHrAkS9VSQRh1W6TJ9V9SOu23/-Lk5ZrckAabWVeQBoxrA)
- [RACE Benchmark results](http://www.cs.cmu.edu/~glai1/data/race/)

I share my scripts and step by step guide to put everyone in the conditions to easily reproduce and understand the above mentioned results.

## Getting started
- We reccomend the use of `Anaconda`.
- Linux was used a OS. 
- `CUDA 11` was used. Experiments were conducted on a `NVIDIA GeForce RTX 3090` GPU, 24GB. Check your CUDA version `nvcc --version`
- [PyTorch](https://pytorch.org/) version >= 1.5.0 is required. Visit pytorch site to know which version better suites you needs.
- Python version >= 3.6 is required.

**Please, follow this doc for the first standard environment installation**, then check the experiments directories you are interested in.

1. **Install** [Anaconda](https://www.anaconda.com/products/individual) **and create a new conda environment:**
    ```shell
    conda env create -f environment.yml
    conda deactivate && conda activate roberta_results
    ```

2. **Git clone this repo:**
    ```shell
    git clone https://github.com/pisalore/roberta_results
    ```  

4. **Git clone fairseq repo:**
    ```shell
    cd roberta_results
    https://github.com/pytorch/fairseq
    cd fairseq
    pip install --editable ./
    ```

5. **Download the models** wherever you want (better in `roberta_results`). RoBERTa uses `roberta.base`, `roberta.large` and `roberta.large.mnli` models to be finutuned and they are used differently for each task. **Download and extract them following these links:**
    - [roberta.base](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)
    - [roberta.large](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz)
    - [roberta.large.mnli](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz)

    For example:
    ```shell
    wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz
    tar -xvf roberta.large.mnli.tar.gz
    rm roberta.large.mnli.tar.gz
    ```
    Now we have the models which can be used with `fairseq-cli` for finetuning and evaluation. 
