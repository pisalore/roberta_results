#!/usr/bin/env bash

# RTE finetuning. Hyperparameters are taken from RoBERTa paper.
# Change the model (base or large or large.mnli) suiting your needs, in ROBERTA_PATH and roberta_arch
# ROBERTA_PATH is the path to roberta pretrained model.
# SAVE_DIR is where to save the finetuned models.
# Check https://github.com/pytorch/fairseq/blob/e3c4282551e819853952284681e9ed60398c5c4a/examples/roberta/README.glue.md

TOTAL_NUM_UPDATES="2036"  
WARMUP_UPDATES="122"     
LR="2e-05"           
NUM_CLASSES="2"
MAX_SENTENCES="16"   
ROBERTA_PATH="/home/lpisaneschi/roberta_results/roberta.large.mnli/model.pt"
SAVE_DIR="checkpoints-RTE/"    


CUDA_VISIBLE_DEVICES=0 TASK_QUEUE_ENABLE=0 fairseq-train RTE-bin/ \
    --save-dir $SAVE_DIR \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric