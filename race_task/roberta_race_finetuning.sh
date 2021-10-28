#!/usr/bin/env bash

# RACE finetuning. Hyperparameters are taken from RoBERTa paper.
# Use roberta.large
# ROBERTA_PATH is the path to roberta pretrained model.
# SAVE_DIR is where to save the finetuned models.
# Check https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.race.md

MAX_EPOCH="5"           
LR="1e-05"
NUM_CLASSES="4"
MAX_SENTENCES="1" 
UPDATE_FREQ="16" # Accumulate gradients to simulate training on 16 GPUs. See https://github.com/pytorch/fairseq/issues/1946.
DATA_DIR="/home/lpisaneschi/roberta_results/race_task/RACE-bin"
ROBERTA_PATH="/home/lpisaneschi/roberta_results/roberta.large/model.pt" 

CUDA_VISIBLE_DEVICES=0,1 fairseq-train $DATA_DIR --ddp-backend=no_c10d \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task sentence_ranking \
    --num-classes $NUM_CLASSES \
    --init-token 0 --separator-token 2 \
    --max-option-length 128 \
    --max-positions 512 \
    --shorten-method "truncate" \
    --arch roberta_large \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_ranking \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler fixed --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --max-epoch $MAX_EPOCH