#!/usr/bin/env bash

# Evaluate your finetuned model on RACE dataset.

# data directory used during training
DATA_DIR="/home/lpisaneschi/ml/fairseq/RACE-BIN-preprocessed"   
# path to the finetuned model checkpoint         
MODEL_PATH="/home/lpisaneschi/ml/fairseq/checkpoints/checkpoint_best_race.pt"  
# can be test (Middle) or test1 (High)                   
TEST_SPLIT="test"                         
fairseq-validate \
    $DATA_DIR \
    --valid-subset $TEST_SPLIT \
    --path $MODEL_PATH \
    --max-sentences 1 \
    --task sentence_ranking \
    --criterion sentence_ranking \