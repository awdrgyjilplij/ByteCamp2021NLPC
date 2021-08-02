#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=bert-base-chinese
export OUTPUT_DIR=$CURRENT_DIR/check_points
TASK_NAME="bert_baseline"
 
python bert.py \
  --model_type='xxlarge' \
  --gpu_ids="0,1,2,3,4,5,6,7" \
  --num_train_epochs=2 \
  --train_batch_size=16 \
  --eval_batch_size=16 \
  --a_dropout_prob=0.1 \
  --h_dropout_prob=0.1 \
  --s_dropout_prob=0.1 \
  --learning_rate=1e-5 \
  --warmup_proportion=0.1 \
  --max_seq_length=512 \
  --do_train \
  --do_eval \
  --gradient_accumulation_steps=16 \
