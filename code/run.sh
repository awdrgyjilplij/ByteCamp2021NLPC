#!/usr/bin/env bash
python bert.py \
  --gpu_ids="0,1,2,3,4,5,6,7" \
  --num_train_epochs=2 \
  --train_batch_size=16 \
  --eval_batch_size=16 \
  --a_dropout_prob=0.1 \
  --h_dropout_prob=0.1 \
  --s_dropout_prob=0.1 \
  --learning_rate=1e-5 \
  --warmup_prop=0.1 \
  --max_seq_length=512 \
