#!/usr/bin/env bash
export MID_DATA_DIR="./data/mid_data"
export RAW_DATA_DIR="./data/raw_data"
export OUTPUT_DIR="./out"

# adjusting args because of single GPU
export GPU_IDS="0"
export LOCAL_RANK="0"
export BERT_TYPE="finbert"  # finbert/bert_base/bert_wwm

if [ "$BERT_TYPE"x = "bert_wwm"x ];then
  export BERT_DIR="./bert_wwm/"
else if [ "$BERT_TYPE"x = "bert_base"x ];then
  export BERT_DIR="./bert-base-chinese/"
else
  export BERT_DIR="./baseline/FinBERT/"
fi

python -m torch.distributed.run --nproc_per_node=1 main.py \
--local_rank=$LOCAL_RANK \
--gpu_ids=$GPU_IDS \
--output_dir=$OUTPUT_DIR \
--mid_data_dir=$MID_DATA_DIR \
--raw_data_dir=$RAW_DATA_DIR \
--bert_dir=$BERT_DIR \
--bert_type=$BERT_TYPE \
--train_epochs=10 \
--attack_train="" \
--train_batch_size=8 \
--eval_batch_size=64 \
--dropout_prob=0.1 \
--max_seq_len=512 \
--lr=2e-5 \
--other_lr=2e-3 \
--seed=123 \
--weight_decay=0.01 \
--loss_type='ls_ce' \
--eval_model \
--swa_start=3 \
--use_fp16
fi
