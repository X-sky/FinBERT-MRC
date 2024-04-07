#!/usr/bin/env bash
export MID_DATA_DIR="./data/mid_data"
export RAW_DATA_DIR="./data/raw_data"
export OUTPUT_DIR="./out"

# adjusting args because of single GPU
export GPU_IDS="0"
export LOCAL_RANK="0"
export BERT_TYPE="roberta"  # default fin_bert pre-trained model

if [ "$BERT_TYPE"x = "bert"x ];then
  export BERT_DIR="./baselines/chinese_bert_wwm_ext/"
else if [ "$BERT_TYPE"x = "roberta"x ];then
  export BERT_DIR="./baselines/chinese_roberta_wwm_ext/"
else
  export BERT_DIR="./baselines/fin_bert/"
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
