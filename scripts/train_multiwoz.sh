max_history_utterance=7
context_seq_len=60
tgt_len=60
lr=3e-4

TRAINDATA=./processed_data/multiwoz/train.cache
VALIDDATA=./processed_data/multiwoz/valid.cache

MODEL_DIR=./checkpoints/multiwoz/$lr
LOG_DIR=./logs/multiwoz/$lr
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

log_path=train_logs/multiwoz_train_$lr.log
echo $MODEL_DIR >> $log_path

CUDA_VISIBLE_DEVICES=0 python train.py \
--train_dataset $TRAINDATA --valid_dataset $VALIDDATA \
--extra_info none \
--save_model_dir $MODEL_DIR \
--log_dir $LOG_DIR \
--max_context_seq_len $context_seq_len --max_tgt_len $tgt_len \
--max_history_utterance $max_history_utterance \
--n_epochs 12 \
--batch_size 32 --gradient_accumulate_steps 2 \
--lr $lr >> $log_path