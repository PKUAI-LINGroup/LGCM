max_history_utterance=7
context_seq_len=60
tgt_len=60
lr=3e-4

TRAINDATA=./processed_data/dailydialog/train.cache
VALIDDATA=./processed_data/dailydialog/valid.cache

MODEL_DIR=./checkpoints/dailydialog/$lr
LOG_DIR=./logs/dailydialog/$lr
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

log_path=train_logs/dailydialog_train_$lr.log
echo $MODEL_DIR >> $log_path

CUDA_VISIBLE_DEVICES=0 python train.py \
--train_dataset $TRAINDATA --valid_dataset $VALIDDATA \
--extra_info none \
--save_model_dir $MODEL_DIR \
--log_dir $LOG_DIR \
--max_context_seq_len $context_seq_len --max_tgt_len $tgt_len \
--max_history_utterance $max_history_utterance \
--n_epochs 15 \
--batch_size 16 --gradient_accumulate_steps 4 \
--lr $lr >> $log_path