max_history_utterance=7
context_seq_len=30
persona_seq_len=20
tgt_len=30
lr=1e-4

TRAINDATA=./processed_data/personachat/train.cache
VALIDDATA=./processed_data/personachat/valid.cache

MODEL_DIR=./checkpoints/personachat/$lr
LOG_DIR=./logs/personachat/$lr
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

log_path=train_logs/personachat_train_$lr.log
echo $MODEL_DIR >> $log_path

CUDA_VISIBLE_DEVICES=0 python train.py \
--train_dataset $TRAINDATA --valid_dataset $VALIDDATA \
--extra_info persona \
--save_model_dir $MODEL_DIR \
--log_dir $LOG_DIR \
--max_persona_seq_len $persona_seq_len \
--max_context_seq_len $context_seq_len --max_tgt_len $tgt_len \
--max_history_utterance $max_history_utterance \
--n_epochs 6 \
--batch_size 32 --gradient_accumulate_steps 2 \
--lr $lr >> $log_path