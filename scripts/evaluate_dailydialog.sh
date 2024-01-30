max_history_utterance=7
context_seq_len=60
tgt_len=0
predict_len=60
lr=3e-4
ckpt=7

TESTDATA=./processed_data/dailydialog/test.cache
CHECKPOINT_PATH=./checkpoints/dailydialog/$lr/checkpoint${ckpt}.pt
SAVE_RESULT_PATH=./results/dailydialog_lgcm_lr${lr}ckpt${ckpt}.txt

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--test_dataset $TESTDATA \
--extra_info none \
--checkpoint_path $CHECKPOINT_PATH \
--save_result_path $SAVE_RESULT_PATH \
--max_context_seq_len $context_seq_len --max_tgt_len $tgt_len \
--max_predict_len $predict_len \
--max_history_utterance $max_history_utterance