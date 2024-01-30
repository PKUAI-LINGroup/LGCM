max_history_utterance=7
context_seq_len=30
persona_seq_len=20
tgt_len=0
predict_len=60
lr=1e-4
ckpt=4

TESTDATA=./processed_data/personachat/test.cache
CHECKPOINT_PATH=./checkpoints/personachat/$lr/checkpoint${ckpt}.pt
SAVE_RESULT_PATH=./results/personachat_lgcm_lr${lr}ckpt${ckpt}.txt

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--test_dataset $TESTDATA \
--extra_info persona \
--checkpoint_path $CHECKPOINT_PATH \
--save_result_path $SAVE_RESULT_PATH \
--max_persona_seq_len $persona_seq_len \
--max_context_seq_len $context_seq_len --max_tgt_len $tgt_len \
--max_predict_len $predict_len \
--max_history_utterance $max_history_utterance