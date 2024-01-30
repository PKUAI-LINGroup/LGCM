# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : prepare_personachat.py
# @Description : script to preprocess personachat dataset


import os
import random
import logging
import torch
from tqdm import tqdm

from vocabs.gpt2_tokenizer import GPT2Vocab

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_data(data_path):
    """
    :param data_path:
    :return: List[{'your_persona': List[str],
                   'partner_persona': List[str],
                   'dialog': List[str]}]
    """
    logger.info("loading dialogs from file")
    with open(data_path, 'r', encoding='utf-8') as file:
        data = []
        for line in tqdm(file.readlines()):
            line = line.strip()

            if len(line) == 0:
                continue

            space_idx = line.find(' ')
            if space_idx == -1:
                dialog_idx = int(line)
            else:
                dialog_idx = int(line[:space_idx])

            if int(dialog_idx) == 1:
                data.append({'your_persona': [], 'partner_persona': [], 'dialog': []})

            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
                data[-1]['your_persona'].append(persona_info)
            if dialog_line[0].startswith('partner\'s persona:'):
                persona_info = dialog_line[0].replace('partner\'s persona: ', '')
                data[-1]['partner_persona'].append(persona_info)

            elif len(dialog_line) > 1:
                data[-1]['dialog'].append(dialog_line[0])
                data[-1]['dialog'].append(dialog_line[1])
        return data


def split_data(data):
    """
    split a multi-turn dialog to multiple {'persona': List[str], 'context': List[str], 'tgt': str} samples
    """
    logger.info("splitting multi-turn dialogs into persona + context + tgt dialogs")
    processed_data = []
    for dialog in tqdm(data):
        assert len(dialog['dialog']) % 2 == 0  # dialog starts from partner and ends with you
        for utter_idx in range(7, len(dialog['dialog'])):
            if utter_idx % 2:  # you
                processed_data.append({'persona': dialog['your_persona'],
                                       'context': dialog['dialog'][: utter_idx],
                                       'tgt': dialog['dialog'][utter_idx]})
            else:  # partner
                processed_data.append({'persona': dialog['partner_persona'],
                                       'context': dialog['dialog'][: utter_idx],
                                       'tgt': dialog['dialog'][utter_idx]})
    return processed_data


def index_data(data, vocab):
    logger.info("tokenize and index data")
    def tokenize(obj):
        if isinstance(obj, str):
            return vocab.string2ids(obj)
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    indexed_data = []
    for dialog in tqdm(data):
        indexed_data.append(tokenize(dialog))
    return indexed_data

def process_train_data(cache_path, train_data_path, vocab_path):
    logger.info("Now processing training data:")
    multi_turn_data = parse_data(train_data_path)
    logging.info(f"After parsing, we get {len(multi_turn_data)} dialogs.")
    processed_data = split_data(data=multi_turn_data)
    logging.info(f"Training dataset has {len(processed_data)} samples.")
    vocab = GPT2Vocab(model_path=vocab_path)
    indexed_data = index_data(data=processed_data, vocab=vocab)
    torch.save(indexed_data, os.path.join(cache_path, "train.cache"))


def process_valid_and_test_data(cache_path, valid_data_path, vocab_path):
    logger.info("Now processing valid and test data:")
    multi_turn_data = parse_data(valid_data_path)
    vocab = GPT2Vocab(model_path=vocab_path)
    # split half of the original validation set into a test set
    random.seed(0)
    random.shuffle(multi_turn_data)
    n_valid_data = len(multi_turn_data)
    valid_multi_turn_data = multi_turn_data[: n_valid_data // 2]
    test_multi_turn_data = multi_turn_data[n_valid_data // 2:]
    print(f"valid data has {n_valid_data} samples, "
          f"now split into valid: {len(valid_multi_turn_data)} and test: {len(test_multi_turn_data)}")
    # for valid
    valid_processed_data = split_data(data=valid_multi_turn_data)
    print(f"Valid dataset has {len(valid_processed_data)} samples.")
    valid_indexed_data = index_data(data=valid_processed_data, vocab=vocab)
    torch.save(valid_indexed_data, os.path.join(cache_path, "valid.cache"))
    # for test
    test_processed_data = split_data(data=test_multi_turn_data)
    print(f"Test dataset has {len(test_processed_data)} samples.")
    test_indexed_data = index_data(data=test_processed_data, vocab=vocab)
    torch.save(test_indexed_data, os.path.join(cache_path, "test.cache"))


if __name__ == "__main__":
    cache_path = "../processed_data/personachat"
    train_data_path = "../raw_data/personachat/train_both_original_no_cands.txt"
    valid_data_path = "../raw_data/personachat/valid_both_original_no_cands.txt"
    gpt2_vocab_path = "../gpt2_vocab"
    process_train_data(cache_path=cache_path, train_data_path=train_data_path, vocab_path=gpt2_vocab_path)
    process_valid_and_test_data(cache_path=cache_path, valid_data_path=valid_data_path, vocab_path=gpt2_vocab_path)