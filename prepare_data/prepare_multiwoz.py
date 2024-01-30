# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : prepare_multiwoz.py
"""
@Description :
script to preprocess multiwoz dataset,
modified from https://github.com/guxd/DialogBERT/blob/master/prepare_data.py
"""


import re
import os
import json
import logging
import torch
from tqdm import tqdm

from vocabs.gpt2_tokenizer import GPT2Vocab

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_data(data_path):
    """
    :param data_path:
    :return: List[{'dialog': List[str]}]
    """
    logger.info("loading dialogs from file")
    timepat = re.compile("\d{1,2}[:]\d{1,2}")
    pricepat = re.compile("\d{1,3}[.]\d{1,2}")

    def normalize(text):
        text = text.lower()
        text = re.sub(r'^\s*|\s*$', '', text)  # replace white spaces in front and end
        # hotel domain pfb30
        text = re.sub(r"b&b", "bed and breakfast", text)
        text = re.sub(r"b and b", "bed and breakfast", text)
        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(': sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))
        # normalize postcode
        ms = re.findall(
            '([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]
        text = re.sub(u"(\u2018|\u2019)", "'", text)  # weird unicode bug
        # replace time and and price
        text = re.sub(timepat, ' [value_time] ', text)
        text = re.sub(pricepat, ' [value_price] ', text)
        # text = re.sub(pricepat2, '[value_price]', text)
        # replace st.
        text = text.replace(';', ',')
        text = re.sub('$\/', '', text)
        text = text.replace('/', ' and ')
        # replace other special characters
        text = text.replace('-', ' ')
        text = re.sub('[\":\<>@\(\)]', '', text)
        text = re.sub(' +', ' ', text)  # remove multiple spaces
        # concatenate numbers
        tmp = text
        tokens = text.split()
        i = 1
        while i < len(tokens):
            if re.match(u'^\d+$', tokens[i]) and re.match(u'\d+$', tokens[i - 1]):
                tokens[i - 1] += tokens[i]
                del tokens[i]
            else:
                i += 1
        text = ' '.join(tokens)
        return text

    dialogs = []
    data = json.load(open(data_path, 'r'))
    for dialogue_name in tqdm(data):
        utts = []
        dialogue = data[dialogue_name]
        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            sent = normalize(turn['text'])
            utts.append(sent)
        dialogs.append({"dialog": utts})
    return dialogs[:-2000], dialogs[-2000:-1000], dialogs[-1000:]


def split_data(data):
    """
    split a multi-turn dialog to multiple [{'context': List[str], 'tgt': str}] samples
    """
    logger.info("splitting multi-turn dialogs into context + tgt dialogs")
    processed_data = []
    for dialog in tqdm(data):
        for utter_idx in range(7, len(dialog['dialog'])):
            processed_data.append({'context': dialog['dialog'][: utter_idx],
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


def process_data(cache_path, raw_data_path, vocab_path):
    train_multi_turn_data, valid_multi_turn_data, test_multi_turn_data = parse_data(raw_data_path)
    print(f"train data has {len(train_multi_turn_data)} dialogs")
    print(f"valid data has {len(valid_multi_turn_data)} dialogs")
    print(f"test data has {len(test_multi_turn_data)} dialogs")
    vocab = GPT2Vocab(model_path=vocab_path)
    # process train
    train_processed_data = split_data(data=train_multi_turn_data)
    logging.info(f"Training dataset has {len(train_processed_data)} samples.")
    train_indexed_data = index_data(data=train_processed_data, vocab=vocab)
    torch.save(train_indexed_data, os.path.join(cache_path, "train.cache"))
    # process valid
    valid_processed_data = split_data(data=valid_multi_turn_data)
    logging.info(f"Validation dataset has {len(valid_processed_data)} samples.")
    valid_indexed_data = index_data(data=valid_processed_data, vocab=vocab)
    torch.save(valid_indexed_data, os.path.join(cache_path, "valid.cache"))
    # process test
    test_processed_data = split_data(data=test_multi_turn_data)
    logging.info(f"Test dataset has {len(test_processed_data)} samples.")
    test_indexed_data = index_data(data=test_processed_data, vocab=vocab)
    torch.save(test_indexed_data, os.path.join(cache_path, "test.cache"))


if __name__ == "__main__":
    cache_path = "../processed_data/multiwoz"
    raw_data_path = "../raw_data/multiwoz/data.json"
    gpt2_vocab_path = "../gpt2_vocab"
    process_data(cache_path=cache_path, raw_data_path=raw_data_path, vocab_path=gpt2_vocab_path)
