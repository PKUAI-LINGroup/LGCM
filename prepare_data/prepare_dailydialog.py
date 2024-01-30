# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : prepare_dailydialog.py
# @Description : script to preprocess dailydialog dataset


import os
import logging
import torch
from tqdm import tqdm

from vocabs.gpt2_tokenizer import GPT2Vocab

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class DataPreprocessor:
    def __init__(self, vocab_path):
        self.vocab = GPT2Vocab(model_path=vocab_path)

    def process_data(self, input_file_path, output_file_path):
        multi_turn_data = self.parse_data(input_file_path)
        logging.info(f"After parsing, we get {len(multi_turn_data)} dialogs.")
        split_data = self.split_data(data=multi_turn_data)
        logging.info(f"After splitting, we get {len(split_data)} samples.")
        indexed_data = self.index_data(data=split_data)
        torch.save(indexed_data, output_file_path)

    def parse_data(self, data_path):
        """
        :param data_path:
        :return: List[{'dialog': List[str]}]
        """
        with open(data_path, 'r', encoding='utf-8') as file:
            data = []
            for line in tqdm(file.readlines()):
                line = line.strip()
                if len(line) == 0:
                    continue
                dialog = [seq.strip() for seq in line.split('__eou__')[:-1]]
                assert len(dialog) > 0
                if len(dialog) > 7:
                    data.append({'dialog': dialog})
            return data

    def split_data(self, data):
        """
        split a multi-turn dialog to multiple [{'context': List[str], 'tgt': str}] samples
        """
        split_data = []
        for dialog in tqdm(data):
            for utter_idx in range(7, len(dialog['dialog'])):
                split_data.append({'context': dialog['dialog'][: utter_idx],
                                   'tgt': dialog['dialog'][utter_idx]})
        return split_data

    def index_data(self, data):
        def tokenize(obj):
            if isinstance(obj, str):
                return self.vocab.string2ids(obj)
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        indexed_data = []
        for dialog in tqdm(data):
            indexed_data.append(tokenize(dialog))
        return indexed_data


if __name__ == "__main__":
    data_dir = "../raw_data/dailydialog"
    cache_dir = "../processed_data/dailydialog"
    phase_map = {"train": "train",
                 "validation": "valid",
                 "test": "test"}
    data_processor = DataPreprocessor(vocab_path="../gpt2_vocab")
    for phase in ["train", "validation", "test"]:
        logger.info(f"Now processing {phase} data:")
        input_file_path = os.path.join(data_dir, phase, f"dialogues_{phase}.txt")
        output_file_path = os.path.join(cache_dir, f"{phase_map[phase]}.cache")
        data_processor.process_data(input_file_path=input_file_path, output_file_path=output_file_path)
