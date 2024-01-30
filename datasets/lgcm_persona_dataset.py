# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : lgcm_persona_dataset.py
# @Description :


"""
suppose encoder's input is (p_1, ..., p_s, x_1, y_1, ..., x_t), decoder's input is y_t
+ Encoder:
    + Sequence:           <bos> p_1 <eos> ... <bos> p_s <eos> <bos>   x_1 <eos>   <bos> y_1 <eos> ... <bos>   x_t <eos>
    + Type:               <bot> ...                     <bot> <human> ... <human> <bot> ... <bot> ... <human> ... <human>
    + Token Position:     0     1 ...     ... 0     1 ...     0       1 ...        0    1 ...     ... 0       1 ...
    + Utterance Position: 2t    ...       ... 2t    ...       2t-1    ...         2t-2  ...       ... 1       1 ...
+ Decoder:
    + Sequence: <bos> y_t <eos>
    + Type      <bot> ... <bot>
    + Position: 0     1 ...
"""

import logging
from itertools import chain

import torch
from torch.utils.data import Dataset


class LGCMPersonaDataset(Dataset):
    def __init__(self, cache_data_path, vocab, max_history_utterance, max_persona_seq_len, max_context_seq_len,
                 max_tgt_len):
        """
        :param max_history_utterance: how many history utterance to keep (including the last utterance)
        :param max_persona_seq_len: max length for each persona sequence
        :param max_context_seq_len: max length for each context sequence
        :param max_tgt_len: max length for the response
        """
        super(LGCMPersonaDataset, self).__init__()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.vocab = vocab
        dialogs = torch.load(cache_data_path)
        self.dialogs = dialogs

        self.contexts = {"persona": [], "context": []}
        self.logger.info("building data from segments")
        data = [self.build_data(dialog=dialog, vocab=vocab, max_history_utterance=max_history_utterance,
                                max_persona_seq_len=max_persona_seq_len, max_context_seq_len=max_context_seq_len,
                                max_tgt_len=max_tgt_len)
                for dialog in dialogs]

        self.logger.info("padding and converting to tensor")
        self.pad_data = self.get_padding_data(data)

    def __len__(self):
        return self.pad_data["context"].shape[0]

    def __getitem__(self, item):
        return {"context": self.pad_data["context"][item, :],  # context_len
                "context_type_ids": self.pad_data["context_type_ids"][item, :],  # context_len
                "context_position_ids": self.pad_data["context_position_ids"][item, :],  # context_len
                "turn_position_ids": self.pad_data["turn_position_ids"][item, :],  # context_len
                "tgt": self.pad_data["tgt"][item, :],  # tgt_len
                "tgt_type_ids": self.pad_data["tgt_type_ids"][item, :],  # tgt_len
                "tgt_position_ids": self.pad_data["tgt_position_ids"][item, :],  # tgt_len
                }

    def build_data(self, dialog, vocab, max_history_utterance, max_persona_seq_len, max_context_seq_len, max_tgt_len=0):

        self.contexts["persona"].append(dialog['persona'])
        self.contexts["context"].append(dialog['context'][-max_history_utterance:])

        persona = dialog['persona']  # persona 实际上看作一个特殊的 context
        context = dialog['context'][-max_history_utterance:]

        persona = [[vocab.bos_id] + seq[:max_persona_seq_len] + [vocab.eos_id] for seq in persona]
        context = [[vocab.bos_id] + seq[:max_context_seq_len] + [vocab.eos_id] for seq in context]

        # turn_position indicates the distance between the sequence where the current token is locateed and the
        # response, for turn positional encoding
        persona_tp = len(context) + 1
        turn_position_persona = [persona_tp for seq in persona for _ in seq]
        turn_position_context = [len(context) - utter_idx for utter_idx, seq in enumerate(context) for _ in seq]

        context_type_ids_persona = [vocab.bot_id for seq in persona for _ in seq]
        context_type_ids_context = [vocab.human_id if (len(context) - i) % 2 else vocab.bot_id
                                    for i, s in enumerate(context) for _ in s]

        context_position_ids_persona = [i for s in persona for i in range(len(s))]
        context_position_ids_context = [i for s in context for i in range(len(s))]

        persona = list(chain(*persona))
        context = list(chain(*context))

        tgt = dialog["tgt"]
        if max_tgt_len:
            tgt = tgt[:max_tgt_len]
        tgt = [vocab.bos_id] + tgt + [vocab.eos_id]
        tgt_type_ids = [vocab.bot_id] * len(tgt)
        tgt_position_ids = list(range(len(tgt)))

        return {"context": persona + context,
                "tgt": tgt,
                "context_type_ids": context_type_ids_persona + context_type_ids_context,
                "tgt_type_ids": tgt_type_ids,
                "context_position_ids": context_position_ids_persona + context_position_ids_context,
                "tgt_position_ids": tgt_position_ids,
                "turn_position": turn_position_persona + turn_position_context}

    def get_padding_data(self, data):
        pad_data = {"context": [],  # n_samples, context_len
                    "tgt": [],  # n_samples, tgt_len
                    "context_type_ids": [],  # n_samples, context_len
                    "tgt_type_ids": [],  # n_samples, tgt_len
                    "context_position_ids": [],  # n_samples, context_len
                    "tgt_position_ids": [],  # n_samples, tgt_len
                    "turn_position": [],  # n_samples, context_len
                    }
        for instance in data:
            for key_name in instance:
                pad_data[key_name].append(instance[key_name])

        # pad context
        max_context_len = max(len(sequence) for sequence in pad_data["context"])
        for key in ["context", "context_type_ids"]:
            pad_data[key] = self.pad_and_convert_to_tensor(pad_data[key], pad_id=self.vocab.pad_id,
                                                           max_seq_len=max_context_len)
        pad_data["context_position_ids"] = self.pad_and_convert_to_tensor(pad_data["context_position_ids"], pad_id=0,
                                                                          max_seq_len=max_context_len)
        pad_data["turn_position"] = self.pad_and_convert_to_tensor(pad_data["turn_position"], pad_id=0,
                                                                   max_seq_len=max_context_len)

        # pad decoder input
        max_dec_seq_len = max(len(sequence) for sequence in pad_data["tgt"])
        for key in ["tgt", "tgt_type_ids"]:
            pad_data[key] = self.pad_and_convert_to_tensor(pad_data[key], pad_id=self.vocab.pad_id,
                                                           max_seq_len=max_dec_seq_len)
        pad_data["tgt_position_ids"] = self.pad_and_convert_to_tensor(pad_data["tgt_position_ids"], pad_id=0,
                                                                      max_seq_len=max_dec_seq_len)

        return {
            "context": pad_data["context"],
            "context_type_ids": pad_data["context_type_ids"],
            "context_position_ids": pad_data["context_position_ids"],
            "turn_position_ids": pad_data["turn_position"],
            "tgt": pad_data["tgt"],
            "tgt_type_ids": pad_data["tgt_type_ids"],
            "tgt_position_ids": pad_data["tgt_position_ids"]
        }

    def collate_func(self, instances):
        batch_data = {}
        for key in instances[0]:
            batch_data[key] = torch.stack([instance[key] for instance in instances])
        return batch_data

    @staticmethod
    def pad_and_convert_to_tensor(sequences, pad_id, max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = max(len(sequence) for sequence in sequences)
        tensor_data = [seq + [pad_id] * (max_seq_len - len(seq)) for seq in sequences]
        tensor_data = torch.tensor(tensor_data, dtype=torch.long)
        return tensor_data