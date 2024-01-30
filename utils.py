# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : utils.py
# @Description :


import os
import random
import torch

import numpy as np

from transformers import GPT2Config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TrmModelConfig:
    def __init__(self, vocab_size, utterance_encoder_n_layer, context_encoder_n_layer, utterance_encoder_n_head=8,
                 context_encoder_n_head=8, decoder_n_layer=6, decoder_n_head=8, n_embd=512):
        self.utterance_encoder_config = GPT2Config(vocab_size=vocab_size, n_embd=n_embd,
                                                   n_layer=utterance_encoder_n_layer, n_head=utterance_encoder_n_head)
        self.context_encoder_config = GPT2Config(vocab_size=vocab_size, n_embd=n_embd, n_layer=context_encoder_n_layer,
                                                 n_head=context_encoder_n_head)
        self.decoder_config = GPT2Config(vocab_size=vocab_size, n_embd=n_embd, n_layer=decoder_n_layer,
                                         n_head=decoder_n_head)



def print_args(args):
    for key in args.__dict__:
        print(f"{key}: {args.__dict__[key]}")


def create_pairwise_dist(turn_position_ids, clip_threhold=0):
    """
    M[i, j] represents the distance (not absolute value) between the utterances where the i-th and j-th token are located.
    """
    pairwise_dist = turn_position_ids.unsqueeze(1).repeat(1, turn_position_ids.shape[1], 1)
    pairwise_dist = pairwise_dist.transpose(-1, -2) - pairwise_dist
    if clip_threhold > 0:
        pairwise_dist = torch.clamp(pairwise_dist, min=-clip_threhold, max=clip_threhold)
    return pairwise_dist


def create_utterance_mask(turn_position_ids):
    """
    :param turn_position_ids: [bsz, seq_len]
    modified from https://github.com/bsantraigi/hier-transformer-pytorch/blob/master/hier_transformer_pytorch/hier_masks.py
    """
    utterance_mask = turn_position_ids.unsqueeze(1).repeat(1, turn_position_ids.shape[1], 1)
    utterance_mask = (utterance_mask * 2 == utterance_mask + utterance_mask.transpose(-1, -2)).float()
    return utterance_mask
