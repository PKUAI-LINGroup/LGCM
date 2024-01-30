# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : evaluate.py
# @Description : Run this script to evaluate LGCM.


import argparse

import torch

from evaluators.lgcm_evaluator import LGCMEvaluator
from vocabs.gpt2_tokenizer import GPT2Vocab
from datasets.lgcm_dataset import LGCMDataset
from datasets.lgcm_persona_dataset import LGCMPersonaDataset

from models.lgcm_model import LGCMModel

from utils import setup_seed, TrmModelConfig, print_args


def main(args):
    setup_seed(args.seed)
    device = torch.device(args.device)
    print_args(args)

    gpt2_vocab = GPT2Vocab(model_path=args.gpt2_vocab_dir)
    if args.extra_info == "persona":  # evaluating on personachat
        print('loading test dataset:')
        test_dataset = LGCMPersonaDataset(cache_data_path=args.test_dataset, vocab=gpt2_vocab,
                                          max_history_utterance=args.max_history_utterance,
                                          max_persona_seq_len=args.max_persona_seq_len,
                                          max_context_seq_len=args.max_context_seq_len,
                                          max_tgt_len=args.max_tgt_len)
        print(f'test dataset has {len(test_dataset)} samples')
    elif args.extra_info == "none":  # evaluating on dailydialog or multiwoz
        print('loading test dataset:')
        test_dataset = LGCMDataset(cache_data_path=args.test_dataset, vocab=gpt2_vocab,
                                   max_history_utterance=args.max_history_utterance,
                                   max_context_seq_len=args.max_context_seq_len,
                                   max_tgt_len=args.max_tgt_len)
        print(f'test dataset has {len(test_dataset)} samples')
    else:
        raise ValueError("extra info is persona or none")

    # initialize model
    model_config = TrmModelConfig(vocab_size=len(gpt2_vocab),
                                  utterance_encoder_n_layer=args.utterance_encoder_n_layer,
                                  context_encoder_n_layer=args.context_encoder_n_layer)

    model = LGCMModel(model_config=model_config, vocab=gpt2_vocab)

    evaluator = LGCMEvaluator(model=model,
                              args=args,
                              test_dataset=test_dataset,
                              device=device,
                              vocab=gpt2_vocab
                              )
    evaluator.evaluate()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2_vocab_dir", default="./gpt2_vocab", help="path to GPT2 tokenizer vocab file")
    parser.add_argument("--extra_info", type=str, help="persona or none")
    parser.add_argument("--test_dataset", type=str, help="processed test dataset path")
    parser.add_argument("--max_history_utterance", type=int,
                        help="how many history utterance to keep (including the last utterance)")
    parser.add_argument("--max_persona_seq_len", type=int, default=20,
                        help="for personachat dataset, max length for each sequence in persona")
    parser.add_argument("--max_context_seq_len", type=int, help="max length for each sequence in context")
    parser.add_argument("--max_tgt_len", type=int, help="max target sequence length")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="cpu or cuda")
    parser.add_argument("--checkpoint_path", type=str, help="path to load model checkpoint")
    parser.add_argument("--utterance_encoder_n_layer", type=int, default=3)
    parser.add_argument("--context_encoder_n_layer", type=int, default=3)
    parser.add_argument("--max_predict_len", type=int, help="max predicted response sequence length")
    parser.add_argument("--save_result_path", type=str, help="path to save prediction results")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
