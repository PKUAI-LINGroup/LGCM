# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @Description : Run this script to train LGCM.


import argparse
import torch

from trainers.lgcm_trainer import LGCMTrainer
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

    if args.extra_info == "persona":  # training on personachat
        print('loading train dataset:')
        train_dataset = LGCMPersonaDataset(cache_data_path=args.train_dataset, vocab=gpt2_vocab,
                                           max_history_utterance=args.max_history_utterance,
                                           max_persona_seq_len=args.max_persona_seq_len,
                                           max_context_seq_len=args.max_context_seq_len,
                                           max_tgt_len=args.max_tgt_len)
        print(f'train dataset has {len(train_dataset)} samples')

        print('loading valid dataset:')
        valid_dataset = LGCMPersonaDataset(cache_data_path=args.valid_dataset, vocab=gpt2_vocab,
                                           max_history_utterance=args.max_history_utterance,
                                           max_persona_seq_len=args.max_persona_seq_len,
                                           max_context_seq_len=args.max_context_seq_len,
                                           max_tgt_len=args.max_tgt_len)
        print(f'valid dataset has {len(valid_dataset)} samples')
    elif args.extra_info == "none":  # training on dailydialog or multiwoz
        print('loading train dataset:')
        train_dataset = LGCMDataset(cache_data_path=args.train_dataset, vocab=gpt2_vocab,
                                    max_history_utterance=args.max_history_utterance,
                                    max_context_seq_len=args.max_context_seq_len,
                                    max_tgt_len=args.max_tgt_len)
        print(f'train dataset has {len(train_dataset)} samples')

        print('loading valid dataset:')
        valid_dataset = LGCMDataset(cache_data_path=args.valid_dataset, vocab=gpt2_vocab,
                                    max_history_utterance=args.max_history_utterance,
                                    max_context_seq_len=args.max_context_seq_len,
                                    max_tgt_len=args.max_tgt_len)
        print(f'valid dataset has {len(valid_dataset)} samples')
    else:
        raise NotImplementedError

    # initialize model
    model_config = TrmModelConfig(vocab_size=len(gpt2_vocab),
                                  utterance_encoder_n_layer=args.utterance_encoder_n_layer,
                                  context_encoder_n_layer=args.context_encoder_n_layer)

    model = LGCMModel(model_config=model_config, vocab=gpt2_vocab)

    trainer = LGCMTrainer(model=model,
                          args=args,
                          train_dataset=train_dataset,
                          valid_dataset=valid_dataset,
                          device=device,
                          vocab=gpt2_vocab
                          )

    # load checkpoint
    last_epoch = 0
    trainer.train(last_epoch=last_epoch)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2_vocab_dir", default="./gpt2_vocab", help="path to GPT2 tokenizer vocab file")
    parser.add_argument("--extra_info", type=str, help="persona or none")
    parser.add_argument("--train_dataset", type=str, help="processed train dataset path")
    parser.add_argument("--valid_dataset", type=str, help="processed test dataset path")
    parser.add_argument("--max_history_utterance", type=int,
                        help="how many history utterance to keep (including the last utterance)")
    parser.add_argument("--max_persona_seq_len", type=int, default=20,
                        help="for personachat dataset, max length for each sequence in persona")
    parser.add_argument("--max_context_seq_len", type=int, help="max length for each sequence in context")
    parser.add_argument("--max_tgt_len", type=int, help="max target sequence length")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="cpu or cuda")
    parser.add_argument("--n_epochs", type=int, help="number of training epochs")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", default=6.25e-5, type=float, help="learning rate")
    parser.add_argument("--gradient_accumulate_steps", type=int, help="accumulate gradient on several steps")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="clip gradient threshold")
    parser.add_argument("--save_model_dir", type=str, help="path to save model checkpoints")
    parser.add_argument("--log_dir", type=str, help="path to tensorboard log")
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--utterance_encoder_n_layer", type=int, default=3)
    parser.add_argument("--context_encoder_n_layer", type=int, default=3)

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
