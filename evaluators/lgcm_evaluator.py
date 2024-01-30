# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : lgcm_evaluator.py
# @Description :


import os
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import create_pairwise_dist, create_utterance_mask
from evaluators.evaluate_utils import greedy_search_for_lgcm, Metrics


class LGCMEvaluator:
    def __init__(self, model, args, test_dataset, device, vocab):
        self.args = args
        self.device = device
        self.model = model.to(device)
        # load checkpoint
        self.load_state_dict(torch.load(args.checkpoint_path, map_location=self.device))
        print('Weights loaded from {}'.format(args.checkpoint_path))
        self.vocab = vocab
        self.test_dataset = test_dataset

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)

    def evaluate(self):
        self.model.eval()
        test_dataloader = DataLoader(self.test_dataset, batch_size=1,
                                     collate_fn=self.test_dataset.collate_func, num_workers=4)
        tqdm_data = tqdm(test_dataloader, desc='Test: ')
        test_losses = []
        bleus, meteors, nists, rougeLs = [], [], [], []
        with torch.no_grad():
            with open(self.args.save_result_path, "w") as fout:
                for i, data in enumerate(tqdm_data):
                    data = {key: data[key].to(self.device) for key in data}
                    pairwise_dist = create_pairwise_dist(data["turn_position_ids"])
                    utterance_mask = create_utterance_mask(data["turn_position_ids"])
                    model_out = self.model(src=data['context'],
                                           tgt=data['tgt'],
                                           src_type_ids=data['context_type_ids'],
                                           tgt_type_ids=data['tgt_type_ids'],
                                           src_position_ids=data['context_position_ids'],
                                           tgt_position_ids=data['tgt_position_ids'],
                                           turn_position_ids=data['turn_position_ids'],
                                           lm_labels=data["tgt"],
                                           pairwise_dist=pairwise_dist,
                                           utterance_mask=utterance_mask
                                           )

                    test_losses.append(model_out["lm_loss"].item())

                    predict_ids = greedy_search_for_lgcm(
                        src=data['context'],
                        src_type_ids=data['context_type_ids'],
                        src_position_ids=data['context_position_ids'],
                        tgt_position_ids=data['tgt_position_ids'][:, 0].unsqueeze(-1),
                        max_len=self.args.max_predict_len,
                        vocab=self.vocab,
                        model=self.model,
                        pairwise_dist=pairwise_dist,
                        utterance_mask=utterance_mask,
                        turn_position_ids=data['turn_position_ids'],
                    )

                    # context
                    context = self.test_dataset.dialogs[i]["context"]
                    context_strings = []
                    for sent_idx, sent in enumerate(context):
                        sent_string = self.vocab.ids2string(sent, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
                        context_strings.append(sent_string)
                    # tgt
                    ref_sent = self.vocab.ids2string(self.test_dataset.dialogs[i]["tgt"],
                                                     skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)

                    # predict
                    pred_sent = self.vocab.ids2string(predict_ids[0],
                                                      skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)

                    fout.write(f"sample {i + 1}:\n")
                    for sent_idx, sent_string in enumerate(context_strings):
                        fout.write(f"context {sent_idx + 1}: {sent_string}\n")
                    fout.write(f"gt: {ref_sent}\n")
                    fout.write(f"pr: {pred_sent}\n\n")

                    _, avg_bleu = Metrics.sim_bleu([pred_sent.split()], ref_sent.split())
                    bleus.append(avg_bleu)
                    _, avg_meteor = Metrics.sim_meteor([pred_sent.split()], ref_sent.split())
                    meteors.append(avg_meteor)
                    _, avg_nist = Metrics.sim_nist([pred_sent.split()], ref_sent.split())
                    nists.append(avg_nist)
                    _, avg_rougeL = Metrics.sim_rougeL([pred_sent.split()], ref_sent.split())
                    rougeLs.append(avg_rougeL)

        ave_lm_loss = float(np.mean(test_losses))
        ppl = torch.exp(torch.tensor(ave_lm_loss)).item()
        bleu = float(np.mean(bleus))
        meteor = float(np.mean(meteors))
        nist = float(np.mean(nists))
        rougeL = float(np.mean(rougeLs))

        print(f"loss: {ave_lm_loss}\n"
              f"ppl: {ppl}\n"
              f"bleu: {bleu}\n"
              f"meteor: {meteor}\n"
              f"nist: {nist}\n"
              f"rougeL: {rougeL}")