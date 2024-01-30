# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : evaluate_utils.py
# @Description :


import numpy as np
from collections import Counter

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist

import torch


def greedy_search_for_lgcm(src, src_type_ids, src_position_ids, tgt_position_ids, vocab,
                           model, max_len, pairwise_dist, utterance_mask, turn_position_ids):
    with torch.no_grad():
        batch_size = src.shape[0]
        device = src.device

        is_end = torch.zeros(batch_size, dtype=torch.bool, device=device)

        results = torch.full((batch_size, 1), fill_value=vocab.bos_id, dtype=torch.long, device=device)
        prevs = torch.full((batch_size, 1), fill_value=vocab.bos_id, dtype=torch.long, device=device)
        past = None

        tgt_type_ids = torch.full(prevs.shape, fill_value=vocab.bot_id, dtype=torch.long, device=device)

        src_embds, src_attention_mask = model.utterance_encoder(src=src, src_type_ids=src_type_ids,
                                                                src_position_ids=src_position_ids,
                                                                utterance_mask=utterance_mask)
        encoder_memory, cross_attention_mask = model.context_encoder(input_embds=src_embds,
                                                                     attention_mask=src_attention_mask,
                                                                     pairwise_dist=pairwise_dist,
                                                                     turn_position_ids=turn_position_ids)

        for i in range(max_len):
            logits, past = model.decoder(tgt=prevs, tgt_type_ids=tgt_type_ids, tgt_position_ids=tgt_position_ids,
                                         encoder_memory=encoder_memory, cross_attention_mask=cross_attention_mask,
                                         past=past, return_past=True)
            selected_idxs = logits[:, -1, :].argmax(-1)
            selected_idxs[is_end] = vocab.pad_id
            is_end[selected_idxs == vocab.eos_id] = 1  # <eos> means end of sentence
            prevs = selected_idxs.unsqueeze(-1)
            tgt_position_ids += 1
            results = torch.cat([results, selected_idxs.unsqueeze(-1)], dim=1)

            if all(is_end.view(-1)):
                break
    return results





class Metrics:
    """
    This class is copied from github.com/guxd/DialogBERT/blob/master/learner.py
    """

    def __init__(self):
        super(Metrics, self).__init__()
        '''
        self.rouge_evaluator = rouge.Rouge(metrics=['rouge-l'],
                           max_n=4,
                           limit_length=True,
                           length_limit=200,
                           length_limit_type='words',
                           apply_avg=True,
                           apply_best=False,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
        '''

    @classmethod
    def sim_bleu(self, hyps, ref):
        """
        :param ref - a list of tokens of the reference
        :param hyps - a list of tokens of the hypothesis
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            try:
                scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                            weights=[1. / 4, 1. / 4, 1. / 4, 1. / 4]))
            except:
                scores.append(0.0)
        return np.max(scores), np.mean(scores)

    @classmethod
    def sim_meteor(self, hyps, ref):
        """
        :param refs - a list of strings representing references
        :param hyps - a list of tokens of the hypothesis
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            # try:
            scores.append(meteor_score([ref], hyp))
            # except:
            #    scores.append(0.0)
        return np.max(scores), np.mean(scores)

    @classmethod
    def sim_nist(self, hyps, ref):
        """
        :param refs - a list of strings representing references
        :param hyps - a list of tokens of the hypothesis
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            try:
                scores.append(sentence_nist([ref], hyp))
            except:
                scores.append(0.0)
        return np.max(scores), np.mean(scores)

    @classmethod
    def sim_rougeL(self, hyps, ref):
        """
        Compute ROUGE-L score given a list of candidates and a reference
        :param hyps: list : candidate sentences to be evaluated
        :param ref: list: reference sentence to be evaluated
        :returns score: float (ROUGE-L score for the candidate evaluated against references)
        This class is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
        """

        def lcs(string, sub):
            """
            Calculates longest common subsequence for a pair of tokenized strings
            :param string : list : tokens from a string split using whitespace
            :param sub : list: shorter string, also split using whitespace
            :returns: length (list of int): length of the longest common subsequence between the two strings
            Note: only gives length of the longest common subsequence, not the actual LCS
            This function is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
            """
            if len(string) < len(sub): sub, string = string, sub
            lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]
            for j in range(1, len(sub) + 1):
                for i in range(1, len(string) + 1):
                    if string[i - 1] == sub[j - 1]:
                        lengths[i][j] = lengths[i - 1][j - 1] + 1
                    else:
                        lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
            return lengths[len(string)][len(sub)]

        def rougeL(hyp, refs):
            assert len(refs) > 0 and type(refs[0]) is list, "number of references should >0 for rouge"
            beta = 1.2
            prec, rec = [], []
            for ref in refs:
                _lcs = lcs(ref, hyp)  # compute the longest common subsequence
                prec.append(_lcs / float(len(hyp)))
                rec.append(_lcs / float(len(ref)))
            prec_max, rec_max = max(prec), max(rec)

            if prec_max != 0 and rec_max != 0:
                score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
            else:
                score = 0.0
            return score

        scores = []
        for hyp in hyps:
            try:
                scores.append(rougeL(hyp, [ref]))
            except:
                print('exception in RougeL')
                scores.append(0.0)
        return np.max(scores), np.mean(scores)

    @classmethod
    def tok_f1(self, predictions, pred_lens, targets, target_lens):
        batch_size = predictions.shape[0]
        f1s = []
        for b in range(batch_size):
            pred = predictions[b][:pred_lens[b]]
            target = targets[b][:target_lens[b]]
            common = Counter(target) & Counter(pred)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.
            precision = 1. * num_same / pred_lens[b]
            recall = 1. * num_same / target_lens[b]
            f1 = (2. * recall * precision) / (precision + recall)
            f1s.append(f1)
        return np.mean(f1)