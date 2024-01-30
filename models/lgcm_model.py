# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : lgcm_model.py
# @Description :


import torch.nn as nn
from modules.local_encoder_module import LocalEncoderModule
from modules.global_encoder_module import GlobalEncoderModule
from modules.decoder_module import DecoderModule


class UtteranceEncoder(nn.Module):
    def __init__(self, core_module, vocab):
        super(UtteranceEncoder, self).__init__()
        self.core_module = core_module
        self.vocab = vocab
        self.pad_id = vocab.pad_id

    def forward(self, src, src_type_ids, src_position_ids, utterance_mask):
        """
        local encoder
        """
        attention_mask = src.ne(self.pad_id).float()
        encoder_out = self.core_module(input_ids=src, token_type_ids=src_type_ids,
                                       position_ids=src_position_ids,
                                       attention_mask=attention_mask, utterance_mask=utterance_mask)
        return encoder_out[0], attention_mask


class ContextEncoder(nn.Module):
    def __init__(self, core_module, vocab):
        super(ContextEncoder, self).__init__()
        self.core_module = core_module
        self.vocab = vocab
        self.pad_id = vocab.pad_id

    def forward(self, input_embds, attention_mask, pairwise_dist, turn_position_ids, output_attentions=None,
                output_gates=False, remove_rpe_weight=False):
        """
        global encoder
        """
        encoder_out = self.core_module(input_embds=input_embds, attention_mask=attention_mask,
                                       pairwise_dist=pairwise_dist, turn_position_ids=turn_position_ids,
                                       output_attentions=output_attentions, output_gates=output_gates,
                                       remove_rpe_weight=remove_rpe_weight)
        if output_attentions:
            return encoder_out[1]
        if output_gates:
            return encoder_out[-1]
        return encoder_out[0], attention_mask


class Decoder(nn.Module):
    def __init__(self, core_module, vocab):
        super(Decoder, self).__init__()
        self.core_module = core_module
        self.vocab = vocab
        self.pad_id = vocab.pad_id
        # init lm head
        embed_dim = core_module.wte.weight.size(1)
        vocab_size = core_module.wte.weight.size(0)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = core_module.wte.weight

    def forward(self, tgt, tgt_type_ids, tgt_position_ids, encoder_memory, cross_attention_mask, past=None,
                return_past=False):
        hidden_states, past = self.core_module(input_ids=tgt, token_type_ids=tgt_type_ids, past=past,
                                               position_ids=tgt_position_ids,
                                               encoder_memory=encoder_memory,
                                               cross_attention_mask=cross_attention_mask)
        lm_logits = self.lm_head(hidden_states)
        if return_past:
            return lm_logits, past
        return lm_logits


class LGCMModel(nn.Module):
    def __init__(self, model_config, vocab):
        super(LGCMModel, self).__init__()
        utterance_encoder_module = LocalEncoderModule(model_config.utterance_encoder_config)
        context_encoder_module = GlobalEncoderModule(model_config.context_encoder_config)
        decoder_module = DecoderModule(model_config.decoder_config)
        decoder_module.wte.weight = utterance_encoder_module.wte.weight

        self.utterance_encoder = UtteranceEncoder(core_module=utterance_encoder_module, vocab=vocab)
        self.context_encoder = ContextEncoder(core_module=context_encoder_module, vocab=vocab)
        self.decoder = Decoder(core_module=decoder_module, vocab=vocab)

        # init loss function
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    def forward(self, src, tgt, src_type_ids, tgt_type_ids, src_position_ids, tgt_position_ids, turn_position_ids,
                pairwise_dist, utterance_mask, lm_labels=None):
        src_embds, src_attention_mask = self.utterance_encoder(src=src, src_type_ids=src_type_ids,
                                                               src_position_ids=src_position_ids,
                                                               utterance_mask=utterance_mask)
        encoder_memory, cross_attention_mask = self.context_encoder(input_embds=src_embds,
                                                                    attention_mask=src_attention_mask,
                                                                    pairwise_dist=pairwise_dist,
                                                                    turn_position_ids=turn_position_ids)
        lm_logits = self.decoder(tgt=tgt, tgt_type_ids=tgt_type_ids, tgt_position_ids=tgt_position_ids,
                                 encoder_memory=encoder_memory, cross_attention_mask=cross_attention_mask)

        if lm_labels is not None:
            lm_loss = self.lm_criterion(lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.shape[-1]),
                                        lm_labels[:, 1:].contiguous().view(-1))
            return {"lm_logits": lm_logits, "lm_loss": lm_loss}
        return {"lm_logits": lm_logits}

