# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : local_encoder_module.py
# @Description :


import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel
from transformers.modeling_utils import Conv1D
from transformers.modeling_gpt2 import MLP


class Attention(nn.Module):
    def __init__(self, nx, config, scale=False):
        super().__init__()

        n_state = nx  # (nx=n_embd)
        assert n_state % config.n_head == 0
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
            self, x, attention_mask=None, output_attentions=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)


class GPT2EncoderBlock(nn.Module):
    def __init__(self, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(
            self, x, attention_mask=None, output_attentions=False):
        output_attn = self.attn(
            x,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        a = output_attn[0]  # output_attn: a, (attentions)

        x = self.ln_1(x + a)
        m = self.mlp(x)
        x = self.ln_2(x + m)

        outputs = [x] + output_attn[1:]
        return outputs  # x, (attentions)


class LocalEncoderModule(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2EncoderBlock(config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
            self,
            input_ids=None,  # bsz, seq_len
            attention_mask=None,
            token_type_ids=None,  # bsz, seq_len
            position_ids=None,  # bsz, seq_len
            output_attentions=None,
            output_hidden_states=None,
            utterance_mask=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("You have to specify input_ids")

        if attention_mask is None:
            raise ValueError("You have to specify attention mask to ignore padding")

        if utterance_mask is None:
            raise ValueError("You have to specify utterance_mask in order to encode each utterance separately")

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        assert batch_size > 0, "batch_size has to be defined and > 0"
        attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

        # utterance mask
        utterance_mask = utterance_mask.unsqueeze(1)
        utterance_mask = (1.0 - utterance_mask) * -10000.0
        attention_mask = utterance_mask + attention_mask

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        all_attentions = []
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]

            if output_attentions:
                all_attentions.append(outputs[1])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (all hidden_states), (attentions)