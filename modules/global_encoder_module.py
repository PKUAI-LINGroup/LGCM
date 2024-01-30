# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : global_encoder_module.py
# @Description :


import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel
from transformers.modeling_utils import Conv1D


class Attention(nn.Module):
    """
    Inter-Attention
    """
    def __init__(self, nx, config, scale=False, rpe_max_len=20):
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

        self.head_dim = nx // config.n_head
        self.rpek = nn.Embedding(rpe_max_len * 2 + 1, self.head_dim)  # to accept negative distance
        # self.rpev = nn.Embedding(rpe_max_len * 2 + 1, self.head_dim)
        self.rpe_max_len = rpe_max_len

    def _attn(self, q, k, v, attention_mask=None, output_attentions=False, pairwise_dist=None, remove_rpe_weight=False):
        w = torch.matmul(q, k)  # [bsz, n_heads, q_len, kv_len]
        # rpe for k
        if pairwise_dist is not None:
            # q: [bsz, n_heads, q_len, head_dim], k: [bsz, n_heads, head_dim, kv_len]
            bsz = q.shape[0]
            q_len = q.shape[2]
            kv_len = k.shape[3]
            k_rpe = self.rpek(pairwise_dist + self.rpe_max_len)  # [bsz, q_len, kv_len, head_dim]
            k_rpe = k_rpe.transpose(-1, -2)  # [bsz, q_len, head_dim, kv_len]
            q_rpe = q.transpose(1, 2)  # [bsz, q_len, n_heads, head_dim]
            w_rpe = torch.matmul(q_rpe, k_rpe).transpose(1, 2)
            if not remove_rpe_weight:
                w = w + w_rpe

        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # rpe for v
        # w: [bsz, n_heads, q_len, kv_len], v: [bsz, n_heads, kv_len, head_dim]
        attn_results = torch.matmul(w, v)  # [bsz, n_heads, q_len, head_dim]
        # if pairwise_dist is not None:
        #     v_rpe = self.rpev(pairwise_dist + self.rpe_max_len)  # [bsz, q_len, kv_len, head_dim]
        #     w_rpe = w.transpose(1, 2)  # [bsz, q_len, n_heads, kv_len]
        #     attn_results_rpe = torch.matmul(w_rpe, v_rpe)  # [bsz, q_len, n_heads, head_dim]
        #     attn_results_rpe = attn_results_rpe.transpose(1, 2)  # [bsz, n_heads, q_len, head_dim]
        #     attn_results = attn_results + attn_results_rpe
        outputs = [attn_results]

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
            self, x, attention_mask=None, output_attentions=False, pairwise_dist=None, remove_rpe_weight=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, output_attentions, pairwise_dist=pairwise_dist,
                                  remove_rpe_weight=remove_rpe_weight)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)


class Gate(nn.Module):
    """
    Gate
    """
    def __init__(self, hid_dim):
        super(Gate, self).__init__()
        self.linear_layer = nn.Linear(hid_dim * 2, hid_dim, bias=False)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, local_info, global_info, output_gates=False):
        context_info = torch.cat([local_info, global_info], dim=2)
        context_info = self.linear_layer(context_info)
        z_info = self.sigmoid_layer(context_info)
        context_info = (1 - z_info) * global_info + z_info * local_info
        if output_gates:
            return context_info, 1 - z_info
        return context_info


class InterAttnGateBlock(nn.Module):
    """
    This is a global encoder layer, consisting of Inter-Attention and Gate.
    """
    def __init__(self, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        # self.mlp = MLP(4 * nx, config)
        self.gate = Gate(hid_dim=nx)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(
            self, x, local_info, attention_mask=None, output_attentions=False, pairwise_dist=None, output_gates=False,
            remove_rpe_weight=False):
        # C = LayerNorm(MultiHead(InterAttention(c,c,c) + c))
        # Inter-Attention
        output_attn = self.attn(
            x,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            pairwise_dist=pairwise_dist,
            remove_rpe_weight=remove_rpe_weight
        )
        a = output_attn[0]  # output_attn: a, (attentions)

        x = self.ln_1(x + a)
        # Gate
        x = self.gate(local_info=local_info, global_info=x, output_gates=output_gates)
        if output_gates:
            outputs = [x[0]] + output_attn[1:] + [x[1]]
        else:
            outputs = [x] + output_attn[1:]
        return outputs  # x, (attentions), (gates)


class GlobalEncoderModule(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.twpe = nn.Embedding(20, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([InterAttnGateBlock(config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
            self,
            input_embds=None,  # bsz, seq_len, d_model
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            pairwise_dist=None,
            turn_position_ids=None,
            output_gates=False,
            remove_rpe_weight=False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pairwise_dist is None:
            raise ValueError("RPE cannot be implemented without pairwise_dist")

        if input_embds is not None:
            input_shape = input_embds.size()[: -1]
            batch_size = input_embds.shape[0]
        else:
            raise ValueError("You have to specify input_embds")

        if attention_mask is None:
            raise ValueError("You have to specify attention mask to ignore padding")

        if turn_position_ids is None:
            raise ValueError("You have to specify turn_position_ids")

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

        turn_position_embeds = self.twpe(turn_position_ids)
        hidden_states = input_embds + turn_position_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        all_attentions = []
        all_gates = []
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                local_info=input_embds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                pairwise_dist=pairwise_dist,
                output_gates=output_gates,
                remove_rpe_weight=remove_rpe_weight
            )

            hidden_states = outputs[0]

            if output_attentions:
                all_attentions.append(outputs[1])

            if output_gates:
                all_gates.append(outputs[-1])

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
        if output_gates:
            all_gates = tuple(all_gates)
            outputs = outputs + (all_gates,)
        return outputs  # last hidden state, (all hidden_states), (attentions), (gates)
