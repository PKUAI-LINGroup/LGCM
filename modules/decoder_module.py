# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : decoder_module.py
# @Description :


import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel
from transformers.modeling_utils import Conv1D
from transformers.modeling_gpt2 import MLP

class SelfAttention(nn.Module):
    # self attention
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()

        n_state = nx  # (nx=n_embd)
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
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
        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd: ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

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
            self, x, layer_past=None, attention_mask=None, use_cache=False, output_attentions=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)

class CrossAttention(nn.Module):
    def __init__(self, nx, config, scale=False):
        super().__init__()

        n_state = nx  # (nx=n_embd)
        assert n_state % config.n_head == 0
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.q_attn = Conv1D(n_state, nx)
        self.kv_attn = Conv1D(n_state * 2, nx)
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
            self, x, ctx, attention_mask=None, output_attentions=False):
        query = self.q_attn(x)
        key, value = self.kv_attn(ctx).split(self.split_size, dim=2)
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


class DecoderBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        """
        :param n_ctx: Dimensionality of the causal mask
        """
        super().__init__()
        nx = config.n_embd

        self.self_attn = SelfAttention(nx, n_ctx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.cross_attn = CrossAttention(nx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_3 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)


    def forward(
            self, x, ctx, layer_past=None, self_attention_mask=None, cross_attention_mask=None, use_cache=False,
            output_attentions=False):
        # self attention
        output_self_attn = self.self_attn(
            x,
            layer_past=layer_past,
            attention_mask=self_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        a = output_self_attn[0]  # output_self_attn: a, present, (self_attentions)
        x = self.ln_1(x + a)

        # cross attention
        output_cross_attn = self.cross_attn(
            x,
            ctx,
            attention_mask=cross_attention_mask,
            output_attentions=output_attentions
        )
        a = output_cross_attn[0]  # output_cross_attn: a, (cross_attentions)
        x = self.ln_2(x + a)

        # FFN
        m = self.mlp(x)
        x = self.ln_3(x + m)

        outputs = [x] + output_self_attn[1:] + output_cross_attn[1:]
        return outputs  # x, present, (self_attentions, cross_attentions)


class DecoderModule(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([DecoderBlock(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
            self,
            input_ids=None,
            encoder_memory=None,  # context embedding from encoder
            past=None,
            self_attention_mask=None,
            cross_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache  # default: True

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Self Attention mask.
        if self_attention_mask is not None:  # self_attention_mask is padding mask, (bsz, tgt_seq_len)
            assert batch_size > 0, "batch_size has to be defined and > 0"
            # Sizes are [batch_size, 1, 1, tgt_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            self_attention_mask = self_attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            self_attention_mask = self_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            self_attention_mask = (1.0 - self_attention_mask) * -10000.0

        # Cross Attention mask.
        if cross_attention_mask is not None:  # padding mask, (bsz, src_seq_len)
            assert batch_size > 0, "batch_size has to be defined and > 0"
            # Sizes are [batch_size, 1, 1, src_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            cross_attention_mask = cross_attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            cross_attention_mask = cross_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            cross_attention_mask = (1.0 - cross_attention_mask) * -10000.0

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_self_attentions = []
        all_cross_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                ctx=encoder_memory,
                layer_past=layer_past,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_self_attentions.append(outputs[2])
                all_cross_attentions.append(outputs[3])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            self_attention_output_shape = input_shape[:-1] + (-1,) + all_self_attentions[0].shape[-2:]
            all_self_attentions = tuple(t.view(*self_attention_output_shape) for t in all_self_attentions)
            cross_attention_output_shape = input_shape[:-1] + (-1,) + all_cross_attentions[0].shape[-2:]
            all_cross_attentions = tuple(t.view(*cross_attention_output_shape) for t in all_cross_attentions)
            outputs = outputs + (all_self_attentions,) + (all_cross_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (self_attentions, cross_attentions)
