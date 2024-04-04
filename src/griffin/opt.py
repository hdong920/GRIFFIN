# Adapted from Hugging Face implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.opt.modeling_opt import OPTAttention, OptFlashAttention2
from griffin.utils import select_neurons


from typing import Optional, Tuple

def get_opt_griffin(model, k_schedule):
    config = model.config
    for i, l in enumerate(model.model.decoder.layers):
        new_l = OPTDecoderLayer(config, k_schedule[i])
        new_l.activation_fn = l.activation_fn
        new_l.load_state_dict(l.state_dict())
        if config.selection_method == 'magnitude':
            stat = l.fc1.weight.data.norm(dim=1).unsqueeze(0)
            _, indices = torch.topk(stat, int(stat.shape[1] * new_l.k_factor), dim=-1)
            new_l.prepare_reduced_weights(indices)
            new_l.mag_mask = torch.ones(stat.shape[-1], dtype=bool)
            new_l.mag_mask[indices[0]] = False

        model.model.decoder.layers[i] = new_l
    
    return model

OPT_ATTENTION_CLASSES = {
    "eager": OPTAttention,
    "flash_attention_2": OptFlashAttention2,
}

class OPTDecoderLayer(nn.Module):
    def __init__(self, config, k_factor):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = OPT_ATTENTION_CLASSES[config._attn_implementation](config=config, is_decoder=True)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = F.relu

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

        self.config = config
        self.k_factor = k_factor
        self.mode = config.mode
        assert self.mode in ['gen', 'class']

    def prepare_reduced_weights(self, topk_indices):
        assert topk_indices.shape[0] == 1
        self.fc1_reduced = nn.Linear(self.fc1.weight.data.shape[1], len(topk_indices))
        self.fc2_reduced = nn.Linear(len(topk_indices), self.fc2.weight.data.shape[0])

        topk_indices = topk_indices[0]
        self.fc1_reduced.weight.data = self.fc1.weight.data[topk_indices]
        self.fc2_reduced.weight.data = self.fc2.weight.data[:, topk_indices]
        
        if self.config.enable_bias:
            self.fc1_reduced.bias.data = self.fc1.bias.data[topk_indices]
            self.fc2_reduced.bias.data = self.fc2.bias.data

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        # hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        k_factor = self.k_factor
        if self.mode == 'gen':
            if hidden_states.shape[1] > 1:
                int_states = self.activation_fn(self.fc1(hidden_states))
                if self.config.selection_method != 'magnitude' and k_factor > 0.0: 
                    k = int(int_states.shape[-1] * k_factor)
                    neuron_stat = ((int_states / int_states.norm(dim=-1).unsqueeze(-1))).norm(dim=1) # B, D
                    topk_weight, topk_indices = select_neurons(neuron_stat, self.config.selection_method, k)
                    self.prepare_reduced_weights(topk_indices)

                down_proj = self.fc2(int_states)
            else:
                if k_factor == 0.0:
                    if self.config.enable_bias:
                        down_proj = 0 * hidden_states + self.fc2_reduced.bias.data.reshape((1, 1, -1))
                    else:
                        down_proj = 0 * hidden_states
                else:
                    down_proj = self.fc2_reduced(self.activation_fn(self.fc1_reduced(hidden_states)))
        elif self.mode == 'class':
            assert hidden_states.shape[1] > 1
            int_states = self.activation_fn(self.fc1(hidden_states))
            if self.config.selection_method != 'magnitude':
                k = int(int_states.shape[-1] * k_factor)
                neuron_stat = ((int_states / int_states.norm(dim=-1).unsqueeze(-1)))[:, :-1].norm(dim=1) # B, D
                topk_weight, topk_indices = select_neurons(neuron_stat, self.config.selection_method, k)

                # Not tested for batch size > 1
                mask = torch.zeros_like(int_states[:, -1], dtype=torch.bool)
                mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(mask))
                int_states[:, -1] = mask * int_states[:, -1]
            else:
                int_states[:, -1, self.mag_mask.to(int_states.device)] = 0

            down_proj = self.fc2(int_states)
        else:
            raise NotImplementedError



        hidden_states = nn.functional.dropout(down_proj, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

