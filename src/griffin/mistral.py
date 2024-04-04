# Adapted from Hugging Face implementation

from griffin.utils import select_neurons
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mistral_griffin(model, k_schedule):
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = MistralMLP(config, k_schedule[i])
        
        new_mlp.gate_proj = l.mlp.gate_proj
        new_mlp.up_proj = l.mlp.up_proj
        new_mlp.down_proj = l.mlp.down_proj
        new_mlp.act_fn = l.mlp.act_fn

        if config.selection_method == 'magnitude':
            gate_stat = l.mlp.gate_proj.weight.data.norm(dim=1)
            up_stat = l.mlp.up_proj.weight.data.norm(dim=1)
            stat = (gate_stat * up_stat).unsqueeze(0)
            _, indices = torch.topk(stat, int(stat.shape[1] * new_mlp.k_factor), dim=-1)
            new_mlp.prepare_reduced_weights(indices)
            new_mlp.mag_mask = torch.ones(stat.shape[-1], dtype=bool)
            new_mlp.mag_mask[indices[0]] = False

        l.mlp = new_mlp
    
    return model


class MistralMLP(nn.Module):
    def __init__(self, config, k_factor):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

        self.k_factor = k_factor
        self.mode = config.mode
        assert self.mode in ['gen', 'class']

    def prepare_reduced_weights(self, topk_indices):
        assert topk_indices.shape[0] == 1
        self.gate_proj_reduced = nn.Linear(self.gate_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.up_proj_reduced = nn.Linear(self.up_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.down_proj_reduced = nn.Linear(len(topk_indices), self.down_proj.weight.data.shape[0], bias=False)

        topk_indices = topk_indices[0]
        self.gate_proj_reduced.weight.data = self.gate_proj.weight.data[topk_indices]
        self.up_proj_reduced.weight.data = self.up_proj.weight.data[topk_indices]
        self.down_proj_reduced.weight.data = self.down_proj.weight.data[:, topk_indices]

    def forward(self, x):
        k_factor = self.k_factor
        if self.mode == 'gen':
            if x.shape[1] > 1:
                int_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

                # GRIFFIN
                if self.config.selection_method != 'magnitude' and k_factor > 0.0:
                    k = int(int_states.shape[-1] * k_factor)
                    neuron_stat = ((int_states / int_states.norm(dim=-1).unsqueeze(-1))).norm(dim=1) # B, D
                    topk_weight, topk_indices = select_neurons(neuron_stat, self.config.selection_method, k)
                    self.prepare_reduced_weights(topk_indices)

                down_proj = self.down_proj(int_states)
                return down_proj
                
            else:
                if k_factor == 0.0:
                    return 0 * x
                else:
                    return self.down_proj_reduced(self.act_fn(self.gate_proj_reduced(x)) * self.up_proj_reduced(x))
        
        elif self.mode == 'class':
            assert x.shape[1] > 1
            int_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            if self.config.selection_method != 'magnitude': ###
                k = int(int_states.shape[-1] * k_factor)
                neuron_stat = ((int_states / int_states.norm(dim=-1).unsqueeze(-1)))[:, :-1].norm(dim=1) # B, D
                topk_weight, topk_indices = select_neurons(neuron_stat, self.config.selection_method, k)
                
                # Not tested for batch size > 1
                mask = torch.zeros_like(int_states[:, -1], dtype=torch.bool)
                mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(mask))
                int_states[:, -1] = mask * int_states[:, -1]
            else:
                int_states[:, -1, self.mag_mask.to(int_states.device)] = 0
                
            down_proj = self.down_proj(int_states)
            return down_proj
        else:
            assert NotImplementedError


