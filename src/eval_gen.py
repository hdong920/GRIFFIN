#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
import argparse
import logging


import numpy as np
from griffin.llama import get_llama_griffin
from griffin.gemma import get_gemma_griffin
from griffin.mistral import get_mistral_griffin
from griffin.opt import get_opt_griffin

import torch
import json
import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from rouge import Rouge



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

models_sizes_dict = {
    'opt': ['125m', '350m', '1.3b', '2.7b', '6.7b', '13b', '30b', '66b'],
    'llama2': ['7b', '13b', '70b'],
    'relu_llama2': ['7B', '13B', '70B'],
    'gemma': ['2b', '7b'],
    'mistral': ['7B'],
}

hugging_name_dict = {
    'opt': lambda x: f'facebook/opt-{x}',
    'llama2': lambda x: f'meta-llama/Llama-2-{x}-hf', 
    'relu_llama2': lambda x: f"SparseLLM/ReluLLaMA-{x}",
    'gemma': lambda x: f'google/gemma-{x}',
    'mistral': lambda x: f'mistralai/Mistral-{x}-v0.1',
}


modify_dict = {
    'opt': get_opt_griffin,
    'llama2': get_llama_griffin,
    'relu_llama2': get_llama_griffin,
    'gemma': get_gemma_griffin,
    'mistral': get_mistral_griffin,
}


def main():
    parser = argparse.ArgumentParser()

    # Setup
    parser.add_argument("--dataset", type=str, default='cnn')
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument('--model_arch', type=str, default='llama2')
    parser.add_argument('--model_size', type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)
    
    # GRIFFIN
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument('--selection_method', type=str, default='topk')
    
    # Generation Parameters
    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument('--greedy', action='store_true')

    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()


    set_seed(args)

    shots = args.shots
    if args.dataset == 'cnn':
        input_paths = [f'../data/cnn_data/cnn_dailymail_{shots}shot.jsonl']
    elif args.dataset == 'xsum':
        input_paths = [f'../data/xsum_data/xsum_{shots}shot.jsonl']
    else:
        raise NotImplementedError

    model_size_name = models_sizes_dict[args.model_arch][args.model_size]

    config = AutoConfig.from_pretrained(hugging_name_dict[args.model_arch](model_size_name), cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(hugging_name_dict[args.model_arch](model_size_name), use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(hugging_name_dict[args.model_arch](model_size_name))
    print("PARAMS: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    

    schedule_k = [args.density for _ in range(config.num_hidden_layers)]
    
    if args.density < 1:
        model.config.mode = 'gen'
        model.config.selection_method = args.selection_method
        
        model = modify_dict[args.model_arch](model, schedule_k)


    model.half()
    model.eval().to(args.device)
    if args.max_length == -1:
        args.max_length = config.max_position_embeddings
    logger.info(args)

    requests = []
    for input_path in input_paths:
         with open(input_path, 'r') as f:
             for line in f:
                 if line.strip() != '':
                     requests.append(json.loads(line))

    requests = requests[:args.sample_num]

    results = []
    rouge = Rouge()

    seq_lens = []
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []

    skipped=0
    
    with torch.no_grad():
        for i, request in enumerate(tqdm.tqdm(requests)):

            stop = ['###']
            temperature = args.temp
            prompt = request['article']
            label = request['summary_gt']
            max_tokens = args.max_tokens
            result = {}
            if args.model_arch == 'gemma':
                input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
            else:
                input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
            if len(input_ids[0]) > args.max_length-max_tokens:
                skipped+=1
                print('skipped', skipped)

            else:
                output_sequences = model.generate(
                    input_ids=input_ids,
                    max_length=max_tokens + len(input_ids[0]),
                    temperature=temperature,
                    top_k=args.k,
                    top_p=1,
                    do_sample=not args.greedy,
                    num_return_sequences=1,
                    return_dict_in_generate=True, output_scores=True,
                    )

                tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
                logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
                top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

                generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
                generate_text = generate_text[: generate_text.find(stop[0])]

                scores = rouge.get_scores(generate_text, label)[0]
                seq_lens.append(len(input_ids[0]))
                rouge1_score_list.append(scores['rouge-1']['f'])
                rouge2_score_list.append(scores['rouge-2']['f'])
                rougel_score_list.append(scores['rouge-l']['f'])

                result['result'] = {
                    "choices": [
                        {
                            "text": generate_text,
                            "logprobs": {
                                "tokens": tokens, 
                                "token_logprobs": logprobs, 
                                "top_logprobs": top_logprobs, 
                                "text_offset": []
                            }, 
                            "finish_reason": "length"
                        }
                    ], 
                    "request_time": {
                        "batch_time": 0, 
                        "batch_size": 1}
                }

                results.append(result)
                print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))

    print("FINAL RESULTS")
    print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))


if __name__ == "__main__":
    main()
