# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/tree/main

import argparse
import json
import logging
from griffin.llama import get_llama_griffin
from griffin.gemma import get_gemma_griffin
from griffin.mistral import get_mistral_griffin
from griffin.opt import get_opt_griffin

from lm_eval import tasks, evaluator, utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


logging.getLogger("openai").setLevel(logging.WARNING)


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

def parse_args():
    parser = argparse.ArgumentParser()
    # LM Eval Harness 
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--sample_checkpoint", type=int, default=0)
    
    # Setup
    parser.add_argument('--model_arch', type=str, default='llama2')
    parser.add_argument("--model_size", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)
    
    # GRIFFIN
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument('--selection_method', type=str, default='topk')
    parser.add_argument('--mode', type=str, default='gen')

    parser.add_argument("--device", type=str, default='cpu')
    

    return parser.parse_args()




def main():
    args = parse_args()


    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    
    if len(task_names) > 1:
        raise NotImplementedError
    
    model_name = hugging_name_dict[args.model_arch](models_sizes_dict[args.model_arch][args.model_size])
    print(model_name)

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)

    schedule_k = [args.density for _ in range(config.num_hidden_layers)]
    
    if args.density < 1:
        model.config.mode = args.mode
        model.config.selection_method = args.selection_method
        model = modify_dict[args.model_arch](model, schedule_k)

    model = model.half()
    model = model.eval().to(args.device)

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    
    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        sample_checkpoint=args.sample_checkpoint,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
        tokenizer=tokenizer,
    )
    
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()

