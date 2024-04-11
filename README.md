# GRIFFIN

This repository contains the implementation of GRIFFIN (**G**ating by **R**epetition **I**n **F**eed**f**orward **I**ntermediate **N**eurons), an efficient and effective method to convert LLM feedforward blocks into mixtures of experts instantaneously, presented in ["Prompt-prompted Mixture of Experts for Efficient LLM Generation"](https://arxiv.org/abs/2404.01365).


Harry Dong, Beidi Chen, Yuejie Chi

Carnegie Mellon University


### Abstract

With the development of transformer-based large language models (LLMs), they have been applied to many fields due to their remarkable utility, but this comes at a considerable computational cost at deployment. Fortunately, some methods such as pruning or constructing a mixture of experts (MoE) aim at exploiting sparsity in transformer feedforward (FF) blocks to gain boosts in speed and reduction in memory requirements. However, these techniques can be very costly and inflexible in practice, as they often require training or are restricted to specific types of architectures. To address this, we introduce GRIFFIN, a novel training-free MoE that selects unique FF experts at the sequence level for efficient generation across a plethora of LLMs with different non-ReLU activation functions. This is possible due to a critical observation that many trained LLMs naturally produce highly structured FF activation patterns within a sequence, which we call flocking. Despite our method's simplicity, we show with 50\% of the FF parameters, GRIFFIN maintains the original model's performance with little to no degradation on a variety of classification and generation tasks, all while improving latency (e.g. 1.25$\times$ speed-up in Llama 2 13B on an NVIDIA L40).


### Usage

GRIFFIN implementations for different models are in `src/griffin/`, and similar implementations for other architectures can be placed here as well. To evaluate on XSum and CNN/DailyMail summarization tasks, use `src/eval_gen.py`. For LM Eval Harness tasks, use `src/lm_eval.py`.

#### Setup

Clone this repository, and then set up the conda environment as follows:

```bash
conda env create -f griffin.yml
conda activate griffin
cd src
```

#### Evaluation

GRIFFIN is designed for generation tasks since the algorithm makes a distinction between the prompt and generation (autoregressive) phases. Example generation evaluations located in `scripts/gen/` can be run like so:

```bash
sh scripts/gen/gemma_7b_coqa.sh 
sh scripts/gen/llama2_7b_xsum.sh 
```

For many classification settings, the model never enters the generation phase, meaning GRIFFIN will produce the same outputs as the full model. For these, we can simulate generation by treating the input sequence except the last token as the prompt and force the model to use the experts for the final token (decribed in more detail in the paper). This is what `--mode class` will do and should be set for all such classification tasks. Examples can be found in `scripts/class/`:

```bash
sh scripts/class/mistral_7b_boolq.sh 
```


### Citation

If you found this repository helpful in your work, please cite our [paper](https://arxiv.org/abs/2404.01365):

    @misc{dong2024promptprompted,
      title={Prompt-prompted Mixture of Experts for Efficient LLM Generation}, 
      author={Harry Dong and Beidi Chen and Yuejie Chi},
      year={2024},
      eprint={2404.01365},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
