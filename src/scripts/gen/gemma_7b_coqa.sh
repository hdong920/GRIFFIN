model=gemma
modelsize=1
dataset=coqa
shots=0
mode=gen
device=0

# Full
python lm_eval.py \
    --tasks $dataset \
    --num_fewshot $shots \
    --model_arch $model \
    --model_size $modelsize \
    --density 1 \
    --device cuda:$device

# Magnitude
python lm_eval.py \
    --tasks $dataset \
    --num_fewshot $shots \
    --model_arch $model \
    --model_size $modelsize \
    --density 0.5 \
    --mode $mode \
    --selection_method magnitude \
    --device cuda:$device

# GRIFFIN
python lm_eval.py \
    --tasks $dataset \
    --num_fewshot $shots \
    --model_arch $model \
    --model_size $modelsize \
    --density 0.5 \
    --mode $mode \
    --selection_method topk \
    --device cuda:$device
