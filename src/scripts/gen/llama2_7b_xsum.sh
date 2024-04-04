model=llama2
modelsize=0 
dataset=xsum
shots=1
device=0

# Full
python eval_gen.py \
    --dataset $dataset \
    --shots $shots \
    --model_arch $model \
    --model_size $modelsize \
    --density 1 \
    --device cuda:$device

# Magnitude
python eval_gen.py \
    --dataset $dataset \
    --shots $shots \
    --model_arch $model \
    --model_size $modelsize \
    --density 0.5 \
    --selection_method magnitude \
    --device cuda:$device

# GRIFFIN
python eval_gen.py \
    --dataset $dataset \
    --shots $shots \
    --model_arch $model \
    --model_size $modelsize \
    --density 0.5 \
    --selection_method topk \
    --device cuda:$device

# Sampling-based GRIFFIN
python eval_gen.py \
    --dataset $dataset \
    --shots $shots \
    --model_arch $model \
    --model_size $modelsize \
    --density 0.5 \
    --selection_method sample \
    --device cuda:$device

# Top-k + Sampling-based GRIFFIN
python eval_gen.py \
    --dataset $dataset \
    --shots $shots \
    --model_arch $model \
    --model_size $modelsize \
    --density 0.5 \
    --selection_method topk_sample \
    --device cuda:$device

