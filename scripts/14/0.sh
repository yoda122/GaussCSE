model_name="bert-base-uncased"
sim_type="pos+rev+neg"
temperature=0.001

for batch_size in 16 32 64 128 256 512; do
    for lr in 1e-5 3e-5 5e-5; do
        poetry run python src/main.py \
            --model_name $model_name \
            --batch_size $batch_size \
            --lr $lr \
            --sim_type $sim_type \
            --temperature $temperature \
            --device "cuda:0"
    done
done
