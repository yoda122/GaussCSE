model_name="bert-base-uncased"
batch_size=16
lr=1e-5

for sim_type in "pos" "pos+neg" "pos+rev" "pos+rev+neg"; do
    poetry run python src/main.py \
        --model_name $model_name \
        --batch_size $batch_size \
        --lr $lr \
        --sim_type $sim_type \
done
