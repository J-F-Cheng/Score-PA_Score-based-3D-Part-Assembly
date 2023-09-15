DATASET="Chair"

python ./train.py  \
    --exp_suffix 'train_1' \
    --category ${DATASET} \
    --train_data_fn "${DATASET}.train.npy" \
    --val_data_fn "${DATASET}.val.npy" \
    --device cuda:0 \
    --num_epoch_every_val 200 \
    --epochs 2000 \
    --level 3 \
    --lr 1e-4 \
    --batch_size 16 \
    --num_workers 8 \
    --num_steps 250 \
    --snr 0.20 \
    --t0 1.0 \
    --cor_steps 1 \
    --cor_final_steps 50 \
    --noise_decay_pow 1
