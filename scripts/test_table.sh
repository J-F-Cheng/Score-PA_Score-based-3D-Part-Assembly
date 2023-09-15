DATASET="Table"
NUM_STEPS=250
COR_STEPS=1
COR_FINAL_STEPS=50
NDP=1
SNR=0.20
CDS_T=0.5
EXP_NAME="${DATASET}_${SNR}_${COR_STEPS}_${COR_FINAL_STEPS}_${NDP}_${NUM_STEPS}_${QDS_T}_${CDS_T}_test_1"

python ./test.py  \
    --exp_suffix "${EXP_NAME}" \
    --category ${DATASET} \
    --train_data_fn "${DATASET}.train.npy" \
    --val_data_fn "${DATASET}.val.npy" \
    --device cuda:0 \
    --model_dir pretrained_models/table.pth \
    --level 3 \
    --batch_size 4 \
    --num_workers 8 \
    --snr ${SNR} \
    --t0 1.0 \
    --cor_steps ${COR_STEPS} \
    --cor_final_steps ${COR_FINAL_STEPS} \
    --noise_decay_pow ${NDP} \
    --repeat_times_per_shape 10 \
    --num_steps ${NUM_STEPS} \
    --cdsThresh ${CDS_T}
