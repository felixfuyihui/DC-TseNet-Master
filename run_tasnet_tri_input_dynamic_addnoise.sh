#/bin/bash

# Copyright  2019  Northwestern Polytechnical University (author: Yihui Fu)
export PATH="/home/work_nfs/wangke/tools/miniconda3/bin:$PATH"
set -euo pipefail

lr="1e-3"
data_dir="data_cv+tt"
norm_type='gLN'
active_func="relu"
encoder_norm_type='cLN'
save_name="DC-TseNet"
mkdir -p exp/${save_name}

num_gpu=3
batch_size=$[num_gpu*3]

    CUDA_VISIBLE_DEVICES="3,4,5" \
    python -u steps/run_tsenet_tri_input_dynamic_addnoise.py \
    --decode="false" \
    --batch-size=${batch_size} \
    --learning-rate=${lr} \
    --weight-decay=1e-5 \
    --epochs=20 \
    --data-dir=${data_dir} \
    --model-dir="exp/${save_name}" \
    --use-cuda="true" \
    --autoencoder1-channels=256 \
    --autoencoder1-kernel-size=20 \
    --autoencoder2-channels=256 \
    --autoencoder2-kernel-size=5 \
    --autoencoder3-channels=256 \
    --autoencoder3-kernel-size=3 \
    --bottleneck-channels=256 \
    --convolution-channels=512 \
    --convolution-kernel-size=3 \
    --number-blocks=8 \
    --number-repeat=2 \
    --number-speakers=1 \
    --normalization-type=${norm_type} \
    --active-func=${active_func} \
    --cleanlist="./speech_noise_rir_list/speech_data.lst" \
    --noiselist="./speech_noise_rir_list/noise_data.lst" \
    --rirlist="./speech_noise_rir_list/rir_data.lst" \
    --repeat=3 \
    --snr-low=-5 \
    --snr-high=20

