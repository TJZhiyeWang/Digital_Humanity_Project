#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 131072
#SBATCH --time 10:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1

cd /home/zwang/dh/mmsegmentation
# python ./tools/train.py /home/zwang/dh/mmsegmentation/work_dirs/fastfcn_r50-d32_jpu_aspp_512x512_160k_ade20k.py
python ./tools/train.py /home/zwang/dh/mmsegmentation/work_dirs/fast_scnn_lr0.12_8x4_160k_cityscapes.py
python ./tools/train.py /home/zwang/dh/mmsegmentation/work_dirs/icnet_r101-d8_in1k-pre_832x832_80k_cityscapes.py
python ./tools/train.py /home/zwang/dh/mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py
# python ./tools/train.py /home/zwang/dh/mmsegmentation/work_dirs/segformer_mit-b0_512x512_160k_ade20k.py
# python ./tools/train.py /home/zwang/dh/mmsegmentation/work_dirs/segmenter_vit-t_mask_8x1_512x512_160k_ade20k.py