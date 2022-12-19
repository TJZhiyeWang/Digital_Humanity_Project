#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 131072
#SBATCH --time 10:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1

cd /home/zwang/dh
#years = ["1890", "1899", "1904", "1909", "1915", "1920", "1925", "1935"]

python infer_folder.py ./first/level20_1890/images/ ./first/level20_1890_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_grey/best_model.pth
python infer_folder.py ./first/level20_1899/images/ ./first/level20_1899_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_grey/best_model.pth
python infer_folder.py ./first/level20_1904/images/ ./first/level20_1904_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_grey/best_model.pth

python infer_folder.py ./first/level20_1909/images/ ./first/level20_1909_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./first/level20_1915/images/ ./first/level20_1915_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./first/level20_1920/images/ ./first/level20_1920_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./first/level20_1925/images/ ./first/level20_1925_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./first/level20_1935/images/ ./first/level20_1935_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth


python infer_folder.py ./second/level20_1890/images/ ./second/level20_1890_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_grey/best_model.pth
python infer_folder.py ./second/level20_1899/images/ ./second/level20_1899_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_grey/best_model.pth
python infer_folder.py ./second/level20_1904/images/ ./second/level20_1904_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_grey/best_model.pth

python infer_folder.py ./second/level20_1909/images/ ./second/level20_1909_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./second/level20_1915/images/ ./second/level20_1915_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./second/level20_1920/images/ ./second/level20_1920_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./second/level20_1925/images/ ./second/level20_1925_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./second/level20_1935/images/ ./second/level20_1935_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth


python infer_folder.py ./third/level20_1890/images/ ./third/level20_1890_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_grey/best_model.pth
python infer_folder.py ./third/level20_1899/images/ ./third/level20_1899_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_grey/best_model.pth
python infer_folder.py ./third/level20_1904/images/ ./third/level20_1904_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_grey/best_model.pth

python infer_folder.py ./third/level20_1909/images/ ./third/level20_1909_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./third/level20_1915/images/ ./third/level20_1915_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./third/level20_1920/images/ ./third/level20_1920_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./third/level20_1925/images/ ./third/level20_1925_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
python infer_folder.py ./third/level20_1935/images/ ./third/level20_1935_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
