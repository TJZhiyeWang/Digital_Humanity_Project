from mmseg.apis import inference_segmentor, init_segmentor
import numpy as np
from PIL import Image
import sys
import os
# python infer.py ./level20_1909/1909.png ./level20_1909/1909_infer_ocr.png ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_hr18_512x512_20k_voc12aug/latest.pth
# python infer.py ./level20_1909/1909.png ./level20_1909/1909_infer_fastscnn.png ./mmsegmentation/work_dirs/fast_scnn_lr0.12_8x4_160k_cityscapes.py ./mmsegmentation/work_dirs/work_dirs/fast_scnn_lr0.12_8x4_160k_cityscapes/latest.pth
# python infer.py ./level20_1909/1909.png ./level20_1909/1909_infer_icnet.png ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_hr18_512x512_20k_voc12aug/latest.pth

# img_path = "/home/zwang/dh/level20_1909/images/"
# save_path = "/home/zwang/dh/level20_1909/images_..."
# config_file = "/home/zwang/dh/mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py"
# seg_file = "/home/zwang/dh/mmsegmentation/work_dirs/work_dirs/ocrnet_hr18_512x512_20k_voc12aug/latest.pth"

# python infer_folder.py ./first/images_1909/ ./first/images_1909_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_color/best_model.pth
# python infer_folder.py ./first/images_1909/ ./first/images_1909_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_grey/best_model.pth

# python infer_folder.py ./images_1909/ ./images_1909_ocr/ ./mmsegmentation/work_dirs/fast_scnn_lr0.12_8x4_160k_cityscapes.py ./mmsegmentation/work_dirs/work_dirs/fast_scnn_lr0.12_8x4_160k_cityscapes/best_model.pth
# python infer_folder.py ./images_1909/ ./images_1909_ocr/ ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_hr18_512x512_20k_voc12aug/best_model.pth

img_path = sys.argv[1]
save_path = sys.argv[2]
if not os.path.exists(save_path):
    os.mkdir(save_path)
    os.mkdir(save_path + 'binary/')

config_file = sys.argv[3]
seg_file = sys.argv[4]
model = init_segmentor(config_file, seg_file)
filename = os.listdir(img_path)
for i in filename:
    res = inference_segmentor(model, img_path + str(i))
    Image.fromarray((res[0] * 255).astype(np.uint8)).save(save_path + 'binary/' + str(i))
    pic = model.show_result(img_path + str(i), res, show = False, opacity=0.5)
    Image.fromarray(pic).save(save_path + str(i))
