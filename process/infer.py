from mmseg.apis import inference_segmentor, init_segmentor
import numpy as np
from PIL import Image
import sys
# python infer.py ./level20_1909/1909.png ./level20_1909/1909_infer_ocr.png ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_hr18_512x512_20k_voc12aug/latest.pth
# python infer.py ./level20_1909/1909.png ./level20_1909/1909_infer_fastscnn.png ./mmsegmentation/work_dirs/fast_scnn_lr0.12_8x4_160k_cityscapes.py ./mmsegmentation/work_dirs/work_dirs/fast_scnn_lr0.12_8x4_160k_cityscapes/latest.pth
# python infer.py ./level20_1909/1909.png ./level20_1909/1909_infer_icnet.png ./mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py ./mmsegmentation/work_dirs/work_dirs/ocrnet_hr18_512x512_20k_voc12aug/latest.pth

# img_path = "/home/zwang/dh/level20_1909/1909.png"
# save_path = "/home/zwang/dh/level20_1909/1909_infer.png"
# config_file = "/home/zwang/dh/mmsegmentation/work_dirs/ocrnet_hr18_512x512_20k_voc12aug.py"
# seg_file = "/home/zwang/dh/mmsegmentation/work_dirs/work_dirs/ocrnet_hr18_512x512_20k_voc12aug/latest.pth"

img_path = sys.argv[1]
save_path = sys.argv[2]
config_file = sys.argv[3]
seg_file = sys.argv[4]
model = init_segmentor(config_file, seg_file)

res = inference_segmentor(model, img_path)
pic = model.show_result(img_path, res, show = False, opacity=0.5)
Image.fromarray(pic).save(save_path)