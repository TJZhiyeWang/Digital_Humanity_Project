# Detection of glacier change using MMSegmentation from the Siegfried map from 1874



This repository is a pipeline to detect glacier change from the Siegfried map from 1874. This pipeline is implemented during the course Foundation of Digital Humanities (DH-405) given in fall 2022 at EPFL. For details please go through our [wiki page](http://fdh.epfl.ch/index.php/Detection_of_glacier_change_using_dhSegment_from_the_Siegfried_map_from_1874). 

## Introduction

In our project, we want to explore glaciers based on the Siegfried map. That means unearthing hidden glacier information behind Siegfried Maps from different years. This repository contains all codes and detailed information on how to implement them.


## Set Up Environment

### Install pytorch and mmsegmentation environment
Install pytorch according to local cofiguration (pytorch 1.13.0+cu116) and then install openmim

```
pip3 install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmsegmentation.git
cd ./mmsegmentation
pip install -e .
```

### Customize dataset
Download dataset from Google drive [Training data](https://drive.google.com/drive/folders/1KdLF_WiWUH_1xi4EublGl6ZUBeBJ07g3?usp=share_link)
And a [Tutorial](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/tutorials/customize_datasets.md) about how to customize own dataset in a mmsegmentation git repository.
After this step, we have an unique name of our customized dataset in the mmopenLab system.

### Customize training configs file & Training model
Assign `data_root` to your local training data directory
Enter to mmsegmentation git repository and run `./tools/train.py` with your customized configs file or use ours.

`python tools/train.py ./configs/ocrnet_hr18_1024x1024_2k_dhdata.py`

## Repository Architecture

```
├── collect
├── ── crawl.ipynb
├── ── concat_img.ipynb
├── label
├── ── geo.ipynb
├── ── generate_label.py
├── ── generate_label_modified.py
├── ── jsontoimg.ipynb
├── infer.py
├── infer_folder.py
├── cmpglacier.ipynb
├── train_model.sh
├── infer_model.sh
├── README.md
```

**For Data Collection**

 `./collect/crawl.ipynb` is used to retrieve the data from the website. Modify the year parameters and download corresponding year's meta data.

 `./collect/concat_img.ipynb` is used to concatenate meta data, crop into 1024x1024 size, stitch together.

**For Data Labeling**

 `./label/geo.ipynb` is used to linear assign geographical coordinate to a pixel(Discard due to not aligned).

 `./label/jsontoimg.ipynb` is used to convert JSON files to the image file. Because when we use the VIA tool, the export label file is a JSON file. (We have already labeled some, dowanload them from the above training data link)

`./label/generate_label.py` and `./label/generate_label_modified.py` is used to generate the corresponding label image file. We can also blend the label with the original images.

**For Prediction and Analysis**

`./infer.py` and  `./infer_folder.py` is used to predict glacier area for  a single image or a folder of images

`./cmpglacier.ipynb` is used to compare predicted glacier areas from different years. It will generate a new image that uses red color to indicate the non-overlapping area.



