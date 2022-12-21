# Detection of glacier change using MMSegmentation from the Siegfried map from 1874



This repository is a pipeline to detect glacier change from the Siegfried map from 1874. This pipeline is implemented during the course Foundation of Digital Humanities (DH-405) given in fall 2022 at EPFL. For details please go through our [wiki page](http://fdh.epfl.ch/index.php/Detection_of_glacier_change_using_dhSegment_from_the_Siegfried_map_from_1874). 

## Introduction

In our project, we want to explore glaciers based on the Siegfried map. That means unearthing hidden glacier information behind Siegfried Maps from different years. This repository contains all codes and detailed information on how to implement them.

## Repository Architecture

```
├── Data_Collect
├──
├── Muti_stairs.gh: Bridge generator with Multi-level stairs
├── round_stair_with_long_edges.gh: Bridge generator with round stairs
├── README.md
```

## Data Collection

use `.ipynb` to  retrieve the data from the website. 

...



After we use the VIA tool, the export label file is a JSON file,  use `jsontoimg.ipynb` to convert it to the image file. Here are some training data labeled by us: 
