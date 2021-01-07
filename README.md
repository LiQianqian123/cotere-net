# cotere-net
# CoTeRe-Net: Discovering Collaborative Ternary Relations in Videos
By Zhensheng Shi, Cheng Guan, Liangjie Cao, Qianqian Li, Ju Liang, Zhaorui Gu, Haiyong Zheng, Bing Zheng

China Shandong Qingdao, Underwater Vision Lab, Ocean University of China

### Table of Contents
0. [Introduction](#introduction)
0. [Installation ](#installation)
0. [Model Zoo and Baselines](#model-zoo-and-Baselines)
0. [Dataset Preparation](#dataset-preparation)
0. [Citation](#citation)

## Introduction

This repository contains a [Torch](http://torch.ch) implementation for the [CoTeRe-Net](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510375.pdf) algorithm for action recognition. The code is based on [PySlowfast](https://github.com/facebookresearch/SlowFast).

[CoTeRe-Net](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510375.pdf) is a novel relation model that discovers relations of both implicit and explicit cues as well as their collaboration in videos. CoTeRe-Net concerns Collaborative Ternary Relations (CoTeRe), where the ternary relation involves channel (C, for implicit), temporal (T, for implicit), and spatial (S, for explicit) relation (R). We devise a flexible and effective CTSR module to collaborate ternary relations for 3D-CNNs, and then construct CoTeRe-Nets for action recognition.


## Installation 
### Requirements
- Python >= 3.6
- Numpy
- PyTorch 1.3
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- GCC >= 4.9
- PyAV: `conda install av -c conda-forge`
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- PyYaml: (will be installed along with fvcore)
- tqdm: (will be installed along with fvcore)
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- tensorboard: `pip install tensorboard`
- moviepy: (optional, for visualizing video on tensorboard) `conda install -c conda-forge moviepy` or `pip install moviepy`
- [Detectron2](https://github.com/facebookresearch/detectron2):
```
    pip install -U torch torchvision cython
    pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    git clone https://github.com/facebookresearch/detectron2 detectron2_repo
    pip install -e detectron2_repo
    # You can find more details at https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
```

### CoTeRe-Net

Clone CoTeRe-Net repository.
```
git clone https://github.com/zhenglab/cotere-net
```

#### Build CoTeRe-Net

After having the above dependencies, run:
```
cd slowfast-base
python setup.py build develop
```

Now the installation is finished, run the pipeline with:
```
cd video-cotere/scripts
sh run_ssv1_r3d_18_32x1.sh
```


## Model Zoo and Baselines

#### Single-crop (224x224) validation error rate
| Network             | GFLOPS | Top-1 Error |  Download   |
| ------------------- | ------ | ----------- | ------------|
| ResNet-50 (1x64d)   |  ~4.1  |  23.9        | [Original ResNet-50](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained)       |
| ResNeXt-50 (32x4d)  |  ~4.1  |  22.2        | [Download (191MB)](https://dl.fbaipublicfiles.com/resnext/imagenet_models/resnext_50_32x4d.t7)       |
| ResNet-101 (1x64d)  |  ~7.8  |  22.0        | [Original ResNet-101](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained)      |
| ResNeXt-101 (32x4d) |  ~7.8  |  21.2        | [Download (338MB)](https://dl.fbaipublicfiles.com/resnext/imagenet_models/resnext_101_32x4d.t7)      |
| ResNeXt-101 (64x4d) |  ~15.6 |  20.4        | [Download (638MB)](https://dl.fbaipublicfiles.com/resnext/imagenet_models/resnext_101_64x4d.t7)       |


## Dataset Preparation
### Kinetics

The Kinetics Dataset could be downloaded via the code released by ActivityNet:

1. Download the videos via the official [scripts](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).

2. After all the videos were downloaded, resize the video to the short edge size of 256, then prepare the csv files for training, validation, and testing set as `train.csv`, `val.csv`, `test.csv`. The format of the csv file is:

```
path_to_video_1 label_1
path_to_video_2 label_2
path_to_video_3 label_3
...
path_to_video_N label_N
```

All the Kinetics models in the Model Zoo are trained and tested with the same data as [Non-local Network](https://github.com/facebookresearch/video-nonlocal-net/blob/master/DATASET.md). For dataset specific issues, please reach out to the [dataset provider](https://deepmind.com/research/open-source/kinetics).

### AVA

The AVA Dataset could be downloaded from the [official site](https://research.google.com/ava/download.html#ava_actions_download)

We followed the same [downloading and preprocessing procedure](https://github.com/facebookresearch/video-long-term-feature-banks/blob/master/DATASET.md) as the [Long-Term Feature Banks for Detailed Video Understanding](https://arxiv.org/abs/1812.05038) do.

You could follow these steps to download and preprocess the data:

1. Download videos

```
DATA_DIR="../../data/ava/videos"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt

for line in $(cat ava_file_names_trainval_v2.1.txt)
do
  wget https://s3.amazonaws.com/ava-dataset/trainval/$line -P ${DATA_DIR}
done
```

2. Cut each video from its 15th to 30th minute

```
IN_DATA_DIR="../../data/ava/videos"
OUT_DATA_DIR="../../data/ava/videos_15min"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
done
```

3. Extract frames

```
IN_DATA_DIR="../../data/ava/videos_15min"
OUT_DATA_DIR="../../data/ava/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
done
```

4. Download annotations

```
DATA_DIR="../../data/ava/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://research.google.com/ava/download/ava_train_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_val_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_action_list_v2.1_for_activitynet_2018.pbtxt -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_train_excluded_timestamps_v2.1.csv -P ${DATA_DIR}
wget https://research.google.com/ava/download/ava_val_excluded_timestamps_v2.1.csv -P ${DATA_DIR}
```

5. Download "frame lists" ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/val.csv)) and put them in
the `frame_lists` folder (see structure above).

6. Download person boxes ([train](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_train_predicted_boxes.csv), [val](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_val_predicted_boxes.csv), [test](https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_test_predicted_boxes.csv)) and put them in the `annotations` folder (see structure above).
If you prefer to use your own person detector, please see details
in [here](https://github.com/facebookresearch/video-long-term-feature-banks/blob/master/GETTING_STARTED.md#ava-person-detector).


Download the ava dataset with the following structure:

```
ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv
|  |_ val.csv
|_ annotations
   |_ [official AVA annotation files]
   |_ ava_train_predicted_boxes.csv
   |_ ava_val_predicted_boxes.csv
```

You could also replace the `v2.1` by `v2.2` if you need the AVA v2.2 annotation. You can also download some pre-prepared annotations from [here](https://dl.fbaipublicfiles.com/pyslowfast/annotation/ava/ava_annotations.tar).


### Charades
1. Please download the Charades RGB frames from [dataset provider](http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar).

2. Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/charades/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/charades/frame_lists/val.csv)).

Please set `DATA.PATH_TO_DATA_DIR` to point to the folder containing the frame lists, and `DATA.PATH_PREFIX` to the folder containing RGB frames.


### Something-Something V2
1. Please download the dataset and annotations from [dataset provider](https://20bn.com/datasets/something-something).

2. Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/val.csv)).

3. Extract the frames at 30 FPS. (We used ffmpeg-4.1.3 with command
`ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"`
   in experiments.) Please put the frames in a structure consistent with the frame lists.


Please put all annotation json files and the frame lists in the same folder, and set `DATA.PATH_TO_DATA_DIR` to the path. Set `DATA.PATH_PREFIX` to be the path to the folder containing extracted frames.


## Citation
If you use CoTeRe-Net in your research, please cite the paper:
```
@inproceedings{shi2020cotere,
  title={CoTeRe-Net: Discovering Collaborative Ternary Relations in Videos},
  author={Shi, Zhensheng and Guan, Cheng and Cao, Liangjie and Li, Qianqian and Liang, Ju and Gu, Zhaorui and Zheng, Haiyong and Zheng, Bing},
  booktitle={European Conference on Computer Vision},
  pages={379--396},
  year={2020},
  organization={Springer}
}
```
