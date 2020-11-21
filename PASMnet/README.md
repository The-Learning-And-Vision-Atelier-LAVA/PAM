# PASMnet: Parallax-Attention Stereo Matching Network

Pytorch implementation of "Parallax Attention for Unsupervised Stereo Correspondence Learning", TPAMI 2020

[[arXiv]](http://arxiv.org/abs/2009.08250)

## Overview
<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/PASMnet.png"/></div>

## Requirements
- Python 3.6
- PyTorch >= 1.1.0
- prefetch_generator

## Train
### 1. Prepare training data
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

### 2. Train on SceneFlow
Run `./train.sh` to train on SceneFlow. Please update `datapath` in the bash file as your training data path.

### 3. Finetune on KITTI
Run `./finetune.sh` to finetune on the KITTI 2012/2015 datasets. Please update `datapath` in the bash file as your training data path.

## Test
### 1. Download pre-trained models
Download pre-trained models to `./log`.
- [Google Drive](https://drive.google.com/file/d/1_eXJnK8p-2NF4kxrj3ki6OHwXptO4iYp/view)
- [Baidu Drive [code:fe12]](https://pan.baidu.com/s/1Yllm8992_n8i5YfwufyJ-Q)

### 2. Test on SceneFlow
Run `./test.sh` to evaluate on the test set of the SceneFlow dataset.

### 3. Test on KITTI
Run `./submission.sh` to save png predictions on the test set of the KITTI datasets to the folder `./results`.

## Results
<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Fig_PASMnet.png"/></div>

<img width="500" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Tab_PASMnet.png"/></div>

## Citation
```
@Article{Wang2020Parallax,
  author    = {Longguang Wang and Yulan Guo and Yingqian Wang and Zhengfa Liang and Zaiping Lin and Jungang Yang and Wei An},
  title     = {Parallax Attention for Unsupervised Stereo Correspondence Learning},
  journal   = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  year      = {2020},
}
```

## Acknowledgement

This code is built on [GwcNet](https://github.com/xy-guo/GwcNet). We thank the authors for sharing their codes.
