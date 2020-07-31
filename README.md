# Parallax-Attention Mechanism (PAM)
Reposity for "Parallax Attention for Unsupervised Stereo Correspondence Learning"

[[arXiv]]()


## Overview

### 1. Network Architecture

<img width="550" src="https://github.com/LongguangWang/PAM/blob/master/Figs/overview.png"/></div>

### 2. Left-Right Consistency & Cycle Consistency

<img width="400" src="https://github.com/LongguangWang/PAM/blob/master/Figs/consistency.png"/></div>


### 3. Valid Mask

<img width="400" src="https://github.com/LongguangWang/PAM/blob/master/Figs/valid_mask_0.png"/></div>

<img width="450" src="https://github.com/LongguangWang/PAM/blob/master/Figs/valid_mask.png"/></div>

### 4. Features

* **Unsupervised stereo correspondence learning without a pre-defined maximum disparity range**
* **Direct regularization on matching cost to produce more reasonable cost distribution**
* **Computational and Memory Efficient**


## Applications

### 1. PAM for Unsupervised Stereo Matching (PASMnet) [[code]](https://github.com/LongguangWang/PAM/tree/master/PASMnet)
#### 1.1 Overview
<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/PASMnet.png"/></div>

#### 1.2 Results
<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Fig_PASMnet.png"/></div>


<img width="500" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Tab_PASMnet.png"/></div>


### 2. PAM for Stereo Image Super-Resolution (PASSRnet) [[code]](https://github.com/LongguangWang/PASSRnet)

#### 2.1 Overview

<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/PASSRnet.png"/></div>


#### 2.2 Results

<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Fig_PASSRnet.png"/></div>


<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Tab_PASSRnet.png"/></div>


## 3. PAM for Other Applications

Our PAM provides a compact and flexible module to perform feature fusion or information interaction for stereo images without explicit disparity estimation, which can be extended to **stereo 3D object detection, stereo image restoration (e.g., super-resolution [1,2,3,4], denoising, deblurring, deraining and dehazing [5]), stereo image style transfer, multi-view stereo, and many other tasks [6,7]**.

[1] Wang et al. "Learning Parallax Attention for Stereo Image Super-Resolution", CVPR 2019.

[2] Ying et al. "A Stereo Attention Module for Stereo Image Super-Resolution", SPL.

[3] Song et al. "Stereoscopic Image Super-Resolution with Stereo Consistent Feature", AAAI 2020.

[4] Xie et al. "Non-Local Nested Residual Attention Network for Stereo Image Super-Resolution", ICASSP 2020.
  
[5] Pang et al. "BidNet: Binocular Image Dehazing Without Explicit Disparity Estimation", CVPR 2020.
  
[6] Wu et al. "Spatial-Angular Attention Network for Light Field Reconstruction", arXiv.

[7] Nakano. "Stereo Vision Based Single-Shot 6D Object Pose Estimation for Bin-Picking by a Robot Manipulator", arXiv


## Citation
```
@InProceedings{Wang2019Learning,
  author    = {Longguang Wang and Yulan Guo and Yingqian Wang and Zhengfa Liang and Zaiping Lin and Jungang Yang and Wei An},
  title     = {Parallax Attention for Unsupervised Stereo Correspondence Learning},
  booktitle = {XXX},
  year      = {2020},
}
```

## Acknowledgement

We would like to thank [@akkaze](https://github.com/akkaze) for constructive advice.
 
 ## Contact
For questions, please send an email to wanglongguang15@nudt.edu.cn
