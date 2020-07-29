# Parallax-Attention Mechanism (PAM)
Reposity for "Parallax Attention for Unsupervised Stereo Correspondence Learning"

[[arXiv]]()


## Overview

### - Network Architecture

<img width="550" src="https://github.com/LongguangWang/PAM/blob/master/Figs/overview.png"/></div>

### - Features:

* **Encode Stereo Correspondence without A Pre-defined Maximum Disparity Range**

<img width="400" src="https://github.com/LongguangWang/PAM/blob/master/Figs/consistency.png"/></div>


* **Encode Occlusion**

<img width="400" src="https://github.com/LongguangWang/PAM/blob/master/Figs/valid_mask_0.png"/></div>

<img width="450" src="https://github.com/LongguangWang/PAM/blob/master/Figs/valid_mask.png"/></div>


## Applications

### 1. PAM for Unsupervised Stereo Matching (PASMnet)
#### 1.1 Overview
<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/PASMnet.png"/></div>

#### 1.2 Results
<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Fig_PASMnet.png"/></div>


<img width="500" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Tab_PASMnet.png"/></div>


### 2. PAM for Stereo Image Super-Resolution (PASSRnet)

#### 2.1 Overview

<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/PASSRnet.png"/></div>


#### 2.2 Results

<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Fig_PASSRnet.png"/></div>


<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Tab_PASSRnet.png"/></div>


## 3. PAM in the Literature (Updated regularly)
Our PAM provides a compact and flexible module to perform feature fusion or information interaction for stereo images without explicit disparity estimation, which can be applied in many tasks like **stereo 3D object detection, stereo image restoration (e.g., denoising, deblurring, deraining, dehazing and super-resolution), stereo image style transfer, and multi-view stereo**. We are looking forward to more applications of PAM.

### - Stereo Image Super-Resolution
* Wang et al., "Learning Parallax Attention for Stereo Image Super-Resolution", CVPR 2019. [[paper]]() [[code]](https://github.com/LongguangWang/PASSRnet)
* Song et al., "Stereoscopic Image Super-Resolution with Stereo Consistent Feature", AAAI 2020. [[paper]](https://www.aaai.org/ojs/index.php/AAAI/article/view/6880) [code]
* Ying et al., "A Stereo Attention Module for Stereo Image Super-Resolution", SPL. [[paper]](https://ieeexplore.ieee.org/document/8998204) [[code]](https://github.com/XinyiYing/SAM)
* Xie et al., "Non-Local Nested Residual Attention Network for Stereo Image Super-Resolution", ICASSP 2020. [[paper]](https://ieeexplore.ieee.org/document/9054687) [code]

### - Stereo Image Dehazing
* Pang et al. "BidNet: Binocular Image Dehazing without Explicit Disparity Estimation", CVPR 2020. [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_BidNet_Binocular_Image_Dehazing_Without_Explicit_Disparity_Estimation_CVPR_2020_paper.pdf) [code]

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

We would like to thank [Sascha Becher](https://www.flickr.com/photos/stereotron)
 and [Tom Bentz](https://www.flickr.com/photos/tombentz) for the approval of using their cross-eye stereo photographs. We would also like to thank [@akkaze](https://github.com/akkaze) for constructive advice.
 
 ## Contact
For questions, please send an email to wanglongguang15@nudt.edu.cn
