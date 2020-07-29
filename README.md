# Parallax-Attention Mechanism (PAM)
Pytorch implementation of "Parallax Attention for Unsupervised Stereo Correspondence Learning"

[[arXiv]]()

## Motivation

## Contributions

## Overview
<img width="550" src="https://github.com/LongguangWang/PAM/blob/master/Figs/overview.png"/></div>

## Applications

### 1. PAM for Unsupervised Stereo Matching (PASMnet)
#### 1.1 Overview
<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/PASMnet.png"/></div>

#### 1.2 Results
<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Fig_PASMnet.png"/></div>


<img width="600" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Tab_PASMnet.png"/></div>


### 2. PAM for Stereo Image Super-Resolution (PASSRnet)

#### 2.1 Overview

<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/PASSRnet.png"/></div>


#### 2.2 Results

<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Fig_PASSRnet.png"/></div>


<img width="800" src="https://github.com/LongguangWang/PAM/blob/master/Figs/Tab_PASSRnet.png"/></div>


## 3. PAM for Other Applications (Updated regularly)
Our PAM provides a compact and flexible module to perform feature fusion or information interaction for stereo images without explicit disparity estimation, which can be applied in many other tasks like stereo matching, stereo 3D object detection, stereo image restoration (e.g., denoising, deraining, dehaze, dehazing and super-resolution), and multi-view stereo. 

### Stereo Image Super-Resolution
* Wang et al., "Learning Parallax Attention for Stereo Image Super-Resolution", CVPR 2019. [[paper]]() [[code]]()
* Song et al., "Stereoscopic Image Super-Resolution with Stereo Consistent Feature", AAAI 2020. [[paper]](https://www.aaai.org/ojs/index.php/AAAI/article/view/6880) [[code]]()
* Ying et al., "A Stereo Attention Module for Stereo Image Super-Resolution", SPL. [[paper]](https://ieeexplore.ieee.org/document/8998204) [[code]]()
* Xie et al., "Non-Local Nested Residual Attention Network for Stereo Image Super-Resolution", ICASSP 2020. [[paper]](https://ieeexplore.ieee.org/document/9054687) [[code]]()

### Stereo Image Dehazing
& Pang et al. "BidNet: Binocular Image Dehazing without Explicit Disparity Estimation", CVPR 2020. [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_BidNet_Binocular_Image_Dehazing_Without_Explicit_Disparity_Estimation_CVPR_2020_paper.pdf) [[code]]()

## Acknowledgement

We would like to thank <a href="https://www.flickr.com/photos/stereotron/" target="_blank">Sascha Becher</a>
 and <a href="https://www.flickr.com/photos/tombentz" target="_blank">Tom Bentz</a> for the approval of using their cross-eye stereo photographs. We would also like to thank <a href="https://github.com/akkaze" target="_blank">akkaze</a> for constructive advice.
