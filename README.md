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


## 3. PAM for Other Applications

Our PAM provides a compact and flexible module to perform feature fusion or information interaction for stereo images without explicit disparity estimation, which can be extended to many other tasks like **stereo 3D object detection, stereo image restoration (e.g., denoising, deblurring, deraining and dehazing), stereo image style transfer, and multi-view stereo**.


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

We would also like to thank [@akkaze](https://github.com/akkaze) for constructive advice.
 
 ## Contact
For questions, please send an email to wanglongguang15@nudt.edu.cn
