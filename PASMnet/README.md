# PASMnet

## Requirements
- 

## Train
### Prepare training data
1. Download the Flickr1024 dataset and put the images in `data/train/Flickr1024` 
(Note: In our paper, we also use 60 images in the Middlebury dataset as the training set.)
2. Cd to `data/train` and run `generate_trainset.m` to generate training data.

### Begin to train
```bash
python train.py --scale_factor 4 --device cuda:0 --batch_size 32 --n_epochs 80 --n_steps 30
```

## Test
### Prepare test data
1. Download the [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) dataset and put folders `testing/colored_0` and `testing/colored_1` in `data/test/KITTI2012/original` 
2. Cd to `data/test` and run `generate_testset.m` to generate test data.
3. (optional) You can also download KITTI2015, Middlebury or other stereo datasets and prepare test data in `data/test` as below:
```
  data
  └── test
      ├── dataset_1
            ├── hr
                ├── scene_1
                      ├── hr0.png
                      └── hr1.png
                ├── ...
                └── scene_M
            └── lr_x4
                ├── scene_1
                      ├── lr0.png
                      └── lr1.png
                ├── ...
                └── scene_M
      ├── ...
      └── dataset_N
```

### Demo
```bash
python demo_test.py --scale_factor 4 --device cuda:0 --dataset KITTI2012
```

## Results
![2x](./Figs/results_2x_KITTI2012_KITTI2015.png)

Figure 5. Visual comparison for 2× SR. These results are achieved on “test_image_013” of the KITTI 2012 dataset and “test_image_019” of the KITTI 2015 dataset. 

![4x](./Figs/results_4x_KITTI2015.png)

Figure 6. Visual comparison for 4× SR. These results are achieved on “test_image_004” of the KITTI 2015 dataset.

![2x](./Figs/results_2x_lab.png)

Figure 7. Visual comparison for 2× SR. These results are achieved on a stereo image pair acquired in our laboratory.

## Acknowledgement

This code is built on [GwcNet](https://github.com/xy-guo/GwcNet). We thank the authors for sharing their codes.
