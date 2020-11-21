# Test on SceneFlow
## max_disp = 0
python test.py --max_disp=0 \
               --datapath='D:/LongguangWang/Data/SceneFlow'
               --ckpt='./log/PASMnet_SceneFlow_epoch10.pth'
               --batch_size=16

## max_disp = 192
python test.py --max_disp=192 \
               --datapath='D:/LongguangWang/Data/SceneFlow'
               --ckpt='./log/PASMnet_192_SceneFlow_epoch10.pth'
               --batch_size=16