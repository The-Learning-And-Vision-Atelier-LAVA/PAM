import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from datasets.data_io import get_transform, read_all_lines, pfm_imread


class SceneFlowDatset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.left_disp_filenames, self.right_disp_filenames = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        left_disp = [x[2] for x in splits]
        right_disp = [x[2][:-13]+'right/'+x[2][-8:] for x in splits]

        return left_images, right_images, left_disp, right_disp

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        left_disp = self.load_disp(os.path.join(self.datapath, self.left_disp_filenames[index]))
        right_disp = self.load_disp(os.path.join(self.datapath, self.right_disp_filenames[index]))

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            left_disp = left_disp[y1:y1 + crop_h, x1:x1 + crop_w]
            right_disp = right_disp[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            # augumentation
            # if random.random()<0.5:
            #     left_img = torch.flip(left_img, [1])
            #     right_img = torch.flip(right_img, [1])
            #     left_disp = np.ascontiguousarray(np.flip(left_disp, 0))
            #     right_disp = np.ascontiguousarray(np.flip(right_disp, 0))

            return {"left": left_img,
                    "right": right_img,
                    "left_disp": left_disp,
                    "right_disp": right_disp}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = left_disp[h - crop_h:h, w - crop_w: w]
            disparity_right = right_disp[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "left_disp": disparity,
                    "right_disp": disparity_right,
                    "top_pad": 0,
                    "right_pad": 0}
