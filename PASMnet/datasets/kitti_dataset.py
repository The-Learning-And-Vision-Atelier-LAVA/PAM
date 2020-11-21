import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines


class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_occ_filenames, self.disp_noc_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_occ_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None, None
        else:
            disp_occ_images = [x[2] for x in splits]
            disp_noc_images = [x[2].replace('occ', 'noc') for x in splits]
            return left_images, right_images, disp_occ_images, disp_noc_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_occ_filenames:  # has disparity ground truth
            disp_occ = self.load_disp(os.path.join(self.datapath, self.disp_occ_filenames[index]))
            disp_noc = self.load_disp(os.path.join(self.datapath, self.disp_noc_filenames[index]))
        else:
            disp_occ = None

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disp_occ = disp_occ[y1:y1 + crop_h, x1:x1 + crop_w]
            disp_noc = disp_noc[y1:y1 + crop_h, x1:x1 + crop_w]
            occ_mask = ((disp_occ - disp_noc) > 0).astype(np.float32)

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            # # augumentation
            # if random.random() < 0.5:
            #     left_img = torch.flip(left_img, [1])
            #     right_img = torch.flip(right_img, [1])
            #     disp_occ = np.ascontiguousarray(np.flip(disp_occ, 0))

            return {"left": left_img,
                    "right": right_img,
                    "left_disp": disp_occ,
                    "occ_mask": occ_mask}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='edge')
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='edge')
            # pad disparity gt
            if self.disp_occ_filenames is not None:
                # assert len(self.disp_occ_filenames.shape) == 2
                disp_occ = np.lib.pad(disp_occ, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if self.disp_occ_filenames is not None:
                return {"left": left_img,
                        "right": right_img,
                        "left_disp": disp_occ,
                        "top_pad": top_pad,
                        "right_pad": right_pad
                        }
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
