"""
@author: Yuanhao Cai
@date:  2020.03
"""

import cv2
import os
import glob

import torchvision

from dataset.JointsDataset import JointsDataset


class AnimeDataset(JointsDataset):

    def __init__(self, DATASET, stage, transform=None):
        super().__init__(DATASET, stage, transform)
        self.cur_dir = os.path.split(os.path.realpath(__file__))[0]

        self.image_dir = os.path.join(self.cur_dir, "images", "train" if stage == "train" else "val")
        self.image_paths = glob.glob(os.path.join(self.image_dir, "*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError('fail to read {}'.format(img_path))

        if self.color_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
            img = torchvision.transforms.Resize(self.input_shape)(img)

        return img


if __name__ == '__main__':
    from dataset.attribute import load_dataset
    dataset = load_dataset('ANIME')
    coco = AnimeDataset(dataset, 'val')
    # print(coco.data_num)
