# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import numpy as np
import tarfile
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms


class CUBGazeDataset(data.Dataset):
    def __init__(self, root1, root2, train=True, transform=None):
        '''
        root1:
        the root path of the normal images.
        root2:
        the root path of the gaze images.
        '''
        self.root1 = root1
        self.root2 = root2
        self.train = train
        self.transform = transform
        img1_path = os.path.join(root1, 'train') if train else os.path.join(root1, 'val')
        img2_path = os.path.join(root2, 'train') if train else os.path.join(root2, 'val')
        self.images_1 = []
        self.images_2 = []
        self.label = {}
        # self.prettycool = 0

        def countFiles(root1_path, root2_path):
            assert (os.path.exists(root1_path) and os.path.exists(root2_path))
            total_files = 0
            item_list = os.listdir(root1_path)
            if len(item_list) == 0:
                return 0
            for item in item_list:
                next_path1 = os.path.join(root1_path, item)
                next_path2 = os.path.join(root2_path, item)
                if os.path.isfile(next_path1):
                    total_files += 1
                    self.images_1.append(next_path1)
                    self.images_2.append(next_path2)
                else:
                    total_files += countFiles(next_path1, next_path2)
                    # todo: 这个prettycool需要变成path中的001或者啥的。
                    self.label[int(item) - 1] = item
                    # self.prettycool += 1
            return total_files

        self.num_examples = countFiles(img1_path, img2_path)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        image1_index = self.images_1[index]
        image2_index = self.images_2[index]
        img1 = Image.open(image1_index).convert('RGB')
        img2 = Image.open(image2_index).convert('RGB')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        label = int(image2_index.split('/')[-2]) - 1
        # label = self.label[image2_index.split('/')[-2]]

        return img1, label
        #todo 使用gaze info才使用img2
        # return img1, label, img2


def build_dataset(is_train, args):
    train_transform = [
        transforms.Resize(size=256),
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    test_pipeline = [
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    t = train_transform if is_train else test_pipeline
    transform = transforms.Compose(t)

    root1 = args.data_path
    root2 = root1 + '_gauss'
    dataset = CUBGazeDataset(root1, root2, is_train, transform)

    if args.data_set == 'CUB_species':
        # 增加CUB数据集
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')
        # dataset = datasets.ImageFolder(root, transform=transform)
        # root1 = args.data_path
        # root2 = root1 + '_gaze'
        # dataset = CUBGazeDataset(root1, root2, is_train, transform)
        nb_classes = 200
    elif args.data_set == 'CUB_genera':
        nb_classes = 122
    elif args.data_set == 'CUB_family':
        nb_classes = 37
    elif args.data_set == 'CUB_orders':
        nb_classes = 13

    return dataset, nb_classes
