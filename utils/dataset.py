from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import os.path as osp
import time
from PIL import Image
import cv2
import numpy as np
from tqdm import *

def binarize(mask):
    mu = np.unique(mask)
    if np.all(np.array_equal(mu, np.array([  0.,  76.], dtype=np.float32))):
        t = mask.copy()
        t[t == 76] = 1
        return t
    elif np.all(np.array_equal(mu, np.array([   0.,   29.,  150.], dtype=np.float32))):
        t = mask.copy()
        t[t > 29] = 0
        t[t == 29] = 1
        return t
    elif np.all(np.array_equal(mu, np.array([   0.,   29.,  149.], dtype=np.float32))):
        t = mask.copy()
        t[t > 29] = 0
        t[t == 29] = 1
        return t
    elif np.all(np.array_equal(mu, np.array([  0.,  29.,  76.], dtype=np.float32))):
        t = mask.copy()
        t[t == 29] = 1
        t[t > 1] = 0
        return t
    else:
        t = mask.copy()
        t[t > 0] = 1
        return t

class InMemoryImgSegmDataset(Dataset):
    def __init__(self, path, img_path, mask_path,
                 train_transform, test_transform,
                 limit_len=-1):
        """
        :param path: path to directory with images and masks directories
        :param img_path: name of directory with images
        :param mask_path: name of directory with masks
        :param train_transform: dual transform applied to train part
        :param test_transform: dual transform applied to test part
        :param limit_len: how many images load in memory
        """
        self._train_images = []
        self._test_masks = []
        self._train_masks = []
        self._test_images = []
        self._test_transform = test_transform
        self._train_transform = train_transform
        self._path = path
        self._mode = 'train'
        self._img_paths = os.listdir(osp.join(path, img_path))
        self.train, self.test = train_test_split(self._img_paths)
        self._limit_len = limit_len
        self._img_path = img_path
        self._mask_path = mask_path
        self.load()

    def set_mode(self, mode):
        self._mode = mode

    def __len__(self):
        if self._limit_len != -1:
            return self._limit_len
        elif self._mode == 'train':
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, index):
        return self.getitemfrom(index)

    def load(self):
        print('start loading')
        if self._limit_len != -1:
            target_len = self._limit_len
        else:
            target_len = len(self.train)

        for i in tqdm(range(target_len)):
            base_name = self.train[i].split('.')[0]  # name without extension
            im_p = osp.join(self._path, self._img_path, base_name + '.jpg')
            img = cv2.imread(im_p).astype(np.float32)
            self._train_images.append(img.copy())
            m_p = osp.join(self._path, self._mask_path, base_name + '.png')
            mask = binarize(cv2.imread(m_p, cv2.IMREAD_GRAYSCALE).astype(np.float32))
            if not np.all(np.equal(np.unique(mask), np.array([0,1], dtype=np.float32))):
                print(m_p)

            self._train_masks.append(mask)

        if self._limit_len != -1:
            target_len = self._limit_len
        else:
            target_len = len(self.test)

        for i in tqdm(range(target_len)):
            base_name = self.test[i].split('.')[0]  # name without extension
            im_p = osp.join(self._path, self._img_path, base_name + '.jpg')
            img = cv2.imread(im_p).astype(np.float32)
            self._test_images.append(img.copy())
            m_p = osp.join(self._path, self._mask_path, base_name + '.png')
            mask = binarize(cv2.imread(m_p, cv2.IMREAD_GRAYSCALE).astype(np.float32))
            if  not np.all(np.equal(np.unique(mask), np.array([0,1], dtype=np.float32))):
                print('warning ',m_p,' is not binary')
            self._test_masks.append(mask)

    def getitemfrom(self, index):
        if self._mode == 'train':
            return self._train_transform(self._train_images[index], self._train_masks[index])

        return self._test_transform(self._test_images[index], self._test_masks[index])


class InMemoryImgDataset(Dataset):
    def __init__(self, path, limit_len=-1):
        self._path = path
        self._mode = 'train'
        self._img_paths = os.listdir(path)
        self._limit_len = limit_len
        self.__load__()

    def __len__(self):
        if self._limit_len != 1:
            return self._limit_len
        elif self._mode == 'train':
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, index):
        return self.__getitemfrom__(index, self._mode)

    def __load__(self):
        self.train, self.test = train_test_split(self._img_paths)
        self._train_images = []
        self._test_images = []

        if self._limit_len != -1:
            target_len = self._limit_len
        else:
            target_len = len(self.train)

        for i in range(target_len):
            if i % 1000 == 0:
                print('loaded ', str(i))
                time.sleep(0.1)
            img = Image.open(osp.join(self._path, self.train[i]))
            self._train_images.append(img.copy())
            img.close()

        if self._limit_len != -1:
            target_len = self._limit_len
        else:
            target_len = len(self.test)

        for i in range(target_len):
            if i % 1000 == 0:
                print('loaded test', str(i))
                time.sleep(0.1)
            img = Image.open(osp.join(self._path, self.test[i]))
            self._test_images.append(img.copy())
            img.close()

    def __getitemfrom__(self, index, mode):
        if mode == 'train':
            return self._train_images[index],
        return self._test_images[index]
