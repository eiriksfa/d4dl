import utility
import sqlalchemy as sa
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import time
import torch
from pathlib import Path
from scipy.ndimage import imread
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import functional as func
from skimage import io


class Transformer:

    def __init__(self):
        self.transformations = [self._to_pil,
                                self._resize,
                                self._adjust_brightness,
                                self._adjust_contrast,
                                self._adjust_gamma,
                                self._adjust_saturation,
                                self._rotate,
                                self._to_label,
                                self._to_tensor]
        self._build_color_map()

    def __call__(self, imgset):
        for f in self.transformations:
            imgset = f(imgset)
        return imgset

    def _build_color_map(self):
        classlib = {
            (0, 0, 0): 0,
            (0, 0, 255): 1,
            (0, 255, 0): 2
        }
        self.color_map = np.ndarray(shape=(256 * 256 * 256), dtype='int64')
        self.color_map[:] = -1
        for rgb, idx in classlib.items():
            rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            self.color_map[rgb] = idx

    def _to_label(self, imgset):
        image = np.dot(imgset[1], np.array([65536, 256, 1], dtype='int64'))
        labelimg = self.color_map[image]
        return imgset[0], labelimg

    @staticmethod
    def _to_pil(imgset):
        return func.to_pil_image(imgset[0]), func.to_pil_image(imgset[1])

    @staticmethod
    def _resize(imgset):
        size = (512, 1024)
        return func.resize(imgset[0], size), func.resize(imgset[1], size)

    @staticmethod
    def _to_tensor(imgset):
        return func.to_tensor(imgset[0]), imgset[1]

    @staticmethod
    def _adjust_brightness(imgset):
        chance = 0.2
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.5, high=1.5)
        return func.adjust_brightness(imgset[0], v), imgset[1]

    @staticmethod
    def _adjust_contrast(imgset):
        chance = 0.2
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.5, high=1.5)
        return func.adjust_contrast(imgset[0], v), imgset[1]

    @staticmethod
    def _adjust_gamma(imgset):
        chance = 0.2
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.5, high=1.5)
        return func.adjust_gamma(imgset[0], v), imgset[1]

    @staticmethod
    def _adjust_saturation(imgset):
        chance = 0.2
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.5, high=1.5)
        return func.adjust_saturation(imgset[0], v), imgset[1]

    @staticmethod
    def _rotate(imgset):
        chance = 0.4
        if np.random.random() > chance:
            return imgset
        v = np.random.randint(-90, 90)
        return func.rotate(imgset[0], v), func.rotate(imgset[1], v)


# class DataLoader:
#
#     def __init__(self):
#         self.transformer = Transformer()
#
#     @staticmethod
#     def _get_image(path):
#         p = Path(path)
#         img = imread(p)
#         return img
#
#     def _process_imageset(self, imageset):
#         (iid, p1, p2, size) = imageset
#         image = self._get_image(p1)
#         target = self._get_image(p2)
#         return self.transformer((image, target))
#
#     def get_imageset(self, n=64, workers=9, random=True):
#         engine = sa.create_engine('sqlite:///data.db')
#         images = utility.get_imageset(engine, number=n, random=random)
#         with ThreadPoolExecutor(max_workers=workers) as executor:
#             results = executor.map(self._process_imageset, images)
#         engine.dispose()
#         return results

class ImageSet2(Dataset):

    def __init__(self):
        self.transformer = Transformer()
        self.img = io.imread('/home/novian/term2/dl4ad/repo2/d4dl/testimg/316.png')
        self.target = io.imread('/home/novian/term2/dl4ad/repo2/d4dl/testimg/317.png')

    def __len__(self):
        return 1

    def _process_image(self):
        return self.transformer((self.img, self.target))

    def __getitem__(self, item):
        return self._process_image()


class ImageSet(Dataset):

    def __init__(self):
        self.transformer = Transformer()
        self._build_dataset()

    def _build_dataset(self):
        engine = sa.create_engine('sqlite:///../data.db')
        self.data = utility.get_imageset(engine)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _get_image(path):
        p = Path(path)
        img = imread(p)
        return img

    def _process_image(self, image):
        (p1, p2) = image
        image = self._get_image(p1)
        target = self._get_image(p2)
        return self.transformer((image, target))

    def __getitem__(self, item):
        image = self._process_image(self.data[item])
        return image


if __name__ == '__main__':
    ts = ImageSet()
    # loader = DataLoader(ts, batch_size=1)
    print(ts.__len__())
    print(ts.__getitem__(1))

    t = time.time()
    t2 = time.time()

    img = ts.__getitem__(5)
    print(img[1])
    print(t2 - t)
    plt.imshow(img[0].numpy().transpose((1, 2, 0)))
    plt.show()
    plt.imshow(img[1].numpy().transpose((1, 2, 0)))
    plt.show()
