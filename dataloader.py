import utility
import sqlalchemy as sa
import numpy as np
from matplotlib import pyplot as plt
import time
import torch
from pathlib import Path
from scipy.ndimage import imread
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import functional as func


class Transformer:

    def __init__(self):
        self.transformations = [self._to_pil,
                                self._resize,
                                self._adjust_brightness,
                                self._adjust_contrast,
                                self._adjust_gamma,
                                self._adjust_saturation,
                                self._rotate,
                                self._to_tensor]

    def __call__(self, imgset):
        for f in self.transformations:
            imgset = f(imgset)
        return imgset

    @staticmethod
    def _to_pil(imgset):
        return func.to_pil_image(imgset[0]), func.to_pil_image(imgset[1])

    @staticmethod
    def _resize(imgset):
        size = (512, 1024)
        return func.resize(imgset[0], size), func.resize(imgset[1], size)

    @staticmethod
    def _to_tensor(imgset):
        return func.to_tensor(imgset[0]), func.to_tensor(imgset[1])

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


class DataLoader:

    def __init__(self):
        self.transformer = Transformer()

    @staticmethod
    def _get_image(path):
        p = Path(path)
        img = imread(p)
        return img

    def _process_imageset(self, imageset):
        (iid, p1, p2, size) = imageset
        image = self._get_image(p1)
        target = self._get_image(p2)
        return self.transformer((image, target))

    def get_imageset(self, n=64, workers=9, random=True):
        engine = sa.create_engine('sqlite:///data.db')
        images = utility.get_imageset(engine, number=n, random=random)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(self._process_imageset, images)
        engine.dispose()
        return results


# TODO: Test out different values
transformations = [
    {
        'chance': 0.7,
        'f': utility.rotate,
        'vf': lambda: np.random.randint(-60, 60)
    },
    {
        'chance': 0.2,
        'f': utility.contrast_set,
        'vf': lambda: None
    },
    {
        'chance': 0.2,
        'f': utility.blur,
        'vf': lambda: 1
    }
]

if __name__ == '__main__':

    loader = DataLoader()
    t = time.time()
    images = loader.get_imageset(n=64)
    t2 = time.time()
    print(t2-t)

    for img in images:
        plt.imshow(img[0].numpy().transpose((1,2,0)))
        plt.show()
        plt.imshow(img[1].numpy().transpose((1,2,0)))
        plt.show()
