import utility
import sqlalchemy as sa
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import imread
from torchvision.transforms import functional as func


class Transformer:

    def __init__(self, tf):
        if tf:
            self.transformations = [self._to_pil,
                                    self._resize,
                                    self._adjust_brightness,
                                    self._adjust_contrast,
                                    self._adjust_gamma,
                                    self._adjust_saturation,
                                    self._rotate,
                                    self._to_label,
                                    self._to_tensor]
        else:
            self.transformations = [self._to_pil,
                                    self._resize,
                                    self._to_label,
                                    self._to_tensor]

    def __call__(self, imgset):
        for f in self.transformations:
            imgset = f(imgset)
        return imgset

    @staticmethod
    def _to_label(imgset):
        return imgset[0], utility.image_to_labels(imgset[1])

    @staticmethod
    def _to_pil(imgset):
        return func.to_pil_image(imgset[0]), func.to_pil_image(imgset[1])

    @staticmethod
    def _resize(imgset):
        size = (256, 512)
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
        chance = 0.25
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.2, high=1.8)
        return func.adjust_contrast(imgset[0], v), imgset[1]

    @staticmethod
    def _adjust_gamma(imgset):
        chance = 0.25
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.2, high=1.8)
        return func.adjust_gamma(imgset[0], v), imgset[1]

    @staticmethod
    def _adjust_saturation(imgset):
        chance = 0.25
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.5, high=1.5)
        return func.adjust_saturation(imgset[0], v), imgset[1]

    @staticmethod
    def _rotate(imgset):
        chance = 0.4
        if np.random.random() > chance:
            return imgset
        v = np.random.randint(-30, 30)
        return func.rotate(imgset[0], v), func.rotate(imgset[1], v)


class ImageSet(Dataset):

    def __init__(self, itype, tf=True):
        self.transformer = Transformer(tf)
        self.itype = itype
        self._build_dataset()

    def _build_dataset(self):
        engine = sa.create_engine('sqlite:///data.db')  #
        self.data = utility.get_imageset(engine, self.itype)

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
    ts = ImageSet(1)
    # loader = DataLoader(ts, batch_size=1)

    img = ts.__getitem__(5)
    plt.imshow(img[0].numpy().transpose((1, 2, 0)))
    plt.show()
    data = img[1]
    for r in data:
        for c in r:
            if c not in [0, 1, 2]:
                print(c, end="", flush=True)
        print("")
    result = utility.labels_to_image(data)
    print(result)
    plt.imshow(result)
    plt.show()
