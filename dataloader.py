import utility
import sqlalchemy as sa
import numpy as np
import torch
from matplotlib import pyplot as plt
import time


class DataLoader:

    def __init__(self, root, transforms=[], use_torch=True):
        self.img_root = root
        self.transforms = transforms
        self.use_torch = use_torch

    def get_image(self, n=64, random=True):
        engine = sa.create_engine('sqlite:///labels.db')
        images = utility.get_imageset(engine, self.img_root, number=n, random=random)

        # code below is given that transforms can be combined
        t_images = []
        for i in images:
            img = i
            for t in self.transforms:
                if np.random.random() < t['chance']:
                    img = t['f'](img, t['vf']())
            if self.use_torch:
                img = (torch.from_numpy(img[0]), torch.from_numpy(img[1]))
            t_images.append(img)
        engine.dispose()
        return t_images


# TODO: Test out different values
transforms = [
    {
        'chance': 0.2,
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

    loader = DataLoader('', transforms=transforms, use_torch=False)
    t = time.time()
    images = loader.get_image(n=64)
    t2 = time.time()
    print(t2-t)  # 25 seconds to get 64 images is much, should probably do something about this,
    # first check if significant compared to training time

    for img in images:
        plt.imshow(img[0])
        plt.show()
        plt.imshow(img[1])
        plt.show()
