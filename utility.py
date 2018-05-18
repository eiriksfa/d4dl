import sqlite3 as sql
import pandas as pd
from pandas.io import sql
import sqlalchemy as sa
from pathlib import Path
from skimage import io, transform, filters, exposure
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

labels = {'road': 0,
          'sidewalk': 1,
          'car': 2,
          'sky': 3,
          'terrain': 4,
          'building': 5,
          'vegetation': 6,
          'pole': 7,
          'traffic sign': 8,
          'static': 9,
          'bicycle': 10,
          'person': 11,
          'license plate': 12,
          'rider': 13,
          'ego vehicle': 14,
          'out of roi': 15,
          'ground': 16,
          'traffic light': 17,
          'dynamic': 18,
          'wall': 19,
          'cargroup': 20,
          'fence': 21,
          'bicyclegroup': 22,
          'motorcycle': 23,
          'parking': 24,
          'persongroup': 25,
          'bus': 26,
          'bridge': 27,
          'trailer': 28,
          'polegroup': 29,
          'tunnel': 30,
          'caravan': 31,
          'truck': 32,
          'guard rail': 33,
          'rectification border': 34,
          'rail track': 35,
          'train': 36,
          'motorcyclegroup': 37,
          'ridergroup': 38}

road = [0]
background = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]


def import_image(imgpath, name, width, height, engine, dtype=1):
    """
    Import a single image into the database (without labels)
    :param imgpath: path of image
    :param engine: sql engine
    :return:
    """
    res = sql.execute('select MAX(ID) from Images', engine)
    iid = res.first()[0]
    iid = 0 if iid is None else iid + 1
    res.close()
    sql.execute('INSERT INTO Images (ID, Path, Name, Type, Width, Height) VALUES (?, ?, ?, ?, ?, ?)',
                       engine, params=[(iid, str(imgpath), name, dtype, int(width), int(height))])
    return iid


def import_polygon(imageid, label, polygon, engine):
    labelid = get_label_id(label, engine)
    sql.execute('INSERT INTO Polygon (ImageID, LabelID, Polygon) VALUES (?, ?, ?)',
                engine, params=[(imageid, labelid, str(polygon))])


def import_label(label, engine):
    res = sql.execute('SELECT ID FROM LABELS WHERE (Type=?)', engine, params=[(label)])
    r = res.first()
    if r is not None:  # Check if multiple? Should be unique
        r = r[0]
        return r
    res.close()
    res = sql.execute('select MAX(ID) from Labels', engine)
    iid = res.first()[0]
    iid = 0 if iid is None else iid + 1
    res.close()
    sql.execute('INSERT INTO Labels (ID, Type) VALUES (?, ?)',
                engine, params=[(iid, label)])
    return iid


def get_label_id(label, engine):
    if label in labels:
        return labels[label]
    else:
        lid = import_label(label, engine)
        labels[label] = lid
        return lid


def import_labels(imgpath, labelpath, engine, polyname='gtFine_polygons', imgname='leftImg8bit', dtype=1):
    """
    Imports image/label mapping to database, based on a folder of images and a folder of labels
    NB: Not threadsafe!
    :param imgpath: path to the image root
    :param labelpath: path to the label root
    :param polyname: Relative polygon file name
    :param imgname: Relative image file name
    :param dtype: Type of data (train=1, validation=2, test=3)
    :return:
    """
    for directory in [pth for pth in imgpath.iterdir()]:
        labeldir = labelpath.joinpath(directory.stem)
        for img in [d for d in directory.iterdir()]:
            jsonn = img.stem.replace(imgname, polyname) + '.json'
            jsonp = labeldir.joinpath(jsonn)
            js = pd.read_json(jsonp)
            h, w = js['imgHeight'][0], js['imgWidth'][0]
            iid = import_image(img, img.stem, w, h, engine, dtype)
            for item in js.itertuples(index=True):
                label = getattr(item, 'objects')['label']
                polygon = getattr(item, 'objects')['polygon']
                import_polygon(iid, label, polygon, engine)


def get_output_image(imageid, engine, size=None):
    if size is None:
        res = sql.execute('SELECT width, height FROM IMAGES WHERE ID=?', engine, params=[(imageid)])
        i = res.first()
        if i is None:
            res.close()
            return
        w, h = i[0], i[1]
        res.close()
        size = (w, h)
    w, h = size[0], size[1]
    res = sql.execute('SELECT labelid, polygon FROM POLYGON WHERE imageid=? ', engine, params=[(imageid)])
    image = np.zeros((h, w, 3), np.uint8)

    for r in res:
        color = (0,0,0)
        arr = json.loads(r[1])
        arr = np.array(arr, np.int32)
        if r[0] in road:
            color = (255, 0, 0)
        if r[0] in background:
            color = (0, 0, 255)
        arr = arr.reshape((-1, 1, 2))
        cv2.fillPoly(image, [arr], color)
    return image


def get_input_image(path):
    p = Path(path)
    img = io.imread(p)
    return img


def get_imageset(engine, root, number=1, type=1, random=True):
    if random:
        s = 'SELECT * FROM Images WHERE TYPE=? ORDER BY RANDOM() LIMIT ?;'
    else:
        s = 'SELECT * FROM Images WHERE TYPE=? LIMIT ? WHERE TYPE=?;'
    res = sql.execute(s, engine, params=[(type, number)])
    resl = []
    for r in res:
        id = r[0]
        path = root + r[1]
        size = (r[4], r[5])
        resl.append((id, path, size))
    resi = []
    for r in resl:
        img = get_input_image(r[1])
        out = get_output_image(r[0], engine, r[2])
        resi.append((img, out))
    return resi


def build_labels(engine):
    res = sql.execute('SELECT * FROM labels', engine)
    labels = {}
    for r in res:
        labels[r[1]] = r[0]
    print(labels)


def rescale(images, output_size):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    image = images[0]
    gt_color = images[1]

    h, w = image.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)

    img = transform.resize(image, (new_h, new_w), order=0)
    gt_col = transform.resize(gt_color, (new_h, new_w), order=0)

    return img, gt_col


def rotate(images, val):
    """Rotate an image to the desired angle.

    Args:
        rotate_val (int): Desired rotation value, in degree.
    """

    image = images[0]
    gt_color = images[1]

    img = transform.rotate(image, val, resize=True, order=0)
    gt_col = transform.rotate(gt_color, val, resize=True, order=0)

    return img, gt_col


def flip_lr(images):
    """Flip the image left to right"""

    image = images[0]
    gt_color = images[1]

    img = np.fliplr(image).copy()
    gt_col = np.fliplr(gt_color).copy()

    return img, gt_col


def blur(images, val):
    """Blur an image, simulation of rainy or foggy weather.

    Args:
        blur_val (int): Desired blur value.
    """

    image = images[0]
    gt_color = images[1]
    img = filters.gaussian(image, sigma=val)

    return img, gt_color


def contrast_set(images, val):
    """Change a contrast of an image, simulation of very light/dark condition.

        Args:
            val (tuple): Desired stretch range of the distribution.
            images (tuple): input and output images
        """
    image = images[0]
    gt_color = images[1]
    if val is None:
        img = exposure.rescale_intensity(image)
    else:
        img = exposure.rescale_intensity(image, (val[0], val[1]))

    return img, gt_color


if __name__ == '__main__':
    pass
    # lpath = Path('E:/gtFine_trainvaltest/gtFine/train/')
    # ipath = Path('E:/leftImg8bit_trainvaltest/leftImg8bit/train')
    # engine = sa.create_engine('sqlite:///labels.db')
    #
    # # ids = import_image('test', 'test2', 2, 2, engine)
    # #for r in ids:
    # #    print(r)
    # #import_labels(ipath, lpath, engine)
    # #img = get_image(79, engine)
    #
    # engine.dispose()
    # plt.imshow(img)
    # plt.show()
    #
    # #build_labels(engine)
    #
    # print('FINISHED')
