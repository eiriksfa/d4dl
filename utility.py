import pandas as pd
from pandas.io import sql
import sqlalchemy as sa
from pathlib import Path
from scipy.misc import imsave
import numpy as np
import cv2
import os

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
          'ridergroup': 38,
          'truckgroup': 39}

road = [0, 16]  # Ground?
background = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39]


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def import_image(img, name, width, height, engine, dtype=1):
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
    sql.execute('INSERT INTO Images (ID, Image, IName, IType, Width, Height) VALUES (?, ?, ?, ?, ?, ?)',
                engine, params=[(iid, str(img), name, dtype, int(width), int(height))])
    return iid


def import_label_image(image, path, imgid, ltid, engine):
    path = path.joinpath(str(imgid) + '.png')
    ensure_dir(path)
    imsave(path, image)
    res = sql.execute('select MAX(ID) from Labels', engine)
    iid = res.first()[0]
    iid = 0 if iid is None else iid + 1
    res.close()
    sql.execute('INSERT INTO Labels (ID, Image, IID, LTID) VALUES (?, ?, ?, ?)',
                engine, params=[(iid, str(path), imgid, ltid)])
    return iid


def process_label_image(path):
    js = pd.read_json(path)
    h, w = js['imgHeight'][0], js['imgWidth'][0]
    image = np.zeros((h, w, 3), np.uint8)
    for f in js['objects']:
        color = (0, 0, 0)
        arr = f['polygon']
        arr = np.array(arr, np.int32)
        label = f['label']
        if labels[label] in road:
            color = (0, 255, 0)
        if labels[label] in background:
            color = (0, 0, 255)
        # arr = arr.reshape((-1, 1, 2))
        cv2.fillPoly(image, [arr], color)
    return image, (w, h)


def import_images_sub(imgpath, labelpath, labelimagepath, engine, polyname='gtFine_polygons', imgname='leftImg8bit',
                      dtype=1):
    labeldir = labelpath  # .joinpath(imgpath.stem)
    for img in [d for d in imgpath.iterdir()]:
        jsonn = img.stem.replace(imgname, polyname) + '.json'
        jsonp = labeldir.joinpath(jsonn)
        labelimg, (w, h) = process_label_image(jsonp)
        iid = import_image(img, img.stem, w, h, engine)  # img, name, width, height, engine
        import_label_image(labelimg, labelimagepath, iid, 1, engine)


def import_images(imgpath, labelpath, engine, polyname='gtFine_polygons', imgname='leftImg8bit', dtype=1):
    pass


def get_imageset(engine, type=1):
    s = 'SELECT Images.Image as IImage, Labels.Image as LImage FROM Images ' \
        'inner join Labels on Images.ID=Labels.IID WHERE Images.IType=?'
    res = sql.execute(s, engine, params=[type])
    resl = []
    for r in res:
        p1 = r[0]
        p2 = r[1]
        resl.append((p1, p2))
    return resl


def build_labels(engine):
    res = sql.execute('SELECT * FROM labels', engine)
    labels = {}
    for r in res:
        labels[r[1]] = r[0]
    print(labels)


if __name__ == '__main__':
    # build_labels(engine)
    engine = sa.create_engine('sqlite:///data.db')

    l = ['weimar', 'zurich']
    for d in l:
        path = Path('E:/gtFine_trainvaltest/gtFine/train/' + d + '/')
        p2 = Path('E:/leftImg8bit_trainvaltest/leftImg8bit/train/' + d + '/')
        p3 = Path('E:/labelimgs/train/' + d + '/')
        import_images_sub(p2, path, p3, engine)
    print('FINISHED')
