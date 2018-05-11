import sqlite3 as sql
import pandas as pd
from pandas.io import sql
import sqlalchemy as sa
from pathlib import Path
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


def get_image(imageid, engine):
    res = sql.execute('SELECT width, height FROM IMAGES WHERE ID=?', engine, params=[(imageid)])
    i = res.first()
    if i is None:
        res.close()
        return
    w, h = i[0], i[1]
    res.close()
    res = sql.execute('SELECT labelid, polygon FROM POLYGON WHERE imageid=? ', engine, params=[(imageid)])
    image = np.zeros((h, w, 3), np.uint8)

    for r in res:
        color = (0,0,0)
        print(type(r[1]))
        arr = json.loads(r[1])
        narr = []
        for i in range(len(arr)):
            p = [arr[i][1], arr[i][0]]
            narr.append(p)
        print(narr)
        narr = np.array(narr)
        if r[0] in road:
            color = (255, 0, 0)
        if r[0] in background:
            color = (0, 0, 255)
        cv2.fillPoly(image, narr, color)


def build_labels(engine):
    res = sql.execute('SELECT * FROM labels', engine)
    labels = {}
    for r in res:
        labels[r[1]] = r[0]
    print(labels)


if __name__ == '__main__':
    lpath = Path('E:/gtFine_trainvaltest/gtFine/train/')
    ipath = Path('E:/leftImg8bit_trainvaltest/leftImg8bit/train')
    engine = sa.create_engine('sqlite:///labels.db')

    # ids = import_image('test', 'test2', 2, 2, engine)
    #for r in ids:
    #    print(r)
    #import_labels(ipath, lpath, engine)
    img = get_image(31, engine)
    plt.imshow(img)

    #build_labels(engine)

    print('FINISHED')
