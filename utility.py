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

# Image - Label mapping
image_label = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (255, 0, 0): 3
}
color_map = np.ndarray(shape=(256 * 256 * 256), dtype='int64')
color_map[:] = 0
for rgb, idx in image_label.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    color_map[rgb] = idx

# Label - Image mapping
label_image = {
        0: (0, 0, 0),
        1: (0, 0, 255),
        2: (0, 255, 0),
        3: (255, 0, 0)
    }
likeys, livalues = zip(*label_image.items())
limap = np.empty((max(likeys) + 1, 3), int)
limap[list(likeys), :] = livalues


def ensure_dir(file_path):
    #directory = os.path.dirname(file_path)
    if not os.path.exists(str(file_path)):
        os.makedirs(str(file_path))


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
    ensure_dir(path)
    path = path.joinpath(str(imgid) + '.png')
    imsave(path, image)
    return import_label_image_sub(path, imgid, ltid, engine)


def import_label_image_sub(path, imgid, ltid, engine):
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
        iid = import_image(img, img.stem, w, h, engine, dtype=dtype)  # img, name, width, height, engine
        import_label_image(labelimg, labelimagepath, iid, 1, engine)


def import_images(ip, lp, op, engine, polyname='gtFine_polygons', imgname='leftImg8bit'):
    p = {'train': 1, 'val': 2, 'test': 3}
    for dir in [d for d in ip.iterdir()]:
        if dir.stem not in p:
            continue
        v = dir.stem
        print('one maindir done')
        for subdir in [d for d in dir.iterdir()]:
            s = subdir.stem
            nlp = lp.joinpath(v + '/' + s)
            nop = op.joinpath(v + '/' + s)
            import_images_sub(subdir, nlp, nop, engine, polyname=polyname, imgname=imgname, dtype=p[v])
            print('one subdir done')


def import_car_images(p, engine, imgname='out', labelname='label'):
    dirn = 0
    for dir in [d for d in p.iterdir()]:
        dirn += 1
        t = 2 if (dirn % 6) == 0 else 1
        for file in [d for d in dir.iterdir()]:
            n = file.stem.split('_')
            it = n[1]
            ii = n[0]
            if it == labelname:
                continue
            elif it == imgname:
                iid = import_image(file, 'None', 2380, 1281, engine, t)
                f2 = dir.joinpath(ii + '_' + labelname + '.png')
                import_label_image_sub(f2, iid, t, engine)


def import_car_images2(p, engine, imgname='out', labelname='label', labeltypename='3c'):
    i = 0
    for file in [d for d in p.iterdir()]:
        t = 2 if (i % 6) == 0 else 1
        n = file.stem.split('_')
        it = n[1]
        ii = n[0]
        if not it == imgname:
            continue
        iid = import_image(file, 'None', 2380, 1281, engine, t)
        f2 = p.joinpath(ii + '_' + labelname + '_' + labeltypename + '.png')
        import_label_image_sub(f2, iid, t, engine)
        i += 1


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


def image_to_labels(image):
    image = np.dot(image, np.array([65536, 256, 1], dtype='int64'))
    return color_map[image]


def labels_to_image(labels):
    img = limap[labels, :]
    return img


def output_labels_to_image(labels):
    labels = labels.numpy().transpose((1, 2, 0))
    labels = np.argmax(labels, axis=2)
    image = labels_to_image(labels)
    return image


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


if __name__ == '__main__':
    p = Path('/mnt/disks/data/segment')
    engine = sa.create_engine('sqlite:///data.db')
    import_car_images2(p, engine)
    # build_labels(engine)
    # engine = sa.create_engine('sqlite:///data2.db')
    # p1 = Path('/mnt/disks/data/cityscapes/leftImg8bit/')
    # p2 = Path('/mnt/disks/data/cityscapes/gtFine/')
    # p3 = Path('/mnt/disks/data/cityscapes/labelimgs/')
    # import_images(p1, p2, p3, engine)
    print('FINISHED')
