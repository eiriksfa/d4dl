import numpy as np
import pandas as pd
from pathlib import Path
import cv2

intrinsics = np.array([[585.2304272868565, 0.0, 1190.0], [0.0, 585.2304272868565, 640.5], [0.0, 0.0, 1.0]])

transform = np.array([[-0.11350664, -0.04971457, -0.99229265, 0.16266196],
                      [0.99340159, -0.02218241, -0.11252213, -1.61062595],
                      [-0.01641745, -0.9985171, 0.05190438, 0.20266677]])

transform = np.array([[-9.88373548e-01, -1.12874824e-03, 1.52040968e-01, 3.55766200e+00],
                        [-1.52045131e-01, 7.93392204e-03, -9.88341708e-01, -1.10985354e+00],
                        [-9.06922267e-05, -9.99967889e-01, -8.01329935e-03, 1.72194294e-01]])


def load_polygons(path):
    p = Path(path)
    polygons = []
    for csv in [f for f in p.iterdir()]:
        df = pd.read_csv(csv)
        polygons.append(df)
    return polygons


def process_image(img, polygons, intr, transform):
    # transform = intr @ extr
    print(transform)
    out = img.copy()
    coords = np.array([[3.805], [-2.333], [-0.027], [1]])
    print(coords)
    coords1 = transform @ coords
    print('========')
    print(coords1)
    coords11 = intrinsics @ coords1
    print('========')
    coords11 = [[(coords11[0][0])/(coords11[2][0]), (coords11[1][0])/(coords11[2][0])]]
    coords11 = np.round(coords11).astype(np.int32)
    print(coords11)
    # for polygon in polygons:
    #     projected = []
    #     # below can probably be vectorized somehow, maybe consider using bboxes as well
    #     for i, r in polygon.iterrows():
    #         coords = np.array([[r.x], [r.y], [r.z], [1]])
    #         coords = transform @ coords
    #         print(coords)
    #         coords = intrinsics @ coords
    #         print(coords)
    #         coords = [[(coords[0][0])/(coords[2][0]), (coords[1][0])/(coords[2][0])]]  #
    #         coords = np.round(coords).astype(np.int32)
    #         print(coords)
    #         projected.append(coords)
    #     projected = np.array(projected)
    #     cv2.fillPoly(img, [projected], (0, 255, 0))
    # cv2.addWeighted(img, 0.7, out, 0.3, 0, out)
    print(out.shape)
    # cv2.circle(out, (coords11[0][0], 1281-(coords11[0][1])), 40, (255, 0, 0), thickness=7)
    test = (coords11[0][0], coords11[0][1])
    print(test)
    cv2.circle(out, (1142, 750), 20, (255, 0, 0), thickness=10)
    return out


def process_csv(csv, polygons, intr):
    df = pd.read_csv(csv)


def main():
    polygons = load_polygons('polygons')
    image = cv2.imread('images/car02-frame000175.png')
    # -0.687271983, 0.014578437, 0.005606003, 0.726232238
    img = process_image(image, polygons, intrinsics, transform)
    cv2.imshow('img', img)
    cv2.imwrite('test.jpg', img)


if __name__ == '__main__':
    main()
