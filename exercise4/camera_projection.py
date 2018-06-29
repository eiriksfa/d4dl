import numpy as np
import pandas as pd
from pathlib import Path
import cv2


intrinsics2 = np.array([
    [585.5644470303048, 0, 630.5897331003082],
    [0, 584.8964075434083, 477.7496476463654],
    [0, 0, 1]])

intrinsics = np.array([[585.2304272868565, 0.0, 1190.0], [0.0, 585.2304272868565, 640.5], [0.0, 0.0, 1.0]])

transform = np.array([[-0.11350664, -0.04971457, -0.99229265, 0.16266196],
                      [0.99340159, -0.02218241, -0.11252213, -1.61062595],
                      [-0.01641745, -0.9985171, 0.05190438, 0.20266677]])


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
    for polygon in polygons:
        projected = []
        # below can probably be vectorized somehow, maybe consider using bboxes as well
        for i, r in polygon.iterrows():
            coords = np.array([[0], [0], [0], [1]])# np.array([[r.x], [r.y], [r.z], [1]])
            coords = transform @ coords
            print(coords)
            coords = intrinsics @ coords
            print(coords)
            coords = [[(coords[0][0])/(coords[2][0]), (coords[1][0])/(coords[2][0])]]  #
            coords = np.round(coords).astype(np.int32)
            print(coords)
            projected.append(coords)
        projected = np.array(projected)
        cv2.fillPoly(img, [projected], (0, 255, 0))
    cv2.addWeighted(img, 0.7, out, 0.3, 0, out)
    return out


def process_csv(csv, polygons, intr):
    df = pd.read_csv(csv)


def main():
    polygons = load_polygons('polygons')
    image = cv2.imread('images/car02-frame000258.png')
    # -0.687271983, 0.014578437, 0.005606003, 0.726232238
    img = process_image(image, polygons, intrinsics, transform)
    print(img.shape)
    cv2.imshow('img', img)
    cv2.imwrite('test.jpg', img)

if __name__ == '__main__':
    main()
