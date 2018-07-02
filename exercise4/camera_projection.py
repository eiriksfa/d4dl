import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import math

intrinsics = np.array([[585.2304272868565, 0.0, 1190.0, 0], [0.0, 585.2304272868565, 640.5, 0], [0.0, 0.0, 1.0, 0.0]])

# transform = np.array([[-0.11350664, -0.04971457, -0.99229265, 0.16266196],
#                       [0.99340159, -0.02218241, -0.11252213, -1.61062595],
#                       [-0.01641745, -0.9985171, 0.05190438, 0.20266677]])

# Rotation matrix (cars rotation in relation to world coordinates)
rot = np.array([[-9.88373548e-01, -1.12874824e-03, 1.52040968e-01],
                      [-1.52045131e-01, 7.93392204e-03, -9.88341708e-01],
                      [-9.06922267e-05, -9.99967889e-01, -8.01329935e-03]])

# Translation matrix (cars position)
translation = np.array([[3.55766200e+00], [-1.10985354e+00], [1.72194294e-01]])

P = -rot @ translation
P = np.concatenate((rot, P), axis=1)

rot = np.append(rot, [[0, 0, 0]], axis=0)
translation = np.append(translation, [[1]], axis=0)

transform = np.concatenate((rot, translation), axis=1)
extrinsic = np.linalg.inv(transform)

# Mapping should be extrinsics @ intrinsics @ coordinates


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis.
    (From: https://github.com/ros/geometry/blob/hydro-devel/tf/src/tf/transformations.py)
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.
    (From: https://github.com/ros/geometry/blob/hydro-devel/tf/src/tf/transformations.py)
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0, 0.0),
                  (0.0, cosa, 0.0),
                  (0.0, 0.0, cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(((0.0, -direction[2], direction[1]),
                   (direction[2], 0.0, -direction[0]),
                   (-direction[1], direction[0], 0.0)),
                  dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.
    (From: https://github.com/ros/geometry/blob/hydro-devel/tf/src/tf/transformations.py)
    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def load_polygons(path):
    p = Path(path)
    polygons = []
    for csv in [f for f in p.iterdir()]:
        df = pd.read_csv(csv)
        polygons.append(df)
    return polygons


def process_image(img, polygons, intr, extrinsic):
    # transform = intr @ extr
    print(p)
    out = img.copy()
    coords = np.array([[3.80983965293], [-4.259930015], [-0.032020637], [1]])
    coords2 = np.array([[3.789993012], [-2.325306419], [-0.034404161], [1]])
    coords3 = np.array([[3.370829342], [-2.317988317], [-0.023907091], [1]])
    coords4 = np.array([[3.384244631], [-4.267839467], [-0.016064657], [1]])
    trans = intrinsics @ extrinsic
    print('========')
    print(coords)
    c = p @ coords
    coords = trans @ coords
    print(c)
    coords2 = trans @ coords2
    coords3 = trans @ coords3
    coords4 = trans @ coords4
    print('========')
    coords = [[(coords[0][0]) / (coords[2][0]), (coords[1][0]) / (coords[2][0])]]
    coords = np.round(coords).astype(np.int32)
    coords2 = [[(coords2[0][0]) / (coords2[2][0]), (coords2[1][0]) / (coords2[2][0])]]
    coords2 = np.round(coords2).astype(np.int32)
    coords3 = [[(coords3[0][0]) / (coords3[2][0]), (coords3[1][0]) / (coords3[2][0])]]
    coords3 = np.round(coords3).astype(np.int32)
    coords4 = [[(coords4[0][0]) / (coords4[2][0]), (coords4[1][0]) / (coords4[2][0])]]
    coords4 = np.round(coords4).astype(np.int32)
    print(coords)
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
    t1 = (coords[0][0], coords[0][1])
    t2 = (coords2[0][0], coords2[0][1])
    t3 = (coords3[0][0], coords3[0][1])
    t4 = (coords4[0][0], coords4[0][1])
    print(t1)
    projected = np.array([t1, t2, t3, t4])
    cv2.fillPoly(img, [projected], (255, 0, 0))
    cv2.addWeighted(img, 0.7, out, 0.3, 0, out)
    return out


def process_csv(csv, polygons, intr):
    df = pd.read_csv(csv)


def main():
    polygons = load_polygons('polygons')
    image = cv2.imread('images/car02-frame000175.png')
    # -0.687271983, 0.014578437, 0.005606003, 0.726232238
    img = process_image(image, polygons, intrinsics, extrinsic)
    cv2.imshow('img', img)
    cv2.imwrite('test.jpg', img)


if __name__ == '__main__':
    main()
