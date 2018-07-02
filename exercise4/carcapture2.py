import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
import tf
from tf import ExtrapolationException, ConnectivityException
import time
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import os
import pandas as pd
import sys
import math
import signal
import image_geometry
from pathlib import Path
from scipy.misc import imsave


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
    length = np.atleast_1d(np.sum(data * data, axis))
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
        sina = (R[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def load_polygons(path):
    p = Path(path)
    polygons = []
    for csv in [f for f in p.iterdir()]:
        df = pd.read_csv(csv)
        polygons.append(df)
    return polygons


def build_matrices(rot, trans):
    P = np.matmul(-rot, trans)
    P = np.concatenate((rot, P), axis=1)
    rot = np.append(rot, [[0, 0, 0]], axis=0)
    trans = np.append(trans, [[1]], axis=0)

    transform = np.concatenate((rot, trans), axis=1)
    extrinsic = np.linalg.inv(transform)
    return P, extrinsic


def point_inside_polygon(x, y, poly):
    """
    # determine if a point is inside a given polygon or not
    # Polygon is a list of [x,y] pairs.
    From http://www.ariel.com.au/a/python-point-int-poly.html
    :param x:
    :param y:
    :param poly:
    :return:
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0][0], poly[0][1]
    for i in range(n + 1):
        p2x, p2y = poly[i % n][0], poly[i % n][1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class CameraPose(object):

    def __init__(self):
        self.tf_listener = tf.TransformListener()
        self.num = 0
        self.broken_amt = 0
        self.imgseq = []
        self.timestamp = []
        self.broken = []
        self.mat = []
        self.prop = None
        self.polygon = load_polygons('/home/novian/term2/dl4ad/repo2/d4dl/exercise4/polygons')
        self.counter = 0
        self.intrinsics = np.array(
            [[585.2304272868565, 0.0, 1190.0, 0], [0.0, 585.2304272868565, 640.5, 0], [0.0, 0.0, 1.0, 0.0]])

    def callback_camerainfo(self, prop):
        self.prop = prop

    def _build_polygons(self, P, extrinsics):
        trans = np.matmul(self.intrinsics, extrinsics)
        polygons = []
        for polygon in self.polygon:
            projected = []
            # below can probably be vectorized somehow, maybe consider using bboxes as well
            for i, r in polygon.iterrows():
                coords = np.array([[r.x], [r.y], [r.z], [1]])
                c = np.matmul(P, coords)
                if c[2] > 0:  # Front of camera, TODO: Clipping
                    coords = np.matmul(trans, coords)
                    coords = [[(coords[0][0]) / (coords[2][0]), (coords[1][0]) / (coords[2][0])]]
                    coords = np.round(coords).astype(np.int32)
                    projected.append(coords)
            if len(projected) > 2:
                projected = np.array(projected)
                polygons.append(projected)
        return polygons

    def _build_label_image(self, shape, polygons, path):
        img = np.zeros(shape)
        for r in img:
            for c in r:
                for p in polygons:
                    if point_inside_polygon(c, r, p):
                        img[r][c] = 1
        namefile = str(self.counter) + '_label_out.png'
        imsave(os.path.join(path, namefile), img)

    def _build_image(self, polygons, img, path):
        out = img.copy()
        for p in polygons:
            cv2.fillPoly(img, [p], (0, 0, 255))
        cv2.addWeighted(img, 0.7, out, 0.3, 0, out)
        namefile = str(self.counter) + '_out.png'
        cv2.imwrite(os.path.join(path, namefile), out)
        return out

    def callback_camera(self, img):
        namefile = '{}{:06d}{}'.format('car02-frame', self.num, '.png')
        broken = False
        try:
            print('got image')
            self.num = self.num + 1
            np_arr = np.fromstring(img.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)
            broken = True
        else:
            print("saving " + namefile)

        mtr = []
        trans = [0.0, 0.0, 0.0]
        rot = [0.0, 0.0, 0.0, 0.0]

        try:
            (trans, rot) = self.tf_listener.lookupTransform('/mocap', 'front_cam', img.header.stamp)
        except ExtrapolationException as e:
            print(e)
            broken = True
            self.broken_amt += 1
        except ConnectivityException as e:
            print(e)
            broken = True
            self.broken_amt += 1

        if not broken:
            # TODO: rot = rotation_matrix(angle, rotation...) FIX!
            rot = np.array(
                rot)  # TODO: Check that it is a numpy array of form 3x3, f'ex: np.array([[-9.88373548e-01, -1.12874824e-03, 1.52040968e-01], [-1.52045131e-01, 7.93392204e-03, -9.88341708e-01], [-9.06922267e-05, -9.99967889e-01, -8.01329935e-03]])
            trans = np.array(
                trans)  # TODO: Numpy array of 3x1: np.array([[3.55766200e+00], [-1.10985354e+00], [1.72194294e-01]])
            P, extrinsics = build_matrices(rot, trans)
            polygons = self._build_polygons(P, extrinsics)
            self.counter += 1
            img = self._build_image(polygons, image_np, '/home/novian/catkin_ws/src/bagfile/car-02n/')
            self._build_label_image((img.shape[0], img.shape[1]), polygons,
                                    '/home/novian/catkin_ws/src/bagfile/car-02n/')
        self.mat.append(mtr)
        self.imgseq.append(namefile)
        self.timestamp.append(img.header.stamp)
        self.broken.append(broken)

    def hook(self):
        print('shutdown')
        rawdata = {
            'imgseq': self.imgseq,
            'timestamp': self.timestamp,
            'broken': self.broken,
            'mat': self.mat}
        df = pd.DataFrame(rawdata, columns=['imgseq', 'timestamp', 'broken', 'mat'])

    # df.to_csv('car-01.csv')

    def listener(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("output/camera_info", CameraInfo, self.callback_camerainfo)
        rospy.Subscriber("output/image/compressed", CompressedImage, self.callback_camera)
        rospy.on_shutdown(self.hook)
        rospy.spin()


if __name__ == '__main__':
    cp = CameraPose()
    cp.listener()
