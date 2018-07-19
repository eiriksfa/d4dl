import rospy
from geometry_msgs.msg import TransformStamped, Point
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
import tf
from tf import ExtrapolationException, ConnectivityException
import time
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import skimage.draw as draw
import cv2
import os
import pandas as pd
import sys
import math
import signal
import image_geometry
from pathlib import Path
from scipy.misc import imsave
import matplotlib.path as mplPath


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
    rot = np.append(rot, [[0, 0, 0]], axis=0)
    trans = np.append(trans, [[1]], axis=0)

    transform = np.concatenate((rot, trans), axis=1)
    extrinsic = np.linalg.inv(transform)
    return extrinsic


class CameraPose(object):

    def __init__(self):
        self.tf_listener = tf.TransformListener()
        self.img_pub = rospy.Publisher('output_2/image/compressed', CompressedImage, queue_size=10)
        self.point_pub = rospy.Publisher('polygon_point', Marker, queue_size=10)
        self.num = 0
        self.broken_amt = 0
        self.imgseq = []
        self.timestamp = []
        self.broken = []
        self.mat = []
        self.prop = None
        self.polygon = load_polygons('/home/novian/term2/dl4ad/repo2/d4dl/exercise4/polygons')
        self.counter = 0
        self.poly_point_msg = None
        self.intrinsics = np.array(
            [[585.2304272868565, 0.0, 1190.0, 0], [0.0, 585.2304272868565, 640.5, 0], [0.0, 0.0, 1.0, 0.0]])

    def callback_camerainfo(self, prop):
        self.prop = prop

    def init_marker(self):
        marker = Marker()
        marker.header.frame_id = "/mocap"
        marker.header.stamp = rospy.get_rostime()
        marker.ns = 'poly_point'
        marker.id = 0
        marker.type = marker.POINTS
        marker.action = marker.ADD

        #marker.pose.position.x = data[0]
        #marker.pose.position.y = data[1]
        #marker.pose.position.z = data[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.scale.z = 0.02

        marker.color.g = 1
        marker.color.a = 1

        return marker


    def add_point_marker(self, marker, cx, cy):
        point = Point()
        point.x = cx
        point.y = cy
        point.z = 0
        marker.points.append(point)

    def _build_polygons(self, extrinsics):
        polygons = []
        lp = []
        lane = []
        for polygon in self.polygon:
            projected = []
            cp = []
            rp = []
            for i, r in polygon.iterrows():
                coords = np.array([[r.x], [r.y], [r.z], [1]])
                coords = np.matmul(extrinsics, coords)
                #self.add_point_marker(self.poly_point_msg, r.x, r.y)
                if coords[2][0] > 0.00001:
                    self.add_point_marker(self.poly_point_msg, r.x, r.y)
                    coords = np.matmul(self.intrinsics, coords)
                    coords = [(coords[0][0]) / (coords[2][0]), (coords[1][0]) / (coords[2][0])]
                    coords = np.round(coords).astype(np.int32)
                    projected.append(coords)
                    rp.append(coords[0]-1)
                    cp.append(coords[1]-1)
            if len(projected) > 2:
                projected = np.array(projected)
                polygons.append(projected)
                lp.append((rp, cp))

                lane_class = 0 # default : other lane
                poly_forcheck = mplPath.Path(projected)
                lb = poly_forcheck.contains_point((0,1280)) # left bottom
                cb = poly_forcheck.contains_point((1300,1280)) # (almost) center bottom
                if lb and cb :
                    # wide polygon = intersection
                    lane_class = 2
                elif not lb and cb :
                    # only in center = our lane
                    lane_class = 1
                lane.append(lane_class)
        return polygons, lane

    def _build_label_image(self, shape, polygons, path):
        img = np.zeros(shape)
        for p in polygons:
            r = p[0]
            c = p[1]
            rr, cc = draw.polygon(r, c)
            img[rr, cc] = 1
        namefile = str(self.counter) + '_label_out.png'
        imsave(os.path.join(path, namefile), img)

    def _build_image(self, polygons, lane, img, path):
        out = img.copy()
        label_1c = np.zeros(out.shape, np.uint8)
        label_3c = np.zeros(out.shape, np.uint8)
        for idx in range(len(polygons)):
            # three-class labels
            if lane[idx]==0:
                # other lane, red
                cv2.fillPoly(img, [polygons[idx]], (0, 0, 255))
                cv2.fillPoly(label_3c, [polygons[idx]], (0, 0, 255))
            elif lane[idx]==1:
                # our lane, blue
                cv2.fillPoly(img, [polygons[idx]], (255, 0, 0))
                cv2.fillPoly(label_3c, [polygons[idx]], (255, 0, 0))
            else:
                #intersection, green
                cv2.fillPoly(img, [polygons[idx]], (0, 255, 0))
                cv2.fillPoly(label_3c, [polygons[idx]], (0, 255, 0))

            # one-class label
            cv2.fillPoly(label_1c, [polygons[idx]], (0, 255, 0))

        cv2.addWeighted(img, 0.7, out, 0.3, 0, out)
        namefile = str(self.counter) + '_out.png'
        labename_1c = str(self.counter) + '_label_1c.png'
        labename_3c = str(self.counter) + '_label_3c.png'
        #cv2.imwrite(os.path.join(path, namefile), out)
        #cv2.imwrite(os.path.join(path, labename_1c), label_1c)
        #cv2.imwrite(os.path.join(path, labename_3c), label_3c)
        # Need to convert to labelimage, and not just ground truth (function already implemented in utility.py)
        print self.counter
        return out, label_1c, label_3c

    def callback_camera(self, img):
        #namefile = '{}{:06d}{}'.format('car02-frame', self.num, '.png')
        namefile=""
        broken = False
        try:
            #print('got image')
            self.num = self.num + 1
            np_arr = np.fromstring(img.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            #print(e)
            broken = True
        else:
            namefile = '{:03d}{}'.format(self.num, '_out.png')
            cv2.imwrite(os.path.join('/home/novian/catkin_ws/src/bagfile/raw-234/', namefile), image_np)
            #print("saving " + namefile)

        mtr = []
        trans = [0.0, 0.0, 0.0]
        rot = [0.0, 0.0, 0.0, 0.0]

        try:
            (trans, rot) = self.tf_listener.lookupTransform('/mocap', 'front_cam', img.header.stamp)
            mtr = self.tf_listener.fromTranslationRotation(trans, rot)
        except (ExtrapolationException, ConnectivityException) as e:
            #print(e)
            broken = True
            self.broken_amt += 1

        if not broken:
            rot_m = mtr[0:3, 0:3]
            trans_v = mtr[0:3, 3:4]

            extrinsics = build_matrices(rot_m, trans_v)
            #print("matrices for "+namefile+" built")
            polygons, lane = self._build_polygons(extrinsics)
            #print(len(polygons))
            #print("polygon for "+namefile+" built")
            self.counter += 1
            (image_raw, label_1c, label_3c) = self._build_image(polygons, lane, image_np, '/home/novian/catkin_ws/src/bagfile/car-new-01')
            #print("image for "+namefile+" built")

            #publish the image to rviz
            self.publish_image(self.img_pub, image_raw)


            #publish seen points of polygons to rviz
            self.point_pub.publish(self.poly_point_msg)

        self.poly_point_msg.points = []
        #self.mat.append(mtr)
        self.imgseq.append(namefile)
        self.timestamp.append(img.header.stamp)
        self.broken.append(broken)
        #print("done for "+namefile)

    def publish_image(self,publisher,image):
        image_msg = CompressedImage()
        image_msg.header.stamp = rospy.Time.now()
        image_msg.format = "jpeg"
        image_msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
        publisher.publish(image_msg)

    def hook(self):
        print('shutdown')
        rawdata = {
            'imgseq': self.imgseq,
            'timestamp': self.timestamp,
            'broken': self.broken}
        df = pd.DataFrame(rawdata, columns=['imgseq', 'timestamp', 'broken'])
        print("broken : " + str(self.broken_amt))
        print("created : " + str(self.counter))
        df.to_csv('/home/novian/catkin_ws/src/bagfile/car-234.csv')

    def listener(self):
        rospy.init_node('listener', anonymous=True)
        self.poly_point_msg = self.init_marker()
        rospy.Subscriber("output/camera_info", CameraInfo, self.callback_camerainfo)
        rospy.Subscriber("output/image/compressed", CompressedImage, self.callback_camera)
        rospy.on_shutdown(self.hook)
        rospy.spin()


if __name__ == '__main__':
    cp = CameraPose()
    print("we are ready")
    cp.listener()
