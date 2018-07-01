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
import signal
import image_geometry
from pathlib import Path


# rosrun image_transport republish compreed in:=output/image raw out:=camera_out/image

intrinsics = np.array([[585.2304272868565, 0.0, 1190.0], [0.0, 585.2304272868565, 640.5], [0.0, 0.0, 1.0]])


def load_polygons(path='/home/novian/term2/dl4ad/repo2/d4dl/exercise4/polygons'):
    p = Path(path)
    polygons = []
    for csv in [f for f in p.iterdir()]:
        df = pd.read_csv(csv)
        polygons.append(df)
    return polygons


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
        self.polygon = load_polygons()
        self.counter = 0

    def callback_camerainfo(self, prop):
        # print('got image')
        print(prop.header.stamp)
        print(rospy.Time.now())
        self.prop = prop

    def callback_camera(self, img):
        namefile = '{}{:06d}{}'.format('car02-frame', self.num, '.png')
        try:
            print('got image')
            self.num = self.num + 1
            np_arr = np.fromstring(img.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)
        else:
            # cv2.imshow("window",image_np)
            # cv2.imwrite('test.jpg',image_np)
            # print("saving car01-frame%06i.png" % self.num)
            print("saving " + namefile)
        # cv2.imwrite(os.path.join('/home/novian/catkin_ws/src/bagfile/car-02/', namefile), image_np)
        mtr = []
        trans = [0.0, 0.0, 0.0]
        rot = [0.0, 0.0, 0.0, 0.0]
        broken = False
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
        else:
            mtr = self.tf_listener.fromTranslationRotation(trans, rot)

        print(mtr)
        # TODO: Convert mtr to a numpy matrix of form:
        # 4x3 (drop last row, it should be [0, 0, 0, 1] and not necessary)
        # np.array(mtr)...

        ig = image_geometry.PinholeCameraModel()
        ig.fromCameraInfo(self.prop)

        out = image_np.copy()
        # for polygon in self.polygon:
        #     projected = []
        #     # below can probably be vectorized somehow, maybe consider using bboxes as well
        #     for i, r in polygon.iterrows():
        #         coords = np.array([[r.x], [r.y], [r.z], [1]])
        #         projected.append(coords)
        #         coords = np.matmul(mtr, coords)
        #         coord = np.matmul(intrinsics, coords)
        #         # TODO: Try to get the below to work, check input type (numpy array, normal array?, shape etc), check if it gives the same coord as above..
        #         #coord = ig.project3dToPixel(coords)
        #         print(coord)
        #     projected = np.array(projected)
        #     cv2.fillPoly(image_np, [projected], (0, 255, 0))
        # cv2.addWeighted(image_np, 0.7, out, 0.3, 0, out)

        coords = np.array([[3.805], [-2.333], [-0.027], [1]])
        coords = np.matmul(mtr, coords)
        coord = np.matmul(intrinsics, coords)
        coord = [(coord[0][0]) / (coord[2][0]), (coord[1][0]) / (coord[2][0])]
        coord = np.round(coord).astype(np.int32)
        # TODO: Try to get the below to work, check input type (numpy array, normal array?, shape etc), check if it gives the same coord as above..
        # (TODO: One function below should replace the three lines above, but it probably needs the original mtr, not the numpy array one with reduced dimensions)
        #coord = ig.project3dToPixel(coords)
        # TODO: Test around with the array below, see if you can get it to make sense?
        cv2.circle(out, (coord[0], out.shape[0]-coord[1]), 10, (255, 0, 0), thickness=10)
        cv2.circle(out, (coord[0], coord[1]), 10, (0, 255, 0), thickness=10)
        print(coord)
        namefile = str(self.counter) + '_out.png'
        self.counter += 1
        cv2.imwrite(os.path.join('/home/novian/catkin_ws/src/bagfile/car-02n/', namefile), out)

        self.mat.append(mtr)
        self.imgseq.append(namefile)
        self.timestamp.append(img.header.stamp)
        self.broken.append(broken)
        print(trans)
        print(rot)
        # print(self.mat)

        print(self.broken_amt)

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
