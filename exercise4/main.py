#! /usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

publisher = None


def publish(data):
    marker = Marker()
    marker.header.frame_id = "/mocap"
    marker.header.stamp = rospy.get_rostime()
    marker.ns = 'measure_p'
    marker.id = 0
    marker.type = marker.CUBE
    marker.action = marker.ADD

    marker.pose.position.x = data[0]
    marker.pose.position.y = data[1]
    marker.pose.position.z = data[2]
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05

    marker.color.g = 1
    marker.color.a = 1
    publisher.publish(marker)


def callback(data):
    marker = data.markers[0]
    vectors = []
    for p in marker.points:
        v = np.array([p.x, p.y, p.z])
        if np.any(np.isnan(v)):
            rospy.loginfo('Found nan, returning')
            return
        vectors.append(v)
    direction = vectors[1] - vectors[0]
    mv = vectors[1] + direction * 0.408
    publish(mv)


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('cortex_marker_array', MarkerArray, callback)
    global publisher
    publisher = rospy.Publisher('wand', Marker, queue_size=10)
    rospy.spin()


if __name__ == '__main__':
    listener()
