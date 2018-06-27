import rospy
from visualization_msgs.msg import *
from std_msgs.msg import String
from rospy_tutorials.msg import HeaderString
from geometry_msgs.msg import Polygon, PolygonStamped, Point32
from keyboard.msg import Key
import time
import poly_util as pu
import pandas as pd

pub = rospy.Publisher('polygon_node', PolygonStamped, queue_size=10)

class PolyCap(object):

    def __init__(self):
        self.npoint = False
        self.save = False
        self.new = False
        self.polygon = Polygon()
        self.msg_poly = PolygonStamped()
        self.count = 0

    def callback_marker(self,marker):
        if self.npoint:
            self.npoint=False
            point = Point32()

            self.msg_poly.header.stamp = rospy.Time.now()
            self.msg_poly.header.frame_id = "/mocap"
            point.x = marker.pose.position.x
            point.y = marker.pose.position.y
            point.z = marker.pose.position.z
            print('appending point')
            self.polygon.points.append(point)
            self.msg_poly.polygon = self.polygon
        elif self.new:
            self.new = False
            self.polygon = Polygon()
            self.msg_poly = PolygonStamped()
            self.count += 1
            print('add new polygon')
        elif self.save:
            self.save = False
            df = pu.ros_to_pd(self.polygon)
            df = pu.interpolate(df, 0.1)
            pu.save(df, 'polygons/poly_' + str(self.count) + '.csv')
            self.new = True
            print('ploygon saved')

        pub.publish(self.msg_poly)

    def callback_button(self,data):
        self.pressed = data.code
        print('pressed')
        if data.code==32:
            self.npoint = True
        elif data.code==115:
            self.save = True
        elif data.code==110:
            self.new = True

    def listener(self):
        rospy.init_node('listener',anonymous=True)
        rospy.Subscriber("wand", Marker, self.callback_marker)
        rospy.Subscriber("keyboard/keyup", Key, self.callback_button)
        rospy.spin()

if __name__ == '__main__':
    pc = PolyCap()
    print('Press space to capture point')
    pc.listener()



