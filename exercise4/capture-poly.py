import rospy
from visualization_msgs.msg import *
from std_msgs.msg import String
from rospy_tutorials.msg import HeaderString
from geometry_msgs.msg import Polygon, PolygonStamped, Point32
from keyboard.msg import Key
import time

pub = rospy.Publisher('polygon_node', PolygonStamped, queue_size=10)
msg_poly = PolygonStamped()
poly = Polygon()

class PolyCap(object):

    def __init__(self):
        self.pressed = None

    def callback_marker(self,marker):
        if self.pressed==True:
            print(marker.pose.position)
            self.pressed=False

            print(marker.pose.position)
            point = Point32()

            msg_poly.header.stamp = rospy.Time.now()
            msg_poly.header.frame_id = "/mocap"
            point.x = marker.pose.position.x
            point.y = marker.pose.position.y
            point.z = marker.pose.position.z

            poly.points.append(point)
            msg_poly.polygon = poly

            print(msg_poly)

        pub.publish(msg_poly)

    def callback_button(self,data):
        self.pressed = data.code
        print('pressed')
        if data.code==32:
            self.pressed=True

    def listener(self):
        rospy.init_node('listener',anonymous=True)
        rospy.Subscriber("wand", Marker, self.callback_marker)
        rospy.Subscriber("keyboard/keyup", Key, self.callback_button)
        rospy.spin()

if __name__ == '__main__':
    pc = PolyCap()
    print('Press space to capture point')
    pc.listener()



