import click
import rospy
import message_filters
from visualization_msgs.msg import *
from std_msgs.msg import String
from rospy_tutorials.msg import HeaderString
from geometry_msgs.msg import Polygon, PolygonStamped, Point32
import time

pub = rospy.Publisher('polygon_node', PolygonStamped, queue_size=10)

msg_poly = PolygonStamped()
poly = Polygon()

def callback(marker):
	key = click.getchar()
	if key=='a':
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

def listener():
	rospy.init_node('listener',anonymous=True)
	marker_sub = message_filters.Subscriber("measure_point", Marker)
	#button_sub = message_filters.Subscriber("button", HeaderString)
	ts = message_filters.TimeSynchronizer([marker_sub],10)
	ts.registerCallback(callback)
	rospy.spin()


if __name__ == '__main__':
	print('press a to capture point')
	listener()

	
