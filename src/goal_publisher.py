# rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped '{header: {stamp: now, frame_id: "map"}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'


#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped

def goal_publisher():
    # Initialize the ROS node
    rospy.init_node('goal_publisher', anonymous=True)

    # Create a publisher with the topic '/goal_pose' and message type 'PoseStamped'
    goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

    # Set the publishing rate in Hz
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        # Create a PoseStamped message object
        goal_msg = PoseStamped()

        # Set the header information
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = 'map'  # Replace 'map' with your desired frame ID

        # Set the pose information
        goal_msg.pose.position.x = 1.0  # x-coordinate
        goal_msg.pose.position.y = 2.0  # y-coordinate
        goal_msg.pose.position.z = 0.0  # z-coordinate

        goal_msg.pose.orientation.x = 0.0  # x-orientation
        goal_msg.pose.orientation.y = 0.0  # y-orientation
        goal_msg.pose.orientation.z = 0.0  # z-orientation
        goal_msg.pose.orientation.w = 1.0  # w-orientation

        # Publish the goal pose
        goal_pub.publish(goal_msg)

        # Sleep for the specified time to achieve the desired publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        goal_publisher()
    except rospy.ROSInterruptException:
        pass

