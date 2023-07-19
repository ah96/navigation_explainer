#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

def publish_global_plan():
    # Initialize the ROS node
    rospy.init_node('global_plan_publisher')

    # Create a publisher for the global plan
    pub = rospy.Publisher('move_base/GlobalPlanner/plan', Path, queue_size=10)
    # move_base/NavfnROS/plan; move_base/GlobalPlanner/plan;

    # Create a Path message
    path = Path()

    # Add poses to the path
    pose1 = PoseStamped()
    pose1.pose.position.x = 1.0
    pose1.pose.position.y = 2.0
    pose1.pose.orientation.w = 1.0
    path.poses.append(pose1)

    pose2 = PoseStamped()
    pose2.pose.position.x = 3.0
    pose2.pose.position.y = 4.0
    pose2.pose.orientation.w = 1.0
    path.poses.append(pose2)

    pose3 = PoseStamped()
    pose3.pose.position.x = 5.0
    pose3.pose.position.y = 6.0
    pose3.pose.orientation.w = 1.0
    path.poses.append(pose3)

    pose4 = PoseStamped()
    pose4.pose.position.x = 7.0
    pose4.pose.position.y = 8.0
    pose4.pose.orientation.w = 1.0
    path.poses.append(pose4)

    pose5 = PoseStamped()
    pose5.pose.position.x = 9.0
    pose5.pose.position.y = 10.0
    pose5.pose.orientation.w = 1.0
    path.poses.append(pose5)

    pose6 = PoseStamped()
    pose6.pose.position.x = 9.0
    pose6.pose.position.y = 10.0
    pose6.pose.orientation.w = 1.0
    path.poses.append(pose6)

    # Set the header information for the path
    path.header.frame_id = 'map'
    path.header.stamp = rospy.Time.now()

    rate = rospy.Rate(10)  # Publish at 10 Hz

    while not rospy.is_shutdown():
    # Publish the global plan
    #for i in range(0, 10):
        pub.publish(path)
        print('published!')
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_global_plan()
    except rospy.ROSInterruptException:
        pass
