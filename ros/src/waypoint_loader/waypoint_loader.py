#!/usr/bin/env python

import os
import csv
import math

from geometry_msgs.msg import Quaternion

from styx_msgs.msg import Lane, Waypoint

import tf
import rospy

# waypoints csv has the following structure
CSV_HEADER = ['x', 'y', 'z', 'yaw']

# this constant is used in calculation of decreasing speeds of last waypoints
# so far it is not clear what that means; the formular doesn't depend on time, so it is hard to say what this is all
# about
MAX_DECEL = 1.0


class WaypointLoader(object):
    """
    Loads trajectory waypoints and publishes them all as /base_waypoints topic.
    Waypoints have assigned default velocities starting to decrease at some point so the last waypoint velocity is 0.

    """
    def __init__(self):
        # init node
        rospy.init_node('waypoint_loader', log_level=rospy.DEBUG)

        # publish the topic, latch means that last published waypoints are memorized
        # and will be received by newly connected subscriber
        self.pub = rospy.Publisher('/base_waypoints', Lane, queue_size=1, latch=True)

        # default velocity in kilometers per hour
        self.velocity = rospy.get_param('~velocity')

        # load waypoints and publish them
        self.new_waypoint_loader(rospy.get_param('~path'))

        # keep this node from exiting until it is stopped
        rospy.spin()

    def new_waypoint_loader(self, path):
        """
        Loads waypoints from file and publishes them to /base_waypoints in /world frame
        :param path:
        :return:
        """
        if os.path.isfile(path):
            waypoints = self.load_waypoints(path)
            self.publish(waypoints)
            rospy.loginfo('Waypoint Loded')
        else:
            rospy.logerr('%s is not a file', path)

    def quaternion_from_yaw(self, yaw):
        """
        Converts yaw to quaternion.
        :param yaw:
        :return:
        """
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def convert_kph2mps(self, velocity):
        """
        Converts velocity from kilometers per hour to meters per second
        :param velocity: velocity in kilometers per hour
        :return:
        """
        return velocity/3.6

    def load_waypoints(self, fname):
        """
        Loads waypoints from file. Assigns each waypoint a linear velocity so most of the time the velocities are constant,
        but at the end waypoints have decreasing velocity so the last velocity is 0.
        :param fname: file name of waypoints file
        :return: list of waypoints
        """
        waypoints = []
        with open(fname) as wfile:
            reader = csv.DictReader(wfile, CSV_HEADER)
            # read all waypoints
            for wp in reader:
                p = Waypoint()
                p.pose.header.frame_id = "/world"
                p.pose.pose.position.x = float(wp['x'])
                p.pose.pose.position.y = float(wp['y'])
                p.pose.pose.position.z = float(wp['z'])
                q = self.quaternion_from_yaw(float(wp['yaw']))
                p.pose.pose.orientation = Quaternion(*q)

                # assign all waypoints default linear velocity
                p.twist.twist.linear.x = float(self.convert_kph2mps(self.velocity))

                waypoints.append(p)

        # recalculate velocities of waypoints at the end of the list so velocities decrease and eventually the last
        # velocity equals to zero
        return self.decelerate(waypoints)

    def distance(self, p1, p2):
        """
        Calculates distance between two points
        :param p1: geometry_msgs/Point
        :param p2: geometry_msgs/Point
        :return: distance
        """
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)

    def decelerate(self, waypoints):
        """
        Set waypoints linear velocities so they start decreasing at some point and the last waypoint has velocity of 0.
        Not clear how deceleration is calculated. It doesn't depend on time.
        :param waypoints: list of waypoints
        :return:
        """
        # remeber last waypoint and set its velocity to 0
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.

        # iterate waypoints starting from the last but one backwards
        for wp in waypoints[:-1][::-1]:
            # calculate distance to the last point
            dist = self.distance(wp.pose.pose.position, last.pose.pose.position)
            # don't understand this, somehow calculate desired velocity so it is lower when closer to the last waypoint
            vel = math.sqrt(2 * MAX_DECEL * dist) * 3.6
            if vel < 1.:
                vel = 0.
            # decrease current waypoint velocity in case proposed velocity is lower
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints

    def publish(self, waypoints):
        """
        Publishes waypoints in the /world frame
        :param waypoints: list of waypoints
        :return:
        """
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointLoader()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint node.')
