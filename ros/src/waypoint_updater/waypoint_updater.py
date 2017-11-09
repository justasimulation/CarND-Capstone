#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32, Bool
from styx_msgs.msg import Lane, Waypoint
import numpy as np


import math

"""
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
"""

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number

DECCELERATION = 1.  # m/s, used for breaking speed calculation
ACCELERATION  = 6.  # m/s, used for increasing speed calculation

WAYPOINTS_SEARCH_HORIZON = 30 # next start waypoint should be this number of waypoints close to the last start waypoint


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        # publisher of trajectory waypoints
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        # all waypoints
        self.waypoints_lane          = None
        # car's current pose
        self.pose                    = None
        # car's next published waypoint index from the last iteration
        self.last_start_waypoint_idx = None
        # last stop line waypoint index
        self.traffic_waypoint_idx    = None
        # current car's velocity
        self.current_velocity        = None

        # for self.pose update
        rospy.Subscriber("/current_pose", PoseStamped, self.pose_cb)
        # for self.waypoints_lane update
        rospy.Subscriber("/base_waypoints", Lane, self.waypoints_cb)
        # for traffic
        rospy.Subscriber("/traffic_waypoint", Int32, self.traffic_waypoint_cb)
        # for self.current_velocity update
        rospy.Subscriber("/current_velocity", TwistStamped, self.current_velocity_cb)

        rospy.Subscriber("/vehicle/dbw_enabled", Bool, self.dbw_enabled_cb)

        rospy.spin()

    def pose_cb(self, msg):
        """
        :param msg:
        :return:
        """
        # TODO: Implement

        self.pose = msg

        if self.waypoints_lane is None:
            rospy.logwarn("pose_cp: pose_cb called before waypoints_cb")
            return

        if self.current_velocity is None:
            rospy.logwarn("pose_cp: velocity is unavailable")
            return

        # 1. find the next waypoint
        next_idx = self.get_closest_next_waypoint_idx()
        self.last_start_waypoint_idx = next_idx

        # 2. create lane some points ahead of the car
        lane = self.create_waypoint_lane(next_idx)

        # 3. adjust waypoints velocities in case there is a traffic light ahead
        self.adjust_velocity_for_traffic_light(lane, next_idx)

        # 4. publish lane
        self.final_waypoints_pub.publish(lane)

    def waypoints_cb(self, waypoints_lane):
        """
        Called once on app start.
        Memorizes all waypoints.
        :param waypoints_lane: Lane message
        """
        # TODO: Implement
        if self.waypoints_lane is not None:
            rospy.logwarn("waypoints_cb: waypoints are already set")

        self.waypoints_lane = waypoints_lane

    def traffic_waypoint_cb(self, msg):
        """
        Called with 10Hz frequency.
        Updates last stop line waypoint index in front of the closest traffic light ahead of the car.
        :param msg: Int message
        """
        # TODO: Callback for /traffic_waypoint message. Implement
        self.traffic_waypoint_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def dbw_enabled_cb(self, dbw_enabled):
        """
        Called on demand.
        Resets last waypoint in case dbw was switched to avoid a situation when a manually driven car is far from the
        last point and all near points are out of search horizon.
        :param dbw_enabled:
        :return:
        """
        self.last_start_waypoint_idx = None


    def current_velocity_cb(self, velocity):
        """
        Called with 50Hz frequency.
        Updates current car's velocity
        :param velocity: Int message
        """
        self.current_velocity = velocity

    def get_closest_next_waypoint_idx(self):
        """
        Finds a waypoint which is the closest to the car's position and is ahead of it
        :return: closest waypoint index
        """
        min_distance  = float("inf")
        min_idx       = -1
        num_waypoints = len(self.waypoints_lane.waypoints)

        start_idx = 0 if self.last_start_waypoint_idx is None else self.last_start_waypoint_idx - WAYPOINTS_SEARCH_HORIZON
        end_idx = num_waypoints if self.last_start_waypoint_idx is None else self.last_start_waypoint_idx + WAYPOINTS_SEARCH_HORIZON

        for idx in range(start_idx, end_idx):
            idx %= num_waypoints
            distance = self.calc_position_distance(self.waypoints_lane.waypoints[idx].pose.pose.position,
                                                   self.pose.pose.position)
            if distance < min_distance:
                min_distance = distance
                min_idx = idx

        if min_idx < 0:
            rospy.logerr("get_closest_next_waypoint_idx: could not find closest waypoint")
            return -1

        next_idx = (min_idx + 1) % num_waypoints

        distance_to_next = self.calc_position_distance(self.pose.pose.position,
                                                       self.waypoints_lane.waypoints[next_idx].pose.pose.position)
        distance_between = self.calc_position_distance(self.waypoints_lane.waypoints[min_idx].pose.pose.position,
                                                       self.waypoints_lane.waypoints[next_idx].pose.pose.position)

        next_idx = next_idx if distance_to_next**2 + min_distance**2 < distance_between**2 else min_idx

        return next_idx

    @classmethod
    def get_velocity_at_distance(self, start_velocity, acceleration, distance):
        """
        Calculates velocity at given distance according to s = v*t + a*t^2/2
        :param start_velocity: start velocity
        :param acceleration: acceleration which is used to calc the speed
        :param distance: distance to calc velocity at
        :return: velocity at given distance
        """
        return np.sqrt(start_velocity ** 2 + (2 * acceleration * distance))

    @classmethod
    def calc_position_distance(self, position1, position2):
        """
        Calculates distance between two positions
        :param position1:
        :param position2:
        :return: distance
        """
        return math.sqrt((position1.x - position2.x)**2 +
                         (position1.y - position2.y)**2 +
                         (position1.z - position2.z)**2)

    @classmethod
    def copy_waypoint(self, ref_waypoint, cur_time):
        """
        Need a copy of waypoint but for some reasons deepcopy takes much time,
        so this manual copying is used at the moment.
        :param ref_waypoint: waypoint to copy
        :param cur_time: current time
        :return: new copy of given waypoint
        """
        waypoint = Waypoint()
        waypoint.pose.header.seq = ref_waypoint.pose.header.seq
        waypoint.pose.header.stamp = cur_time
        waypoint.pose.header.frame_id = ref_waypoint.pose.header.frame_id

        waypoint.pose.pose.position.x = ref_waypoint.pose.pose.position.x
        waypoint.pose.pose.position.y = ref_waypoint.pose.pose.position.y
        waypoint.pose.pose.position.z = ref_waypoint.pose.pose.position.z

        waypoint.pose.pose.orientation.x = ref_waypoint.pose.pose.orientation.x
        waypoint.pose.pose.orientation.y = ref_waypoint.pose.pose.orientation.y
        waypoint.pose.pose.orientation.z = ref_waypoint.pose.pose.orientation.z
        waypoint.pose.pose.orientation.w = ref_waypoint.pose.pose.orientation.w

        waypoint.twist.header.seq = ref_waypoint.twist.header.seq
        waypoint.twist.header.stamp = cur_time
        waypoint.twist.header.frame_id = ref_waypoint.twist.header.frame_id

        waypoint.twist.twist.linear.x = ref_waypoint.twist.twist.linear.x

        return waypoint

    def create_waypoint_lane(self, next_idx):
        """
        Creates a lane of LOOKAHEAD_WPS waypoints starting from next_idx waypoint.
        Only linear speed is set. Its max value is the value set in the original waypoints by waypoint_loader
        :param next_idx:
        :param num_waypoints: overall number of waypoints
        :return: Lane
        """
        cur_time = rospy.Time().now()

        lane = Lane()
        lane.header.frame_id = self.waypoints_lane.header.frame_id
        lane.header.stamp = cur_time
        lane.waypoints = []

        cur_velocity = self.current_velocity.twist.linear.x

        for i in range(LOOKAHEAD_WPS):
            idx = (next_idx + i) % len(self.waypoints_lane.waypoints)

            waypoint = self.waypoints_lane.waypoints[idx]
            waypoint = self.copy_waypoint(waypoint, cur_time)

            dist = self.calc_position_distance(self.pose.pose.position, waypoint.pose.pose.position)

            # note here the default velocity is set which is calculated so the car accelerates from the current
            # velocity to max velocity which is set to waypoints by waypoint loader
            waypoint.twist.twist.linear.x = min(waypoint.twist.twist.linear.x,
                                                self.get_velocity_at_distance(cur_velocity, ACCELERATION, dist))

            lane.waypoints.append(waypoint)

        return lane


    def adjust_velocity_for_traffic_light(self, lane, start_waypoint_idx):
        """
        Sets waypoints velocity in case there is a traffic light ahead so velocity slows down to 0 at stop line.
        :param lane:
        :param start_waypoint_idx:
        :return:
        """
        # in case there is a traffic light ahead
        if self.traffic_waypoint_idx >= 0:
            num_waypoints = len(self.waypoints_lane.waypoints)

            # shift next_idx and traffic waypoint index so next_idx is at 0
            # note that this is a circle so, traffic light can have larger index but in fact be behind the car
            delta = num_waypoints - start_waypoint_idx
            # note that the waypoint previous to traffic_waypoint_idx is used, just to have some more space for breking
            rel_traffic_waypoint_idx = (self.traffic_waypoint_idx - 1 + delta) % num_waypoints
            rel_traffic_waypoint_idx = rel_traffic_waypoint_idx if rel_traffic_waypoint_idx < num_waypoints / 2 else\
                -(num_waypoints - rel_traffic_waypoint_idx)

            # in case stop waypoint is ahead of the car calc velocities
            if rel_traffic_waypoint_idx >= 0 and rel_traffic_waypoint_idx < LOOKAHEAD_WPS:
                traffic_waypoint_position = lane.waypoints[rel_traffic_waypoint_idx].pose.pose.position

                # calc velocity so it is 0 at stop point and increases according to s=v*t+a*t^2/2 to the start point
                for i in range(rel_traffic_waypoint_idx, -1, -1):
                    dist = self.calc_position_distance(traffic_waypoint_position, lane.waypoints[i].pose.pose.position)
                    lane.waypoints[i].twist.twist.linear.x = min(lane.waypoints[i].twist.twist.linear.x,
                                                                 self.get_velocity_at_distance(0, DECCELERATION, dist))

                # the rest waypoints get 0 velocity
                for i in range(rel_traffic_waypoint_idx + 1, LOOKAHEAD_WPS):
                    lane.waypoints[i].twist.twist.linear.x = 0

            elif rel_traffic_waypoint_idx < 0:
                # it may happen that the car missed the stop waypoint, in this case all the waypoints get 0 velocity
                for i in range(LOOKAHEAD_WPS):
                    lane.waypoints[i].twist.twist.linear.x = 0


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
