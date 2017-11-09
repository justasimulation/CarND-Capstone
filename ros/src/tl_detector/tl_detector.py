#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.object_detector import ObjectDetector
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np
import scipy.misc
import time
import PIL.Image
import threading

from light_classification.async_pipeline import AsyncPipeline

# to reduce computation cost, classification is done only once in this number of calls of image_cb
CALLS_PER_CLASSIFICATION = 2

LIGHT_NAMES = {TrafficLight.RED: "Red",
               TrafficLight.YELLOW: "Yellow",
               TrafficLight.GREEN: "Green",
               TrafficLight.UNKNOWN: "Unknown"}


class TLDetector(object):
    """
    This node is responsible for publishing waypoint index of next traffic light stop line.
    It receives color image and does traffic lights detection and classification in a dedicated thread.
    """
    def __init__(self):
        rospy.init_node('tl_detector')

        # classifier working in a dedicated thread
        self.classifier = AsyncPipeline()

        # Current car pose. Updated with 50Hz
        self.pose           = None
        # List of all waypoints. Set once on start.
        self.waypoints_lane = None
        # List of traffic lights data. Updated with 50Hz
        self.lights         = None

        # This flag indicates that green or yellow light were detected after the car crossed the stop line.
        # Once the car saw green or yellow after stop line, it can proceed regardless of traffic light.
        self.saw_yellow_green_after_stop_line = False

        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)

        # contains list of [[x, y]...] stop lines coordinates
        # each traffic light has a corresponding stop line in front of it
        self.stop_line_positions = config["stop_line_positions"]
        # contains indices of waypoints closest to stop lines
        # the waypoints are calculated once when base_waypoints are set
        self.stop_line_waypoints_indices = None

        # last traffic light's stop waypoint that was detected enough times
        self.last_ackholedged_waypoint_idx = -1

        # for self.pose update
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        # for self.wayppoints_line update
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        # for self.lights update
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        # traffic light detection is done on this callback
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        # publisher of the next stop line waypoint
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # start classification thread
        self.classifier.start_thread()

        rospy.spin()

    def pose_cb(self, msg):
        """
        Called with 50Hz frequency.
        Remembers car's pose.
        :param msg:
        """
        self.pose = msg

    def waypoints_cb(self, waypoints_lane):
        """
        Called once on app start.
        1. List of waypoints is memorized in self.waypoints_lane.
        2. Indices of waypoints closest to  self.stop_line_positions are calculated and put into
           self.stop_line_waypoints_indices
        :param waypoints_lane: Lane message
        """
        if self.waypoints_lane is not None:
            rospy.logwarn("Waypoints are being reset")

        self.waypoints_lane = waypoints_lane

        waypoint_indices = []

        for x, y in self.stop_line_positions:
            closest_idx = self.find_closest_waypoint_idx(x, y, 0, self.waypoints_lane.waypoints)
            waypoint_indices.append(closest_idx)

        self.stop_line_waypoints_indices = waypoint_indices

    def traffic_cb(self, msg):
        """
        Called with 50Hz frequency.
        Memorizes traffic lights states. For each traffic light pose and state are reported.
        At test time states will be unavailable.
        :param msg: TrafficLightArray (which contains a list of TrafficLight messages)
        :return:
        """
        self.lights = msg.lights

    def image_cb(self, msg):
        """
        Called with 10Hz frequency.
        Identifies red lights in the incoming camera image and publishes the index
        of the waypoint closest to the red light's stop line to /traffic_waypoint

        :param msg: image from car-mounted camera
        """
        # 1. Send image to classifier
        self.classifier.sh_set_image(msg)

        # 2. Find closest traffic light and its stop line
        light, stop_line_idx = self.get_next_light_and_stopline()

        # 3. Get classification results
        #    classification is performed in a different thread, so this call doesn't take long, it just
        #    gets cached results
        acknowledged_state, last_classified_state, last_classification_time = self.classifier.sh_get_state_info()

        # 4. Get stop line waypoint based on location of the car and current state
        #    Sometimes due to classification latency, start on green light is very slow which may lead
        #    to red light observation in the middle of an intersection. To avoid stopping in such a situation,
        #    waypoint is calculated based on current state and location. In case the car observes green or yellow
        #    light after stop line, it is free to proceed even if the traffic light changes to red.
        acknowledged_waypoint_idx = self.get_traffic_waypont_idx(acknowledged_state, stop_line_idx)

        if self.last_ackholedged_waypoint_idx < 0 and acknowledged_waypoint_idx > 0:
            rospy.loginfo("Stop light detected")
        elif self.last_ackholedged_waypoint_idx >= 0 and acknowledged_waypoint_idx < 0:
            rospy.loginfo("Stop light is no longer valid")

        self.last_ackholedged_waypoint_idx = acknowledged_waypoint_idx

        # 4. Publish waypoint
        self.upcoming_red_light_pub.publish(Int32(acknowledged_waypoint_idx))

        # 5. Update knowledge about green/yellow light observed after stop line
        self.update_saw_yellow_green_state(stop_line_idx, acknowledged_state)


        # debugging
        #light_str = "None" #LIGHT_NAMES[light.state] if light is not None else "None"
        #classified_str = LIGHT_NAMES[last_classified_state] if last_classified_state is not None else "None"
        #print("True: {}, Classified: {}, Time: {:.3f}".format(light_str, classified_str, last_classification_time))

    def update_saw_yellow_green_state(self, stop_line_idx, acknowledged_state):
        """
        Updates flag indicating whether green or yellow light was observed after the car crossed stop line.
        In case it was observed, the car is free to proceed. This is needed to avoid situation when due to slow start,
        the car observes red light in the middle of the intersection and stops.
        :param stop_line_idx: stop line index in self.stop_line_positions
        :param acknowledged_state: traffic light state acknowledged by the classifier
        :return:
        """

        if stop_line_idx < 0: # in case there is no stop line reset the state
            self.saw_yellow_green_after_stop_line = False
        else:
            #otherwise calculate stop line x coordinate in car's frame
            stop_line_position = self.stop_line_positions[stop_line_idx]
            rel_stop_line_x, _, _ = self.to_frame(self.pose, stop_line_position[0], stop_line_position[1], 0)

            if rel_stop_line_x >= 0: # reset state in case stop line is ahead of the car
                self.saw_yellow_green_after_stop_line = False
            elif acknowledged_state == TrafficLight.GREEN or acknowledged_state == TrafficLight.YELLOW:
                # in case stop line is behind the car and current state is yellow/green set the state
                self.saw_yellow_green_after_stop_line = True
            # in case stop line is behind the car and state is red, do nothing

    def get_traffic_waypont_idx(self, new_state, stopline_idx):
        """
        Returns index of a waypoint closest to given stop line. In case stop line is behind the car and yellow or green
        light was observed, no waypoint is returned, because there is no need to stop in such a scenario.
        :param new_state: current traffic light state
        :param stopline_idx: stop line index in self.stop_line_positions
        :return: index of a waypoint, corresponding to given stop line, or -1 if there is no waypoint or stop is not
                 needed
        """
        # 1. in case yellow/uknown state or not all needed variables are initialized, return nothing
        if new_state == TrafficLight.GREEN or new_state == TrafficLight.UNKNOWN or \
                        stopline_idx < 0 or \
                        self.stop_line_waypoints_indices is None or self.stop_line_positions is None:
            return -1

        # 2. find index of waypoint correspnding to given stopline
        waypoint_idx = self.stop_line_waypoints_indices[stopline_idx]

        # 3. find relative position of the stopline to learn if it is in front of behind the car
        stop_line_position = self.stop_line_positions[stopline_idx]
        rel_stop_line_x, _, _ = self.to_frame(self.pose, stop_line_position[0], stop_line_position[1], 0)

        # 4. in case of yellow, consider is a stop sign if stop line was not crossed, otherwise consider it as
        #    green sign
        if new_state == TrafficLight.YELLOW:
            return waypoint_idx if rel_stop_line_x >= 0 else -1

        # 5. in case of red, consider it as top sign only in case the car did not cross stop line on yellow/green
        if new_state == TrafficLight.RED:
            return -1 if self.saw_yellow_green_after_stop_line else waypoint_idx

        # 6. should not happen
        return -1

    def find_closest_waypoint_idx(self, x, y, z, waypoints):
        """
        Finds waypoint closest to x, y, z coordinates
        :param x:
        :param y:
        :param z:
        :param waypoints: list of waypoints
        :return: index of closest waypoint
        """
        min_dist = float("inf")
        min_idx = -1

        for i, waypoint in enumerate(waypoints):
            dx = x - waypoint.pose.pose.position.x
            dy = y - waypoint.pose.pose.position.y
            dz = z - waypoint.pose.pose.position.z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    def get_next_light_and_stopline(self):
        """
        Returns next traffic light and its stopline index

        NOTE! The assumption is that both stop line and its waypoint are located closer to the car than traffic light in
        car's frame.
        :return: (TrafficLight, stop line index)
        """

        # 1. find the closest light that is ahead of the car
        next_light = self.find_next_light()
        if next_light is not None:
            # 2. find stop line which is the closest to the light found before
            closest_stop_line_idx = self.find_light_stopline_idx(next_light)

            if closest_stop_line_idx >= 0:
                return next_light, closest_stop_line_idx
            else:
                return None, -1     # in case stop line is unavailable do not return the light
        else:
            rospy.logwarn("Cannot find the next light")
            return None, -1

    def find_next_light(self):
        """
        Finds closest traffic light which is ahead of the car.
        :return: TrafficLight message or None
        """
        if self.pose is None or self.lights is None:
            rospy.logwarn("Cannot find closest light. Pose or lights info is unavailable.")
            return None
        else:
            closest_light = None
            closest_light_dist = float("inf")

            # 1. iterate over all lights
            for light in self.lights:
                # 2. convert light position to car's frame

                light_position = light.pose.pose.position
                light_rel_x, light_rel_y, light_rel_z = self.to_frame(self.pose,
                                                                      light_position.x,
                                                                      light_position.y,
                                                                      light_position.z)

                # 3. find the closest light ahead of the car
                if light_rel_x >= 0:
                    light_dist = np.sqrt(light_rel_x**2 + light_rel_y**2 + light_rel_z**2)
                    if light_dist < closest_light_dist:
                        closest_light_dist = light_dist
                        closest_light = light

            return closest_light

    def to_frame(self, frame_pose, x, y, z):
        """
        Converts given x, y, z coordinates to frame defined by given pose.
        frame_pose and x, y, z should be defined in the same frame.
        :param frame_pose: frame defining pose
        :param x: coordinate to convert
        :param y: coordinate to convert
        :param z: coordinate to convert
        :return: (coverted x, converted y, converted z)
        """
        # 1 convert to np
        position = np.asarray([x, y, z, 1])

        # 2 new frame translation matrix
        frame_center = frame_pose.pose.position
        mat_translation = tf.transformations.translation_matrix((-frame_center.x, -frame_center.y, -frame_center.z))

        # 3 new frame rotation matrix
        frame_orientation = frame_pose.pose.orientation
        frame_quaternion = np.array([frame_orientation.x, frame_orientation.y, frame_orientation.z, frame_orientation.w])

        frame_inverse_quternion = tf.transformations.quaternion_inverse(frame_quaternion)
        mat_rotation = tf.transformations.quaternion_matrix(frame_inverse_quternion)

        # 4 concatenated transform matrix
        mat_transform = np.dot(mat_translation.T, mat_rotation.T)

        # 5 coordinates in new frame
        rel_x, rel_y, rel_z = np.dot(position, mat_transform)[:3]

        return rel_x, rel_y, rel_z

    def find_light_stopline_idx(self, light):
        """
        Finds a stop line which is the closest to the light.

        NOTE!! the assumption is that the stop line is in front of the light.
        :param light: TrafficLight message
        :return: stop line index in self.stop_line_positions
        """
        if self.stop_line_positions is None:
            rospy.logwarn("Cannot find next stop line because stop line info is unavailable.")
            return -1
        else:
            closest_stop_line_idx = -1
            closest_stop_line_dist = float("inf")
            light_position = light.pose.pose.position

            for i, (x, y) in enumerate(self.stop_line_positions):
                dx, dy, dz = x - light_position.x, y - light_position.y, 0 - light_position.z
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                if dist < closest_stop_line_dist:
                    closest_stop_line_dist = dist
                    closest_stop_line_idx = i

            return closest_stop_line_idx

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
