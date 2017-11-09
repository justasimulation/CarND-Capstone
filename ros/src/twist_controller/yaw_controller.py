from math import atan


class YawController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        """
        Store given values.
        :param wheel_base: distance between centers of rear and front wheels in bicycle model
        :param steer_ratio: ratio [steering wheel angle]/[front wheels angle] for passenger cars is usually 12:1, 20:1
        :param min_speed:
        :param max_lat_accel:
        :param max_steer_angle:
        """
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel

        self.min_angle = -max_steer_angle
        self.max_angle = max_steer_angle

    def get_angle(self, radius):
        """
        Returns steering angle needed to make a turn with a given turning radius
        :param radius: turning radius
        :return:
        """
        # calculate turning angle of the bicycle model based on given turning radius and
        # convert it to steering angle by multiplying by steering ratio
        angle = atan(self.wheel_base / radius) * self.steer_ratio
        return max(self.min_angle, min(self.max_angle, angle))

    def get_steering(self, prop_linear_velocity, prop_angular_velocity, cur_linear_velocity):
        """
        Returns steering angle which accordance with given velocity and proposed velocities
        :param prop_linear_velocity: proposed linear velocity
        :param prop_angular_velocity: proposed angular velocity
        :param cur_linear_velocity: current linear velocity
        :return: angle for steering wheel
        """
        # calculate angular velocity needed to turn with proposed radius
        # r_prop = v_prop / w_prop
        # w = v_cur / r_prop
        prop_angular_velocity = cur_linear_velocity * prop_angular_velocity / prop_linear_velocity \
            if abs(prop_linear_velocity) > 0. else 0.

        # check for maximum centripetal acceleration
        if abs(cur_linear_velocity) > 0.1:
            # calc max angular velocity
            # a = v^2/r => a/v = v/r => a/v = w
            max_yaw_rate = abs(self.max_lat_accel / cur_linear_velocity)
            prop_angular_velocity = max(-max_yaw_rate, min(max_yaw_rate, prop_angular_velocity))

        # correct radius if needed with respect to minimum speed and max angular velocity
        # and get steering angle corresponding to proposed radius
        return self.get_angle(max(cur_linear_velocity, self.min_speed) / prop_angular_velocity) \
            if abs(prop_angular_velocity) > 0. else 0.0
