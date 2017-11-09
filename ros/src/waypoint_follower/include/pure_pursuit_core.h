/*
 *  Copyright (c) 2015, Nagoya University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name of Autoware nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef PURE_PURSUIT_CORE_H
#define PURE_PURSUIT_CORE_H

// ROS includes
#include <ros/ros.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
// C++ includes
#include <memory>
#include "libwaypoint_follower.h"

// not used
enum class Mode
{
  waypoint,
  dialog
};

namespace waypoint_follower
{

class PurePursuit
{
private:

    //constant
    const double RADIUS_MAX_; // maximum radius
    const double KAPPA_MIN_; // minimum curvature

    bool linear_interpolate_;

    // config topic
    //not used
    int param_flag_;              // 0 = waypoint, 1 = Dialog
    //not used
    double const_lookahead_distance_;  // meter
    //not used
    double initial_velocity_;     // km/h

    double lookahead_distance_calc_ratio_; // lookahead distance is current velocity times this number
    double minimum_lookahead_distance_;  // the next waypoint must be at least this far away from the vehicle

    // maximum distance from a line defined by 2nd and 3d waypoints, if distance is more than this
    // that vehicle is considered as not following the path
    double displacement_threshold_;

    // maximum angle between vehicle and 2nd waypoint, if the angle is more than this value
    // vehicle is considered as not following the path
    double relative_angle_threshold_;

    // true if values are set
    bool waypoint_set_;
    bool pose_set_;
    bool velocity_set_;

    // index of next waypoint lying beyond lookahead distance or last waypoint if not found, or -1 if not present
    int num_of_next_waypoint_;
    // next waypoint position
    geometry_msgs::Point position_of_next_target_;
    // we won't consider waypoints closer than this distance
    double lookahead_distance_;

    geometry_msgs::PoseStamped current_pose_;
    geometry_msgs::TwistStamped current_velocity_;
    WayPoints current_waypoints_;

    //returns linear.x part of velocity stored in waypoint
    double getCmdVelocity(int waypoint) const;

    //calculates minimum distance that waypoints of interest should be away from the vehicle
    void calcLookaheadDistance(int waypoint);
    //calculates curvature of a circle passing through current point and target point
    //circle center is (0, R) in vehicle frame
    double calcCurvature(geometry_msgs::Point target) const;
    //is not used
    double calcRadius(geometry_msgs::Point target) const;

    // finds next waypoint location by linearly interpolating between existing waypoints
    bool interpolateNextTarget(int next_waypoint, geometry_msgs::Point *next_target) const;
    bool verifyFollowing() const;
    geometry_msgs::Twist calcTwist(double curvature, double cmd_velocity) const;

    //finds the first waypoint beyond lookahead distance, or returns the last point if not found, or -1 if not present
    void getNextWaypoint();

    //returns zero twist
    geometry_msgs::TwistStamped outputZero() const;

    //create stamped twist out of given twist
    //controls for maximum centripetal acceleration
    geometry_msgs::TwistStamped outputTwist(geometry_msgs::Twist t) const;

public:
    PurePursuit(bool linear_interpolate_mode)
    : RADIUS_MAX_(9e10)
    , KAPPA_MIN_(1/RADIUS_MAX_)
    , linear_interpolate_(linear_interpolate_mode)
    , param_flag_(0)
    , const_lookahead_distance_(4.0)
    , initial_velocity_(5.0)
    , lookahead_distance_calc_ratio_(2.0)
    , minimum_lookahead_distance_(6.0)
    , displacement_threshold_(0.2)
    , relative_angle_threshold_(5.)
    , waypoint_set_(false)
    , pose_set_(false)
    , velocity_set_(false)
    , num_of_next_waypoint_(-1)
    , lookahead_distance_(0)
    {
    }
    ~PurePursuit()
    {
    }

    // for ROS
    // subscribers for topics
    void callbackFromCurrentPose(const geometry_msgs::PoseStampedConstPtr &msg);
    void callbackFromCurrentVelocity(const geometry_msgs::TwistStampedConstPtr &msg);
    void callbackFromWayPoints(const styx_msgs::LaneConstPtr &msg);

    // for debug
    geometry_msgs::Point getPoseOfNextWaypoint() const
    {
        return current_waypoints_.getWaypointPosition(num_of_next_waypoint_);
    }
    geometry_msgs::Point getPoseOfNextTarget() const
    {
        return position_of_next_target_;
    }
    geometry_msgs::Pose getCurrentPose() const
    {
        return current_pose_.pose;
    }

    double getLookaheadDistance() const
    {
        return lookahead_distance_;
    }

    // processing for ROS
    // calculates proposed twist
    geometry_msgs::TwistStamped go();
};

}

#endif  // PURE_PURSUIT_CORE_H
