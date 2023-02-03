#ifndef UAV_SYSTEM_ROS_H
#define UAV_SYSTEM_ROS_H

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <mrs_lib/transform_broadcaster.h>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/subscribe_handler.h>
#include <mrs_lib/mutex.h>
#include <mrs_lib/attitude_converter.h>

#include <mrs_multirotor_simulator/uav_system/uav_system.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>

#include <mrs_msgs/HwApiAttitudeRateCmd.h>
#include <mrs_msgs/HwApiAttitudeCmd.h>
#include <mrs_msgs/HwApiAccelerationCmd.h>
#include <mrs_msgs/HwApiVelocityCmd.h>
#include <mrs_msgs/HwApiPositionCmd.h>

namespace mrs_multirotor_simulator
{

class UavSystemRos {

public:
  UavSystemRos(ros::NodeHandle& nh, const std::string name);

  void makeStep(const double dt);

  void crash(void);

  void applyForce(const Eigen::Vector3d& force);

  Eigen::Vector3d getPose(void);

private:
  std::atomic<bool> is_initialized_ = false;
  std::string       _uav_name_;

  // | ------------------------ UavSystem ----------------------- |

  UavSystem  uav_system_;
  std::mutex mutex_uav_system_;

  ros::Time  time_last_input_;
  std::mutex mutex_time_last_input_;

  ModelParams model_params_;

  bool   _iterate_without_input_;
  double _input_timeout_;

  std::string _frame_world_;
  std::string _frame_fcu_;
  std::string _frame_rangefinder_;
  bool        _publish_rangefinder_tf_;
  bool        _publish_fcu_tf_;

  // | ----------------------- publishers ----------------------- |

  mrs_lib::PublisherHandler<sensor_msgs::Imu>   ph_imu_;
  mrs_lib::PublisherHandler<nav_msgs::Odometry> ph_odom_;
  mrs_lib::PublisherHandler<sensor_msgs::Range> ph_rangefinder_;

  void publishOdometry(const MultirotorModel::State& state);
  void publishIMU(const MultirotorModel::State& state);
  void publishRangefinder(const MultirotorModel::State& state);

  // | --------------------------- tf --------------------------- |

  std::shared_ptr<mrs_lib::TransformBroadcaster> tf_broadcaster_;

  // | ----------------------- subscribers ---------------------- |

  void callbackAttitudeRateCmd(mrs_lib::SubscribeHandler<mrs_msgs::HwApiAttitudeRateCmd>& wrp);
  void callbackAttitudeCmd(mrs_lib::SubscribeHandler<mrs_msgs::HwApiAttitudeCmd>& wrp);
  void callbackAccelerationCmd(mrs_lib::SubscribeHandler<mrs_msgs::HwApiAccelerationCmd>& wrp);
  void callbackVelocityCmd(mrs_lib::SubscribeHandler<mrs_msgs::HwApiVelocityCmd>& wrp);
  void callbackPositionCmd(mrs_lib::SubscribeHandler<mrs_msgs::HwApiPositionCmd>& wrp);

  mrs_lib::SubscribeHandler<mrs_msgs::HwApiAttitudeRateCmd> sh_attitude_rate_cmd_;
  mrs_lib::SubscribeHandler<mrs_msgs::HwApiAttitudeCmd>     sh_attitude_cmd_;
  mrs_lib::SubscribeHandler<mrs_msgs::HwApiAccelerationCmd> sh_acceleration_cmd_;
  mrs_lib::SubscribeHandler<mrs_msgs::HwApiVelocityCmd>     sh_velocity_cmd_;
  mrs_lib::SubscribeHandler<mrs_msgs::HwApiPositionCmd>     sh_position_cmd_;
};

}  // namespace mrs_multirotor_simulator

#endif  // UAV_SYSTEM_ROS_H