#include <uav_system_ros.h>

namespace mrs_multirotor_simulator
{

/* UavSystemRos //{ */

UavSystemRos::UavSystemRos(ros::NodeHandle& nh, const std::string uav_name) {

  time_last_input_ = ros::Time(0);

  _uav_name_ = uav_name;

  mrs_lib::ParamLoader param_loader(nh, "UavSystemRos");

  std::string type;
  param_loader.loadParam(uav_name + "/type", type);

  // | --------------------- general params --------------------- |

  param_loader.loadParam("frames/world/name", _frame_world_);
  bool prefix_world_name;
  param_loader.loadParam("frames/world/prefix_with_uav_name", prefix_world_name);

  if (prefix_world_name) {
    _frame_world_ = uav_name + "/" + _frame_world_;
  }

  param_loader.loadParam("frames/rangefinder/name", _frame_rangefinder_);

  _frame_rangefinder_ = uav_name + "/" + _frame_rangefinder_;

  param_loader.loadParam("frames/rangefinder/publish_tf", _publish_rangefinder_tf_);
  param_loader.loadParam("frames/fcu/publish_tf", _publish_fcu_tf_);

  param_loader.loadParam("frames/fcu/name", _frame_fcu_);

  _frame_fcu_ = uav_name + "/" + _frame_fcu_;

  param_loader.loadParam("g", model_params_.g);
  param_loader.loadParam("iterate_without_input", _iterate_without_input_);
  param_loader.loadParam("input_timeout", _input_timeout_);
  param_loader.loadParam("ground/enabled", model_params_.ground_enabled);
  param_loader.loadParam("ground/z", model_params_.ground_z);
  param_loader.loadParam("takeoff_patch/enabled", model_params_.takeoff_patch_enabled);

  // | ------------------ model-specific params ----------------- |

  param_loader.loadParam(type + "/n_motors", model_params_.n_motors);
  param_loader.loadParam(type + "/mass", model_params_.mass);
  param_loader.loadParam(type + "/arm_length", model_params_.arm_length);
  param_loader.loadParam(type + "/body_height", model_params_.body_height);
  param_loader.loadParam(type + "/air_resistance_coeff", model_params_.air_resistance_coeff);
  param_loader.loadParam(type + "/motor_time_constant", model_params_.motor_time_constant);
  param_loader.loadParam(type + "/propulsion/prop_radius", model_params_.prop_radius);
  param_loader.loadParam(type + "/propulsion/force_constant", model_params_.kf);
  param_loader.loadParam(type + "/propulsion/moment_constant", model_params_.km);
  param_loader.loadParam(type + "/propulsion/rpm/min", model_params_.min_rpm);
  param_loader.loadParam(type + "/propulsion/rpm/max", model_params_.max_rpm);

  // | --------------------- spawn location --------------------- |

  double spawn_x;
  double spawn_y;
  double spawn_z;
  double spawn_heading;

  param_loader.loadParam(uav_name + "/spawn/x", spawn_x);
  param_loader.loadParam(uav_name + "/spawn/y", spawn_y);
  param_loader.loadParam(uav_name + "/spawn/z", spawn_z);
  param_loader.loadParam(uav_name + "/spawn/heading", spawn_heading);

  // create the inertia matrix
  model_params_.J = Eigen::Matrix3d::Zero();
  model_params_.J(0, 0) =
      model_params_.mass * (3.0 * model_params_.arm_length * model_params_.arm_length + model_params_.body_height * model_params_.body_height) / 12.0;
  model_params_.J(1, 1) =
      model_params_.mass * (3.0 * model_params_.arm_length * model_params_.arm_length + model_params_.body_height * model_params_.body_height) / 12.0;
  model_params_.J(2, 2) = (model_params_.mass * model_params_.arm_length * model_params_.arm_length) / 2.0;

  model_params_.allocation_matrix = param_loader.loadMatrixDynamic2(type + "/propulsion/allocation_matrix", 4, -1);

  model_params_.allocation_matrix.row(0) *= model_params_.arm_length * model_params_.kf;
  model_params_.allocation_matrix.row(1) *= model_params_.arm_length * model_params_.kf;
  model_params_.allocation_matrix.row(2) *= model_params_.km * (3.0 * model_params_.prop_radius) * model_params_.kf;
  model_params_.allocation_matrix.row(3) *= model_params_.kf;

  uav_system_ = UavSystem(model_params_, Eigen::Vector3d(spawn_x, spawn_y, spawn_z), spawn_heading);

  // | -------------------------- mixer ------------------------- |

  Mixer::Params mixer_params;

  param_loader.loadParam("mixer/desaturation", mixer_params.desaturation);

  uav_system_.setMixerParams(mixer_params);

  // | --------------------- rate controller -------------------- |

  RateController::Params rate_controller_params;

  param_loader.loadParam("rate_controller/kp", rate_controller_params.kp);
  param_loader.loadParam("rate_controller/kd", rate_controller_params.kd);
  param_loader.loadParam("rate_controller/ki", rate_controller_params.ki);

  uav_system_.setRateControllerParams(rate_controller_params);

  // | --------------------- attitude controller -------------------- |

  AttitudeController::Params attitude_controller_params;

  param_loader.loadParam("attitude_controller/kp", attitude_controller_params.kp);
  param_loader.loadParam("attitude_controller/kd", attitude_controller_params.kd);
  param_loader.loadParam("attitude_controller/ki", attitude_controller_params.ki);
  param_loader.loadParam("attitude_controller/max_rate_roll_pitch", attitude_controller_params.max_rate_roll_pitch);
  param_loader.loadParam("attitude_controller/max_rate_yaw", attitude_controller_params.max_rate_yaw);

  uav_system_.setAttitudeControllerParams(attitude_controller_params);

  // | ------------------- velocity controller ------------------ |

  VelocityController::Params velocity_controller_params;

  param_loader.loadParam("velocity_controller/kp", velocity_controller_params.kp);
  param_loader.loadParam("velocity_controller/kd", velocity_controller_params.kd);
  param_loader.loadParam("velocity_controller/ki", velocity_controller_params.ki);
  param_loader.loadParam("velocity_controller/max_acceleration", velocity_controller_params.max_acceleration);

  uav_system_.setVelocityControllerParams(velocity_controller_params);

  // | ------------------- position controller ------------------ |

  PositionController::Params position_controller_params;

  param_loader.loadParam("position_controller/kp", position_controller_params.kp);
  param_loader.loadParam("position_controller/kd", position_controller_params.kd);
  param_loader.loadParam("position_controller/ki", position_controller_params.ki);
  param_loader.loadParam("position_controller/max_velocity", position_controller_params.max_velocity);

  uav_system_.setPositionControllerParams(position_controller_params);

  if (!param_loader.loadedSuccessfully()) {
    ROS_ERROR("[%s]: failed to load all parameters", _uav_name_.c_str());
    ros::shutdown();
  }

  // | ----------------------- publishers ----------------------- |

  ph_imu_         = mrs_lib::PublisherHandler<sensor_msgs::Imu>(nh, uav_name + "/imu", 1, false);
  ph_odom_        = mrs_lib::PublisherHandler<nav_msgs::Odometry>(nh, uav_name + "/odom", 1, false);
  ph_rangefinder_ = mrs_lib::PublisherHandler<sensor_msgs::Range>(nh, uav_name + "/rangefinder", 1, false);

  // | ----------------------- subscribers ---------------------- |

  mrs_lib::SubscribeHandlerOptions shopts;
  shopts.nh                 = nh;
  shopts.node_name          = _uav_name_;
  shopts.no_message_timeout = mrs_lib::no_timeout;
  shopts.threadsafe         = true;
  shopts.autostart          = true;
  shopts.queue_size         = 10;
  shopts.transport_hints    = ros::TransportHints().tcpNoDelay();

  sh_attitude_rate_cmd_ =
      mrs_lib::SubscribeHandler<mrs_msgs::HwApiAttitudeRateCmd>(shopts, uav_name + "/attitude_rate_cmd", &UavSystemRos::callbackAttitudeRateCmd, this);

  sh_attitude_cmd_ = mrs_lib::SubscribeHandler<mrs_msgs::HwApiAttitudeCmd>(shopts, uav_name + "/attitude_cmd", &UavSystemRos::callbackAttitudeCmd, this);

  sh_acceleration_cmd_ =
      mrs_lib::SubscribeHandler<mrs_msgs::HwApiAccelerationCmd>(shopts, uav_name + "/acceleration_cmd", &UavSystemRos::callbackAccelerationCmd, this);

  sh_velocity_cmd_ = mrs_lib::SubscribeHandler<mrs_msgs::HwApiVelocityCmd>(shopts, uav_name + "/velocity_cmd", &UavSystemRos::callbackVelocityCmd, this);

  sh_position_cmd_ = mrs_lib::SubscribeHandler<mrs_msgs::HwApiPositionCmd>(shopts, uav_name + "/position_cmd", &UavSystemRos::callbackPositionCmd, this);

  // | --------------------- tf broadcaster --------------------- |

  tf_broadcaster_ = std::make_shared<mrs_lib::TransformBroadcaster>();

  // | ------------------ first model iteration ----------------- |

  // * we need to iterate the model first to initialize its state
  // * this needs to happen in order to publish the correct state
  //   when using iterate_without_input == false

  reference::Actuators actuators_cmd;

  actuators_cmd.motors = Eigen::VectorXd::Zero(model_params_.n_motors);

  // set the motor input for the model
  uav_system_.setInput(actuators_cmd);

  // iterate the model twise to initialize all the states
  uav_system_.makeStep(0.01);
  uav_system_.makeStep(0.01);

  is_initialized_ = true;

  ROS_INFO("[%s]: initialized", _uav_name_.c_str());
}

//}

/* makeStep() //{ */

void UavSystemRos::makeStep(const double dt) {

  // | ---------------- check timeout of an input --------------- |

  auto time_last_input = mrs_lib::get_mutexed(mutex_time_last_input_, time_last_input_);

  if (time_last_input > ros::Time(0)) {
    if ((ros::Time::now() - time_last_input).toSec() > _input_timeout_) {
      ROS_WARN("[%s]: input timeouted", _uav_name_.c_str());

      {
        std::scoped_lock lock(mutex_uav_system_);
        uav_system_.setInput();
      }

      {
        std::scoped_lock lock(mutex_time_last_input_);
        time_last_input_ = ros::Time(0);
      }
    }
  }

  // | --------------------- model iteration -------------------- |

  if (_iterate_without_input_ || time_last_input_ > ros::Time(0)) {

    std::scoped_lock lock(mutex_uav_system_);

    // iterate the model
    uav_system_.makeStep(dt);
  }

  // extract the current state
  MultirotorModel::State state = uav_system_.getState();

  // publish data

  publishOdometry(state);

  publishIMU(state);

  publishRangefinder(state);
}

//}

/* getPose() //{ */

Eigen::Vector3d UavSystemRos::getPose(void) {

  return uav_system_.getState().x;
}

//}

/* crash() //{ */

void UavSystemRos::crash(void) {
  uav_system_.crash();
}

//}

/* applyForce() //{ */

void UavSystemRos::applyForce(const Eigen::Vector3d& force) {
  uav_system_.applyForce(force);
}

//}

// | ----------------------- publishers ----------------------- |

/* publishOdometry() //{ */

void UavSystemRos::publishOdometry(const MultirotorModel::State& state) {

  nav_msgs::Odometry odom;

  odom.header.stamp    = ros::Time::now();
  odom.header.frame_id = _frame_world_;
  odom.child_frame_id  = _frame_fcu_;

  odom.pose.pose.orientation = mrs_lib::AttitudeConverter(state.R);

  odom.pose.pose.position.x = state.x[0];
  odom.pose.pose.position.y = state.x[1];
  odom.pose.pose.position.z = state.x[2];

  Eigen::Vector3d vel_body = state.R.transpose() * state.v;

  odom.twist.twist.linear.x = vel_body[0];
  odom.twist.twist.linear.y = vel_body[1];
  odom.twist.twist.linear.z = vel_body[2];

  odom.twist.twist.angular.x = state.omega[0];
  odom.twist.twist.angular.y = state.omega[1];
  odom.twist.twist.angular.z = state.omega[2];

  ph_odom_.publish(odom);
}

//}

/* publishIMU() //{ */

void UavSystemRos::publishIMU(const MultirotorModel::State& state) {

  sensor_msgs::Imu imu;

  imu.header.stamp    = ros::Time::now();
  imu.header.frame_id = _frame_fcu_;

  imu.angular_velocity.x = state.omega[0];
  imu.angular_velocity.y = state.omega[1];
  imu.angular_velocity.z = state.omega[2];

  auto acc = uav_system_.getImuAcceleration();

  imu.linear_acceleration.x = acc[0];
  imu.linear_acceleration.y = acc[1];
  imu.linear_acceleration.z = acc[2];

  ph_imu_.publish(imu);
}

//}

/* publishRangefinder() //{ */

void UavSystemRos::publishRangefinder(const MultirotorModel::State& state) {

  // | ----------------------- publish tf ----------------------- |

  const Eigen::Vector3d body_z          = state.R.col(2);
  const Eigen::Vector3d rangefinder_dir = -body_z;

  // calculate the angle between the drone's z axis and the world's z axis
  double tilt = acos(rangefinder_dir.dot(Eigen::Vector3d(0, 0, -1)));

  double range_measurement;

  if (body_z[2] > 0) {
    range_measurement = (state.x[2] - model_params_.ground_z) / cos(tilt);
  } else {
    range_measurement = std::numeric_limits<double>::max();
  }

  if (range_measurement > 40.0) {
    range_measurement = 41.0;
  }

  sensor_msgs::Range range;

  range.header.frame_id = _frame_rangefinder_;
  range.header.stamp    = ros::Time::now();
  range.max_range       = 40.0;
  range.min_range       = 0.15;
  range.range           = range_measurement;
  range.radiation_type  = range.INFRARED;
  range.field_of_view   = 0.01;

  ph_rangefinder_.publish(range);

  if (_publish_rangefinder_tf_) {

    geometry_msgs::TransformStamped tf;

    tf.header.stamp    = ros::Time::now();
    tf.header.frame_id = _frame_fcu_;
    tf.child_frame_id  = _frame_rangefinder_;

    tf.transform.translation.x = 0;
    tf.transform.translation.y = 0;
    tf.transform.translation.z = -0.05;

    tf.transform.rotation = mrs_lib::AttitudeConverter(0, 1.57, 0);

    tf_broadcaster_->sendTransform(tf);
  }

  if (_publish_fcu_tf_) {

    geometry_msgs::TransformStamped tf;

    tf.header.stamp    = ros::Time::now();
    tf.header.frame_id = _frame_world_;
    tf.child_frame_id  = _frame_fcu_;

    tf.transform.translation.x = state.x[0];
    tf.transform.translation.y = state.x[1];
    tf.transform.translation.z = state.x[2];

    tf.transform.rotation = mrs_lib::AttitudeConverter(state.R.transpose());

    tf_broadcaster_->sendTransform(tf);
  }
}

//}

// | ------------------------ callbacks ----------------------- |

/* callbackAttitudeRateCmd() //{ */

void UavSystemRos::callbackAttitudeRateCmd([[maybe_unused]] mrs_lib::SubscribeHandler<mrs_msgs::HwApiAttitudeRateCmd>& wrp) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[%s]: getting attitude rate command", _uav_name_.c_str());

  mrs_msgs::HwApiAttitudeRateCmd::ConstPtr msg = wrp.getMsg();

  reference::AttitudeRate cmd;

  cmd.throttle = msg->throttle;
  cmd.rate_x   = msg->body_rate.x;
  cmd.rate_y   = msg->body_rate.y;
  cmd.rate_z   = msg->body_rate.z;

  {
    std::scoped_lock lock(mutex_uav_system_);

    uav_system_.setInput(cmd);
  }

  {
    std::scoped_lock lock(mutex_time_last_input_);

    time_last_input_ = ros::Time::now();
  }
}

//}

/* callbackAttitudeCmd() //{ */

void UavSystemRos::callbackAttitudeCmd([[maybe_unused]] mrs_lib::SubscribeHandler<mrs_msgs::HwApiAttitudeCmd>& wrp) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[%s]: getting attitude command", _uav_name_.c_str());

  mrs_msgs::HwApiAttitudeCmd::ConstPtr msg = wrp.getMsg();

  reference::Attitude cmd;

  cmd.throttle = msg->throttle;

  cmd.orientation = mrs_lib::AttitudeConverter(msg->orientation);

  {
    std::scoped_lock lock(mutex_uav_system_);

    uav_system_.setInput(cmd);
  }

  {
    std::scoped_lock lock(mutex_time_last_input_);

    time_last_input_ = ros::Time::now();
  }
}

//}

/* callbackAccelerationCmd() //{ */

void UavSystemRos::callbackAccelerationCmd([[maybe_unused]] mrs_lib::SubscribeHandler<mrs_msgs::HwApiAccelerationCmd>& wrp) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[%s]: getting acceleration command", _uav_name_.c_str());

  mrs_msgs::HwApiAccelerationCmd::ConstPtr msg = wrp.getMsg();

  reference::Acceleration cmd;

  cmd.heading = msg->heading;

  cmd.acceleration[0] = msg->acceleration.x;
  cmd.acceleration[1] = msg->acceleration.y;
  cmd.acceleration[2] = msg->acceleration.z;

  {
    std::scoped_lock lock(mutex_uav_system_);

    uav_system_.setInput(cmd);
  }

  {
    std::scoped_lock lock(mutex_time_last_input_);

    time_last_input_ = ros::Time::now();
  }
}

//}

/* callbackVelocityCmd() //{ */

void UavSystemRos::callbackVelocityCmd([[maybe_unused]] mrs_lib::SubscribeHandler<mrs_msgs::HwApiVelocityCmd>& wrp) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[%s]: getting velocity command", _uav_name_.c_str());

  mrs_msgs::HwApiVelocityCmd::ConstPtr msg = wrp.getMsg();

  reference::Velocity cmd;

  cmd.heading = msg->heading;

  cmd.velocity[0] = msg->velocity.x;
  cmd.velocity[1] = msg->velocity.y;
  cmd.velocity[2] = msg->velocity.z;

  {
    std::scoped_lock lock(mutex_uav_system_);

    uav_system_.setInput(cmd);
  }

  {
    std::scoped_lock lock(mutex_time_last_input_);

    time_last_input_ = ros::Time::now();
  }
}

//}

/* callbackPositionCmd() //{ */

void UavSystemRos::callbackPositionCmd([[maybe_unused]] mrs_lib::SubscribeHandler<mrs_msgs::HwApiPositionCmd>& wrp) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[%s]: getting position command", _uav_name_.c_str());

  mrs_msgs::HwApiPositionCmd::ConstPtr msg = wrp.getMsg();

  reference::Position cmd;

  cmd.heading = msg->heading;

  cmd.position[0] = msg->position.x;
  cmd.position[1] = msg->position.y;
  cmd.position[2] = msg->position.z;

  {
    std::scoped_lock lock(mutex_uav_system_);

    uav_system_.setInput(cmd);
  }

  {
    std::scoped_lock lock(mutex_time_last_input_);

    time_last_input_ = ros::Time::now();
  }
}

//}

}  // namespace mrs_multirotor_simulator