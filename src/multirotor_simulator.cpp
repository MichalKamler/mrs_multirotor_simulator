/* includes //{ */

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <uav_system_ros.h>

#include <rosgraph_msgs/Clock.h>

#include <geometry_msgs/PoseArray.h>

#include <mrs_lib/param_loader.h>
#include <mrs_lib/publisher_handler.h>

#include <dynamic_reconfigure/server.h>
#include <mrs_multirotor_simulator/multirotor_simulatorConfig.h>

#include <KDTreeVectorOfVectorsAdaptor.h>
#include <Eigen/Dense>

#include <mrs_msgs/VelocityReferenceStampedSrv.h>
#include <mrs_msgs/ReferenceStamped.h>
#include <std_srvs/Trigger.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

namespace mrs_multirotor_simulator
{

typedef std::vector<Eigen::VectorXd> my_vector_of_vectors_t;

class MultirotorSimulator : public nodelet::Nodelet {

public:
  virtual void onInit();

private:
  ros::NodeHandle   nh_;
  std::atomic<bool> is_initialized_;

  // | ------------------------- params ------------------------- |

  double _simulation_rate_;

  ros::Time  sim_time_;
  std::mutex mutex_sim_time_;

  std::mutex mutex_current_state_;
  int current_state_;

  double _clock_min_dt_;

  std::string _world_frame_name_;

  // | ------------------------- timers ------------------------- |

  ros::WallTimer timer_main_;
  void           timerMain(const ros::WallTimerEvent& event);

  ros::WallTimer timer_status_;
  void           timerStatus(const ros::WallTimerEvent& event);

  // | ------------------------ rtf check ----------------------- |

  double    actual_rtf_ = 1.0;
  ros::Time last_sim_time_status_;

  // | ----------------------- publishers ----------------------- |

  mrs_lib::PublisherHandler<rosgraph_msgs::Clock>     ph_clock_;
  mrs_lib::PublisherHandler<geometry_msgs::PoseArray> ph_poses_;

  // | ------------------------- system ------------------------- |

  std::vector<std::unique_ptr<UavSystemRos>> uavs_;

  // | -------------------------- time -------------------------- |

  ros::Time last_published_time_;

  // | ------------------------- methods ------------------------ |

  void handleCollisions(void);

  void publishPoses(void);

  // | --------------- dynamic reconfigure server --------------- |

  boost::recursive_mutex                                       mutex_drs_;
  typedef mrs_multirotor_simulator::multirotor_simulatorConfig DrsConfig_t;
  typedef dynamic_reconfigure::Server<DrsConfig_t>             Drs_t;
  boost::shared_ptr<Drs_t>                                     drs_;
  void                                                         callbackDrs(mrs_multirotor_simulator::multirotor_simulatorConfig& config, uint32_t level);
  DrsConfig_t                                                  drs_params_;
  std::mutex                                                   mutex_drs_params_;
  
  // | -------------- custom added -------------------------------|
  mrs_msgs::VelocityReferenceStampedSrv vel_srv_;
  
  std::vector<ros::ServiceClient> client_vel_ref_arr;
  std::vector<Eigen::VectorXd> uavs_odom;
  void updateVelocities(void);

  Eigen::VectorXd dPhi_V_of(const Eigen::VectorXd &Phi, const Eigen::VectorXd &V);
  Eigen::VectorXd dt_V_of(const Eigen::VectorXd &V_now, const Eigen::VectorXd &V_prev, double dt); 
  std::pair<double, double> compute_state_variables(double vel_now, const Eigen::VectorXd &Phi, const Eigen::VectorXd &V_now, 
                                                                      ros::Time currente_time, ros::Time prev_time, const Eigen::VectorXd &V_prev);
  void compute_visual_field(int id_uav, Eigen::VectorXd &visual_field, double uav_heading);

  void updateUavsOdomVec(void);

  void block_V_field(Eigen::VectorXd &V_i, double psi, double x, double y);
  void block_part_of_V_field(Eigen::VectorXd &V_i, int half_angle_width, int center);

  bool activationServiceCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
  ros::ServiceServer service_activate_control_;
  bool control_allowed = false;

  // | -------------- 3d model -------------------------------|
  void updateVelocities_3d(void);
  Eigen::Vector3d compute_state_variables_3d(double vel_now, std::vector<Eigen::Vector3d> local_frame_coords);
  double compute_dvel_3d(double a_2, double phi_uav, double theta_uav);
  double compute_dpsi_3d(double a_2, double phi_uav, double theta_uav);
  double compute_dvz_3d(double a_2, double phi_uav, double theta_uav);
  std::vector<Eigen::Vector3d> recalculate_uavs_positions_to_uavi_frame(double psi, int id_uav);
  Eigen::Vector3d compute_dvel_dpsi_dvz_1_uav(double dist, double a_2, double phi_uav, double theta_uav, double vel_now);

  double GAM;
  double V0;                                                  
  double ALP0;
  double ALP1;
  double ALP2;
  double BET0;
  double BET1;
  double BET2;
  double LAM0;
  double LAM1;
  double LAM2;
  double R;
  int VIS_FIELD_SIZE;

  bool USE_3D;

  double D_PHI;
  double D_THETA;

  ros::Time current_time;
  ros::Time prev_time;
  std::vector<Eigen::VectorXd> uavs_prev_V;

  bool USE_BOUNDARY_BOX;
  double BOX_WIDTH;
  double BOX_LENGHT;
  double SQUARE_AROUND_UAV;
};

void MultirotorSimulator::onInit() {

  is_initialized_ = false;

  nh_ = nodelet::Nodelet::getMTPrivateNodeHandle();

  if (!(nh_.hasParam("/use_sim_time"))) {
    nh_.setParam("/use_sim_time", true);
  }

  srand(time(NULL));

  mrs_lib::ParamLoader param_loader(nh_, "MultirotorSimulator");

  std::string custom_config_path;

  param_loader.loadParam("custom_config", custom_config_path);

  if (custom_config_path != "") {
    param_loader.addYamlFile(custom_config_path);
  }

  param_loader.addYamlFileFromParam("config");
  param_loader.addYamlFileFromParam("config_uavs");

  param_loader.loadParam("simulation_rate", _simulation_rate_);
  param_loader.loadParam("realtime_factor", drs_params_.realtime_factor);
  param_loader.loadParam("collisions/enabled", drs_params_.collisions_enabled);
  param_loader.loadParam("collisions/crash", drs_params_.collisions_crash);
  param_loader.loadParam("collisions/rebounce", drs_params_.collisions_rebounce);
  param_loader.loadParam("frames/world/name", _world_frame_name_);
  param_loader.loadParam("fish_model_params/GAM",GAM);
  param_loader.loadParam("fish_model_params/ALP0",ALP0);
  param_loader.loadParam("fish_model_params/ALP1",ALP1);
  param_loader.loadParam("fish_model_params/ALP2",ALP2);
  param_loader.loadParam("fish_model_params/BET0",BET0);
  param_loader.loadParam("fish_model_params/BET1",BET1);
  param_loader.loadParam("fish_model_params/BET2",BET2);
  param_loader.loadParam("fish_model_params/LAM0",LAM0);
  param_loader.loadParam("fish_model_params/LAM1",LAM1);
  param_loader.loadParam("fish_model_params/LAM2",LAM2);
  param_loader.loadParam("fish_model_params/V0",V0);
  param_loader.loadParam("fish_model_params/R",R);
  param_loader.loadParam("fish_model_params/VIS_FIELD_SIZE",VIS_FIELD_SIZE);
  param_loader.loadParam("boundary_box/BOX_WIDTH",BOX_WIDTH);
  param_loader.loadParam("boundary_box/BOX_LENGHT",BOX_LENGHT);
  param_loader.loadParam("boundary_box/USE_BOUNDARY_BOX",USE_BOUNDARY_BOX);
  param_loader.loadParam("boundary_box/SQUARE_AROUND_UAV",SQUARE_AROUND_UAV);
  param_loader.loadParam("fish_model_params/USE_3D",USE_3D);
  param_loader.loadParam("fish_model_params/D_PHI",D_PHI);
  param_loader.loadParam("fish_model_params/D_THETA",D_THETA);

  D_PHI = (2*M_PI)/D_PHI;
  D_THETA = M_PI/D_THETA;
  
  double clock_rate;
  param_loader.loadParam("clock_rate", clock_rate);

  bool sim_time_from_wall_time;
  param_loader.loadParam("sim_time_from_wall_time", sim_time_from_wall_time);

  if (sim_time_from_wall_time) {
    sim_time_ = ros::Time(ros::WallTime::now().toSec());
  } else {
    sim_time_ = ros::Time(0);
  }

  last_published_time_  = sim_time_;
  last_sim_time_status_ = sim_time_;

  drs_params_.paused = false;

  std::vector<std::string> uav_names;

  param_loader.loadParam("uav_names", uav_names);

  Eigen::VectorXd Vis_field = Eigen::VectorXd::Zero(VIS_FIELD_SIZE);
  for (size_t i = 0; i < uav_names.size(); i++) {

    std::string uav_name = uav_names.at(i);

    ros::ServiceClient client_vel_ref_ = nh_.serviceClient<mrs_msgs::VelocityReferenceStampedSrv>("/" + uav_name + "/control_manager/velocity_reference");
    client_vel_ref_arr.push_back(client_vel_ref_);

    uavs_prev_V.push_back(Vis_field);

    ROS_INFO("[MultirotorSimulator]: initializing '%s'", uav_name.c_str());

    uavs_.push_back(std::make_unique<UavSystemRos>(nh_, uav_name));
  }

  // | --------------- dynamic reconfigure server --------------- |

  drs_.reset(new Drs_t(mutex_drs_, nh_));
  drs_->updateConfig(drs_params_);
  Drs_t::CallbackType f = boost::bind(&MultirotorSimulator::callbackDrs, this, _1, _2);
  drs_->setCallback(f);

  if (!param_loader.loadedSuccessfully()) {
    ROS_ERROR("[MultirotorSimulator]: could not load all parameters!");
    ros::shutdown();
  }

  _clock_min_dt_ = 1.0 / clock_rate;

  // | ----------------------- publishers ----------------------- |

  ph_clock_ = mrs_lib::PublisherHandler<rosgraph_msgs::Clock>(nh_, "clock_out", 10, false);

  ph_poses_ = mrs_lib::PublisherHandler<geometry_msgs::PoseArray>(nh_, "uav_poses_out", 10, false);

  mrs_lib::SubscribeHandlerOptions shopts;
  shopts.nh                 = nh_;
  shopts.node_name          = "AreaMonitoringController";
  shopts.no_message_timeout = mrs_lib::no_timeout;
  shopts.threadsafe         = true;
  shopts.autostart          = true;
  shopts.queue_size         = 10;
  shopts.transport_hints    = ros::TransportHints().tcpNoDelay();

  service_activate_control_   = nh_.advertiseService("control_activation_in", &MultirotorSimulator::activationServiceCallback, this);

  // | ------------------------- timers ------------------------- |

  timer_main_ = nh_.createWallTimer(ros::WallDuration(1.0 / (_simulation_rate_ * drs_params_.realtime_factor)), &MultirotorSimulator::timerMain, this);

  timer_status_ = nh_.createWallTimer(ros::WallDuration(1.0), &MultirotorSimulator::timerStatus, this);

  // | ----------------------- finish init ---------------------- |

  current_time = ros::Time::now();
  prev_time = ros::Time::now();

  is_initialized_ = true;

  ROS_INFO("[MultirotorSimulator]: initialized");
}

void MultirotorSimulator::timerMain([[maybe_unused]] const ros::WallTimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  ROS_INFO_ONCE("[MultirotorSimulator]: main timer spinning");

  double simulation_step_size = 1.0 / _simulation_rate_;

  // step the time
  sim_time_ = sim_time_ + ros::Duration(simulation_step_size);

  for (size_t i = 0; i < uavs_.size(); i++) {
    uavs_.at(i)->makeStep(simulation_step_size);
  }

  publishPoses();
  // handleCollisions();
  
  static ros::Time last_velocity_update_time = ros::Time(0);  // Store last update time
  // updateUavsOdomVec();
  if ((sim_time_ - last_velocity_update_time).toSec() >= 0.1) {  // 0.5 seconds = 2 Hz
    if (USE_3D){
      updateVelocities_3d();
    }else{
      updateVelocities();
    }
    last_velocity_update_time = sim_time_;  // Update the time
  }

  // | ---------------------- publish time ---------------------- |

  if ((sim_time_ - last_published_time_).toSec() >= _clock_min_dt_) {

    rosgraph_msgs::Clock ros_time;

    ros_time.clock.fromSec(sim_time_.toSec());

    ph_clock_.publish(ros_time);

    last_published_time_ = sim_time_;
  }

  
}

void MultirotorSimulator::timerStatus([[maybe_unused]] const ros::WallTimerEvent& event) {

  if (!is_initialized_) {
    return;
  }

  auto sim_time   = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);
  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  ros::Duration last_sec_sim_dt = sim_time - last_sim_time_status_;

  last_sim_time_status_ = sim_time;

  double last_sec_rtf = last_sec_sim_dt.toSec() / 1.0;

  actual_rtf_ = 0.9 * actual_rtf_ + 0.1 * last_sec_rtf;

  ROS_INFO_THROTTLE(0.1, "[MultirotorSimulator]: %s, desired RTF = %.2f, actual RTF = %.2f", drs_params.paused ? "paused" : "running",
                    drs_params.realtime_factor, actual_rtf_);
}

void MultirotorSimulator::callbackDrs(mrs_multirotor_simulator::multirotor_simulatorConfig& config, [[maybe_unused]] uint32_t level) {

  {
    // | ----------------- pausing the simulation ----------------- |

    auto old_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

    if (!old_params.paused && config.paused) {
      timer_main_.stop();
    } else if (old_params.paused && !config.paused) {
      timer_main_.start();
    }
  }

  // | --------------------- save the params -------------------- |

  {
    std::scoped_lock lock(mutex_drs_params_);

    drs_params_ = config;
  }

  // | ----------------- set the realtime factor ---------------- |

  timer_main_.setPeriod(ros::WallDuration(1.0 / (_simulation_rate_ * config.realtime_factor)), true);

  ROS_INFO("[MultirotorSimulator]: DRS updated params");
}

void MultirotorSimulator::handleCollisions(void) {

  auto drs_params = mrs_lib::get_mutexed(mutex_drs_params_, drs_params_);

  if (!(drs_params.collisions_crash || drs_params.collisions_enabled)) {
    return;
  }

  std::vector<Eigen::VectorXd> poses;

  for (size_t i = 0; i < uavs_.size(); i++) {
    poses.push_back(uavs_.at(i)->getPose());
  }

  typedef KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t, double> my_kd_tree_t;

  my_kd_tree_t mat_index(3, poses, 10);

  std::vector<nanoflann::ResultItem<int, double>> indices_dists;

  std::vector<Eigen::Vector3d> forces;

  for (size_t i = 0; i < uavs_.size(); i++) {
    forces.push_back(Eigen::Vector3d::Zero());
  }

  for (size_t i = 0; i < uavs_.size(); i++) {

    MultirotorModel::State       state_1  = uavs_.at(i)->getState();
    MultirotorModel::ModelParams params_1 = uavs_.at(i)->getParams();

    nanoflann::RadiusResultSet<double, int> resultSet(3.0, indices_dists);

    mat_index.index->findNeighbors(resultSet, &state_1.x(0));

    for (size_t j = 0; j < resultSet.m_indices_dists.size(); j++) {

      const size_t idx  = resultSet.m_indices_dists.at(j).first;
      const double dist = resultSet.m_indices_dists.at(j).second;

      if (idx == i) {
        continue;
      }

      MultirotorModel::State       state_2  = uavs_.at(idx)->getState();
      MultirotorModel::ModelParams params_2 = uavs_.at(idx)->getParams();

      const double crit_dist = params_1.arm_length + params_1.prop_radius + params_2.arm_length + params_2.prop_radius;

      const Eigen::Vector3d rel_pos = state_1.x - state_2.x;

      if (dist < crit_dist) {
        if (drs_params.collisions_crash) {
          uavs_.at(idx)->crash();
        } else {
          forces.at(i) += drs_params.collisions_rebounce * rel_pos.normalized() * params_1.mass * (params_2.mass / (params_1.mass + params_2.mass));
        }
      }
    }
  }

  for (size_t i = 0; i < uavs_.size(); i++) {
    uavs_.at(i)->applyForce(forces.at(i));
  }
}

void MultirotorSimulator::publishPoses(void) {

  auto sim_time = mrs_lib::get_mutexed(mutex_sim_time_, sim_time_);

  geometry_msgs::PoseArray pose_array;

  pose_array.header.stamp    = sim_time;
  pose_array.header.frame_id = _world_frame_name_;

  for (size_t i = 0; i < uavs_.size(); i++) {

    auto state = uavs_.at(i)->getState();

    geometry_msgs::Pose pose;

    pose.position.x  = state.x(0);
    pose.position.y  = state.x(1);
    pose.position.z  = state.x(2);
    pose.orientation = mrs_lib::AttitudeConverter(state.R);

    pose_array.poses.push_back(pose);
  }

  ph_poses_.publish(pose_array);
}

void MultirotorSimulator::updateVelocities_3d(void){
  if (!control_allowed){return;}
  updateUavsOdomVec();
  for (size_t i = 0; i < uavs_.size(); i++) { 
    double x_i = uavs_odom[i][0];
    double y_i = uavs_odom[i][1];
    double yaw = uavs_odom[i][6];
    double vx_local = uavs_odom[i][3];  // local x velocity
    double vy_local = uavs_odom[i][4];  // local y velocity
    double vz_global = uavs_odom[i][5]; // z is same local and global

    Eigen::VectorXd visual_field_i_uav = Eigen::VectorXd::Zero(VIS_FIELD_SIZE);
    Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(VIS_FIELD_SIZE, -M_PI, M_PI);
    
    double vel_norm = sqrt(pow(vx_local ,2) + pow(vx_local,2)); //norm of velocity
    double vx_global = vx_local * cos(yaw) - vy_local * sin(yaw);
    double vy_global = vx_local * sin(yaw) + vy_local * cos(yaw);
    double psi = atan2(vy_global, vx_global); //global coord heading

    if(USE_BOUNDARY_BOX){
      ROS_INFO("BOUNDARY BOX not defined yet for 3d model");
    }
    Eigen::Vector3d state_var; //norm of dv_i, d_psi, dvz

    prev_time = current_time;
    current_time = ros::Time::now();

    std::vector<Eigen::Vector3d> local_frame_coords = recalculate_uavs_positions_to_uavi_frame(psi, i);

    state_var = compute_state_variables_3d(vel_norm, local_frame_coords);

    vel_norm = vel_norm + state_var(0);
    psi = psi + state_var(1);

    vel_srv_.request.reference.reference.velocity.x = vel_norm * cos(psi); //global coords set up
    vel_srv_.request.reference.reference.velocity.y = vel_norm * sin(psi); 
    vel_srv_.request.reference.reference.velocity.z = vz_global + state_var(2); 

    if (i==0){
      ROS_INFO("x %.3f, y %.3f, z %.3f", vel_norm * cos(psi), vel_norm * sin(psi), vz_global + state_var(2));
    }

    vel_srv_.request.reference.reference.use_heading = true;
    vel_srv_.request.reference.reference.heading = psi;

    if (client_vel_ref_arr[i].call(vel_srv_)){
      // ROS_INFO("------------------ Service call succesful uav ---------------");
    } else {
      ROS_ERROR("------------------ Service call unsuccesful uav ---------------");
    }
  }
}

Eigen::Vector3d MultirotorSimulator::compute_state_variables_3d(double vel_now, std::vector<Eigen::Vector3d> local_frame_coords) { //
  // Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(VIS_FIELD_SIZE, -M_PI, M_PI);
  // Eigen::VectorXd theta = Eigen::VectorXd::LinSpaced(VIS_FIELD_SIZE/2, -M_PI_2, M_PI_2);

  double radius = R;
  double dvel = 0;
  double dpsi = 0;
  double dvel_z = 0; 

  ROS_INFO("__________start___________");
  for (unsigned int i = 0; i<local_frame_coords.size(); i++){
    double x = local_frame_coords[i](0);
    double y = local_frame_coords[i](1); //cords of some next uav 
    double z = local_frame_coords[i](2);
    double dist = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
    double a_2 = atan(R/dist); //"widht height" of the circle in the visual field in rad, a_2 as half of the angle i need to iterate through
    double phi_uav = atan2(y, x);
    double theta_uav = atan(z/sqrt(pow(x,2)+pow(y,2)));
    // ROS_INFO("dist %.3f a2 %.3f phiuav %.3f thetauav %.3f", dist, a_2, phi_uav, theta_uav);
    
    dvel += compute_dvel_3d(a_2, phi_uav, theta_uav);
    dpsi += compute_dpsi_3d(a_2, phi_uav, theta_uav);
    dvel_z += compute_dvz_3d(a_2, phi_uav, theta_uav);

    // Eigen::Vector3d state_var = compute_dvel_dpsi_dvz_1_uav(dist, a_2, phi_uav, theta_uav, vel_now);

    // dvel += state_var(0);
    // dpsi += state_var(1);
    // dvel_z += state_var(2);
  }
  ROS_INFO("__________end___________");
  dvel += GAM * (V0 - vel_now);
  
  return Eigen::Vector3d(dvel, dpsi, dvel_z);;
}

// Eigen::Vector3d MultirotorSimulator:: compute_dvel_dpsi_dvz_1_uav(double dist, double a_2, double phi_uav, double theta_uav, double vel_now){
//   int center_j = (phi_uav + M_PI) / (2 * M_PI / (VIS_FIELD_SIZE - 1)); //center of the j uav in the list of the visual field in the i uav frame of reference to psi
//   int half_angle_width = static_cast<int>(a_2 / (2 * M_PI / (VIS_FIELD_SIZE - 1)));

//   Eigen::VectorXd Phi = Eigen::VectorXd::LinSpaced(VIS_FIELD_SIZE, -M_PI, M_PI);

//   Eigen::VectorXd V_i = Eigen::VectorXd::Zero(VIS_FIELD_SIZE);
//   for (unsigned int k = 0; k <= 2 * half_angle_width; k++) {
//     if(dist>R){
//       int idx = (center_j - half_angle_width + k + VIS_FIELD_SIZE) % VIS_FIELD_SIZE;
//       V_i[idx] = 1;
//     }else{
//       ROS_ERROR("uav_i above or under uav_i");
//     }
//   }
//   double dPhi = (2*M_PI)/VIS_FIELD_SIZE;

//   Eigen::VectorXd dPhi_V = dPhi_V_of(Phi, V_i);
  
//   Eigen::ArrayXd G_vel = -V_i.array();
//   Eigen::ArrayXd G_psi = -V_i.array();
//   Eigen::ArrayXd G_spike = dPhi_V.array().square();
  
//   Eigen::ArrayXd cos_phi = Phi.array().cos();
//   Eigen::ArrayXd sin_phi = Phi.array().sin();

//   Eigen::ArrayXd integrand_dvel = G_vel * cos_phi;
//   Eigen::ArrayXd integrand_dpsi = G_psi * sin_phi;

//   double integral_dvel = dPhi * (0.5 * integrand_dvel[0] + integrand_dvel.segment(1, VIS_FIELD_SIZE - 2).sum() + 0.5 * integrand_dvel[VIS_FIELD_SIZE - 1]);
//   double integral_dpsi = dPhi * (0.5 * integrand_dpsi[0] + integrand_dpsi.segment(1, VIS_FIELD_SIZE - 2).sum() + 0.5 * integrand_dpsi[VIS_FIELD_SIZE - 1]);

//   double dvel = GAM * (V0 - vel_now) + ALP0 * integral_dvel + ALP0 * ALP1 * (cos_phi * G_spike).sum();
//   double dpsi = BET0 * integral_dpsi + BET0 * BET1 * (sin_phi * G_spike).sum();
//   double dvz = 0;
//   return Eigen::Vector3d(dvel, dpsi, dvz);
// }


// ∂t vψi(t ) = ∫−π/2π/2d θi cos θi ∫−ππd ϕi cos ϕi α0(−Vi( ϕi, θi, t ) +α1 ( ∂ϕi Vi( ϕi, θi,t ) )2 ) + γ(v0 − vi(t ) )

double MultirotorSimulator::compute_dvel_3d(double a_2, double phi_uav, double theta_uav){
  double dvel = 0;
  double phi_part = 0;
  for (double theta = theta_uav-a_2; theta < theta_uav+a_2; theta += D_THETA){ //t as theta     
    phi_part = 0;
    for (double phi = phi_uav-a_2; phi < phi_uav+a_2; phi += D_PHI){ // p as phi 
      phi_part += cos(phi) * ALP0 * (-1);
    }
    dvel += D_PHI * phi_part;
  }
  dvel *= D_THETA;
  dvel += cos(theta_uav)*ALP0*ALP1*(cos(phi_uav-a_2) + cos(phi_uav+a_2));   //!!! reaction to edges only left right rigt now 
  return dvel;
}

// double MultirotorSimulator::compute_dvel_3d(double a_2, double phi_uav, double theta_uav){
//   double dvel = 0;
//   double dvel_theta_part = 0;
//   for (double phi = phi_uav-a_2; phi < phi_uav+a_2; phi += D_PHI){ // p as phi 
//     dvel_theta_part = 0;
//     for (double theta = theta_uav-a_2; theta < theta_uav+a_2; theta += D_THETA){ //t as theta 
//       // dvel_theta_part += D_THETA*cos(theta)*D_PHI*cos(phi)*ALP0*(-1); //reaction to mass
//       dvel_theta_part += cos(theta)*cos(phi)*ALP0*(-1); //reaction to mass
//     }
//     dvel += dvel_theta_part*D_THETA;
//   }
//   dvel *= D_PHI;
//   dvel += cos(theta_uav)*ALP0*ALP1*(cos(phi_uav-a_2) + cos(phi_uav+a_2));   //!!! reaction to edges only left right rigt now 
//   ROS_INFO("3d dvel edge is: %.3f without the costheta: %.3f", cos(theta_uav)*ALP0*ALP1*(cos(phi_uav-a_2) + cos(phi_uav+a_2)), ALP0*ALP1*(cos(phi_uav-a_2) + cos(phi_uav+a_2)));
//   return dvel;
// }

double MultirotorSimulator::compute_dpsi_3d(double a_2, double phi_uav, double theta_uav){
  double dpsi = 0;
  double dpsi_theta_part = 0;
  for (double phi = phi_uav-a_2; phi < phi_uav+a_2; phi += D_PHI){ // p as phi 
    dpsi_theta_part = 0;
    for (double theta = theta_uav-a_2; theta < theta_uav+a_2; theta += D_THETA){ //t as theta 
      dpsi_theta_part += cos(theta)*sin(phi)*BET0*(-1); //reaction to mass
    }
    dpsi += dpsi_theta_part*D_THETA;
  }
  dpsi *= D_PHI;
  dpsi += cos(theta_uav)*BET0*BET1*(sin(phi_uav-a_2) + sin(phi_uav+a_2));   //!!! reaction to edges only left right rigt now 
  return dpsi;
}

double MultirotorSimulator::compute_dvz_3d(double a_2, double phi_uav, double theta_uav){
  double dvz = 0;
  double dvz_theta_part = 0;
  for (double phi = phi_uav-a_2; phi < phi_uav+a_2; phi += D_PHI){ // p as phi 
    dvz_theta_part = 0;
    for (double theta = theta_uav-a_2; theta < theta_uav+a_2; theta += D_THETA){ //t as theta 
      dvz_theta_part += sin(theta)*LAM0*(-1); //reaction to mass
    }
    dvz += dvz_theta_part*D_THETA;
  }
  dvz *= D_PHI;
  dvz += LAM0*LAM1*sin(theta_uav)*(2);   //!!! reaction to edges only left right rigt now 
  return dvz;
}

std::vector<Eigen::Vector3d> MultirotorSimulator::recalculate_uavs_positions_to_uavi_frame(double psi, int id_uav){
  std::vector<Eigen::Vector3d> updated_positions;

  double x_i = uavs_odom[id_uav](0);
  double y_i = uavs_odom[id_uav](1);
  double z_i = uavs_odom[id_uav](2);

  for (int j = 0; j < uavs_.size(); j++) {
    if (j != id_uav) {
      Eigen::Vector3d positions_ith_uav;
      double x_j = uavs_odom[j](0) - x_i;
      double y_j = uavs_odom[j](1) - y_i;

      positions_ith_uav(0) = x_j * cos(psi) - y_j * sin(psi); //x
      positions_ith_uav(1) = x_j * sin(psi) + y_j * cos(psi); //y
      positions_ith_uav(2) = uavs_odom[j](2) - z_i; //z

      updated_positions.push_back(positions_ith_uav);
    }
  }
  return updated_positions;
}

void MultirotorSimulator::updateVelocities(void){
  if (!control_allowed){return;}

  updateUavsOdomVec();

  for (size_t i = 0; i < uavs_.size(); i++) { 
    double x_i = uavs_odom[i][0];
    double y_i = uavs_odom[i][1];
    double yaw = uavs_odom[i][6];
    double vx_local = uavs_odom[i][3];  // local x velocity
    double vy_local = uavs_odom[i][4];  // local y velocity

    Eigen::VectorXd visual_field_i_uav = Eigen::VectorXd::Zero(VIS_FIELD_SIZE);
    Eigen::VectorXd phi = Eigen::VectorXd::LinSpaced(VIS_FIELD_SIZE, -M_PI, M_PI);
    
    double vel_norm = sqrt(pow(vx_local ,2) + pow(vx_local,2)); //norm of velocity
    double vx_global = vx_local * cos(yaw) - vy_local * sin(yaw);
    double vy_global = vx_local * sin(yaw) + vy_local * cos(yaw);
    double uav_vel_heading = atan2(vy_global, vx_global); //global coord heading

    compute_visual_field(i, visual_field_i_uav, uav_vel_heading);

    if(USE_BOUNDARY_BOX){
      block_V_field(visual_field_i_uav, uav_vel_heading, x_i, y_i);
    }

    prev_time = current_time;
    current_time = ros::Time::now();

    Eigen::VectorXd prev_V_field = uavs_prev_V[i]; 
    auto dvel_dpsi = compute_state_variables(vel_norm, phi, visual_field_i_uav, current_time, prev_time, prev_V_field);
    uavs_prev_V[i] = visual_field_i_uav;

    Eigen::Vector3d state_var; //norm of dv_i, d_psi, dvz
    // uav_vel_heading = 0;
    std::vector<Eigen::Vector3d> local_frame_coords = recalculate_uavs_positions_to_uavi_frame(uav_vel_heading, i);
    state_var = compute_state_variables_3d(vel_norm, local_frame_coords);

    if(i==0){
      ROS_INFO("dvel 2d: %.3f, dvel 3d: %.3f, dpsi 2d: %.3f, dpsi 3d: %.3f", dvel_dpsi.first, state_var(0), dvel_dpsi.second, state_var(1));
      // ROS_INFO("local_frame_coords x0: %.3f, y0: %.3f, z0: %.3f, x1: %.3f, y1: %.3f, z1: %.3f", local_frame_coords[0](0), local_frame_coords[0](1), local_frame_coords[0](2), local_frame_coords[1](0), local_frame_coords[1](1), local_frame_coords[1](2));
    }



    vel_norm = vel_norm + dvel_dpsi.first;
    uav_vel_heading = uav_vel_heading + dvel_dpsi.second;

    vel_srv_.request.reference.reference.velocity.x = vel_norm * cos(uav_vel_heading); //global coords set up
    vel_srv_.request.reference.reference.velocity.y = vel_norm * sin(uav_vel_heading); 
    if(uavs_odom[i][2]>1.8){
      ROS_INFO("too high up setting vel z down");
      vel_srv_.request.reference.reference.velocity.z = -0.1; 
    }else{
      vel_srv_.request.reference.reference.velocity.z = 0; 
    }

    // if(uavs_odom[i][2]<3 && i==0){
    //   ROS_INFO("too high up setting vel z down");
    //   vel_srv_.request.reference.reference.velocity.z = 0.1; 
    // }else{
    //   vel_srv_.request.reference.reference.velocity.z = 0; 
    // }

    vel_srv_.request.reference.reference.use_heading = true;
    vel_srv_.request.reference.reference.heading = uav_vel_heading;

    if (client_vel_ref_arr[i].call(vel_srv_)){
      // ROS_INFO("------------------ Service call succesful uav ---------------");
    } else {
      ROS_ERROR("------------------ Service call unsuccesful uav ---------------");
    }
  }
}

void MultirotorSimulator::updateUavsOdomVec(void){
  uavs_odom.resize(0);
  for (size_t i = 0; i < uavs_.size(); i++) {
    Eigen::VectorXd uav_od;
    uav_od.resize(7);
    auto state = uavs_.at(i)->getState();
    uav_od[0] = state.x(0); //x
    uav_od[1] = state.x(1); //y
    uav_od[2] = state.x(2); //z
    Eigen::Vector3d vel_body = state.R.transpose() * state.v; //this is not global
    uav_od[3] = vel_body(0); // vel x
    uav_od[4] = vel_body(1); // vel y
    uav_od[5] = vel_body(2); // vel z
    uav_od[6] = mrs_lib::AttitudeConverter(state.R).getHeading(); //yaw
    uavs_odom.push_back(uav_od);
  }
}

Eigen::VectorXd MultirotorSimulator::dPhi_V_of(const Eigen::VectorXd &Phi, const Eigen::VectorXd &V) {

  Eigen::VectorXd padV(V.size() + 2);
  padV << V(V.size() - 1), V, V(0);

  Eigen::VectorXd dPhi_V_raw = padV.tail(padV.size() - 1) - padV.head(padV.size() - 1);

  if (dPhi_V_raw.size() > 1) {
    if (dPhi_V_raw(0) > 0 && dPhi_V_raw(dPhi_V_raw.size() - 1) > 0) {
      dPhi_V_raw = dPhi_V_raw.head(dPhi_V_raw.size() - 1);
    } else {
      dPhi_V_raw = dPhi_V_raw.tail(dPhi_V_raw.size() - 1);
    }
  } else {
    ROS_ERROR("dPhi_V_raw size is too small for operations!");
  }
  return dPhi_V_raw;
}

Eigen::VectorXd MultirotorSimulator::dt_V_of(const Eigen::VectorXd &V_now, const Eigen::VectorXd &V_prev, double dt) {
  if (dt == 0) {
    ROS_ERROR("dt cannot be 0");
  }
  Eigen::VectorXd dt_V = (V_prev - V_now)/dt;
  return dt_V;
}

std::pair<double, double> MultirotorSimulator::compute_state_variables(double vel_now, const Eigen::VectorXd &Phi, const Eigen::VectorXd &V_now, 
                                                                      ros::Time currente_time, ros::Time prev_time, const Eigen::VectorXd &V_prev) {    
  

  double dPhi = (2*M_PI)/VIS_FIELD_SIZE;

  double dt = 1.0 / _simulation_rate_; 

  // double dt = (currente_time.nsec - prev_time.nsec) * pow(10, -9);
  // ROS_INFO("current time: %.3f, prev time: %.3f, dt in s: %.16f", currente_time.nsec, prev_time.nsec, dt);

  Eigen::VectorXd dt_V = dt_V_of(V_now, V_prev, dt);
    
  Eigen::VectorXd dPhi_V = dPhi_V_of(Phi, V_now);
  
  Eigen::ArrayXd G_vel = -V_now.array() + ALP2 * dt_V.array();
  Eigen::ArrayXd G_psi = -V_now.array() + BET2 * dt_V.array();
  Eigen::ArrayXd G_spike = dPhi_V.array().square();
  
  Eigen::ArrayXd cos_phi = Phi.array().cos();
  Eigen::ArrayXd sin_phi = Phi.array().sin();

  // Eigen::ArrayXd bump = (-2 * Phi.array().square()).exp();  // Wider bump
  // cos_phi = cos_phi + bump * (Phi.array().abs() <= M_PI / 2).cast<double>();

  Eigen::ArrayXd integrand_dvel = G_vel * cos_phi;
  Eigen::ArrayXd integrand_dpsi = G_psi * sin_phi;

  double integral_dvel = dPhi * (0.5 * integrand_dvel[0] + integrand_dvel.segment(1, VIS_FIELD_SIZE - 2).sum() + 0.5 * integrand_dvel[VIS_FIELD_SIZE - 1]);
  double integral_dpsi = dPhi * (0.5 * integrand_dpsi[0] + integrand_dpsi.segment(1, VIS_FIELD_SIZE - 2).sum() + 0.5 * integrand_dpsi[VIS_FIELD_SIZE - 1]);

  double dvel = GAM * (V0 - vel_now) + ALP0 * integral_dvel + ALP0 * ALP1 * (cos_phi * G_spike).sum();
  ROS_INFO("2d dvel edge is: %.3f",ALP0 * ALP1 * (cos_phi * G_spike).sum());
  double dpsi = BET0 * integral_dpsi + BET0 * BET1 * (sin_phi * G_spike).sum();

  // ROS_INFO("V part is: %3f and edge part is: %3f", BET0 * integral_dpsi, BET0 * BET1 * (sin_phi * G_spike).sum());

  return std::make_pair(dvel, dpsi);
} 

void MultirotorSimulator::compute_visual_field(int id_uav, Eigen::VectorXd &V_i, double psi) {
  //psi - direction of the movement

  double x_i = uavs_odom[id_uav](0);
  double y_i = uavs_odom[id_uav](1);

  for (int j = 0; j < uavs_.size(); j++) {
    if (j != id_uav) {
      double x_j = uavs_odom[j](0);
      double y_j = uavs_odom[j](1); 

      double d_i_j = sqrt( pow(x_i - x_j, 2) + pow(y_i - y_j, 2) ); //distance between current uav and j th uav
      double Phi_i_j = atan2( (y_j - y_i), (x_j - x_i) ); // angle of i and j uav with reagards to the global coord sys
      double dPhi_i_j = atan(R / d_i_j);

      double angle_uav_frame = atan2( sin(Phi_i_j - psi), cos(Phi_i_j - psi) );
      int center_j = (angle_uav_frame + M_PI) / (2 * M_PI / (VIS_FIELD_SIZE - 1)); //center of the j uav in the list of the visual field in the i uav frame of reference to psi
      int half_angle_width = static_cast<int>(dPhi_i_j / (2 * M_PI / (VIS_FIELD_SIZE - 1)));

      for (unsigned int k = 0; k <= 2 * half_angle_width; k++) {
        if(d_i_j>R){
          int idx = (center_j - half_angle_width + k + VIS_FIELD_SIZE) % VIS_FIELD_SIZE;
          V_i[idx] = 1;
        }else{
          ROS_ERROR("uav_i above or under uav_i");
        }
      }
    }
  }
}

bool MultirotorSimulator::activationServiceCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
  ROS_INFO("[AreaMonitoringController]: Activation service called.");
  res.success = true;
  if (control_allowed) {
    res.message = "Control was already allowed.";
    ROS_WARN("[AreaMonitoringController]: %s", res.message.c_str());
  } else {
    control_allowed = true;
    res.message      = "Control allowed.";
    ROS_INFO("[AreaMonitoringController]: %s", res.message.c_str());
  }
  return true;
}

void MultirotorSimulator::block_V_field(Eigen::VectorXd &V_i, double psi, double x, double y){
  /*
  x, y - cur pos of uav
  psi - velocity heading
  V_i - visual field
  */
  if(!USE_BOUNDARY_BOX){
    return;
  }
  double sq = SQUARE_AROUND_UAV/2;
  double w = BOX_WIDTH/2;
  double l = BOX_LENGHT/2;

  if(x+sq >= w){
    double angle_uav_frame = atan2( sin(0 - psi), cos(0 - psi) );
    int center = (angle_uav_frame + M_PI) / (2 * M_PI / (VIS_FIELD_SIZE - 1)); //center of the j uav in the list of the visual field in the i uav frame of reference to psi
    int half_angle_width = static_cast<int>(M_PI_4 / (2 * M_PI / (VIS_FIELD_SIZE - 1)));
    block_part_of_V_field(V_i, half_angle_width, center);
  }else if(x-sq <= -w){
    double angle_uav_frame = atan2( sin(-M_PI - psi), cos(-M_PI - psi) );
    int center = (angle_uav_frame + M_PI) / (2 * M_PI / (VIS_FIELD_SIZE - 1)); //center of the j uav in the list of the visual field in the i uav frame of reference to psi
    int half_angle_width = static_cast<int>(M_PI_4 / (2 * M_PI / (VIS_FIELD_SIZE - 1)));
    block_part_of_V_field(V_i, half_angle_width, center);
  }

  if(y+sq >= l){
    double angle_uav_frame = atan2( sin(M_PI_2 - psi), cos(M_PI_2 - psi) );
    int center = (angle_uav_frame + M_PI) / (2 * M_PI / (VIS_FIELD_SIZE - 1)); //center of the j uav in the list of the visual field in the i uav frame of reference to psi
    int half_angle_width = static_cast<int>(M_PI_4 / (2 * M_PI / (VIS_FIELD_SIZE - 1)));
    block_part_of_V_field(V_i, half_angle_width, center);
  }else if(y-sq <= -l){
    double angle_uav_frame = atan2( sin(-M_PI_2 - psi), cos(-M_PI_2 - psi) );
    int center = (angle_uav_frame + M_PI) / (2 * M_PI / (VIS_FIELD_SIZE - 1)); //center of the j uav in the list of the visual field in the i uav frame of reference to psi
    int half_angle_width = static_cast<int>(M_PI_4 / (2 * M_PI / (VIS_FIELD_SIZE - 1)));
    block_part_of_V_field(V_i, half_angle_width, center);
  }
}

void MultirotorSimulator::block_part_of_V_field(Eigen::VectorXd &V_i, int half_angle_width, int center){
  for (unsigned int k = 0; k <= 2 * half_angle_width; k++) {
    int idx = (center - half_angle_width + k + VIS_FIELD_SIZE) % VIS_FIELD_SIZE;
    V_i[idx] = 1;
  }
}


}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mrs_multirotor_simulator::MultirotorSimulator, nodelet::Nodelet)
