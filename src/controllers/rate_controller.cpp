#include <controllers/rate_controller.h>
#include <math.h>
#include <iostream>

namespace mrs_multirotor_simulator
{

// constructor
RateController::RateController() {
}

void RateController::setParams(const Params& params) {

  params_ = params;

  pid_x_.reset();
  pid_y_.reset();
  pid_z_.reset();

  pid_x_.setParams(params.kp, params.kd, params.ki);
  pid_y_.setParams(params.kp, params.kd, params.ki);
  pid_z_.setParams(params.kp, params.kd, params.ki);
}

reference::ControlGroup RateController::getControlSignal(const QuadrotorModel::State& state, const reference::AngularRate& reference, const double& dt) {

  Eigen::Vector3d ang_rate_ref = Eigen::Vector3d(reference.rate_x, reference.rate_y, reference.rate_z);

  // angular rate error
  Eigen::Vector3d wr = ang_rate_ref - state.omega;

  reference::ControlGroup output;

  output.roll     = pid_x_.update(wr(0), dt);
  output.pitch    = pid_x_.update(wr(1), dt);
  output.yaw      = pid_z_.update(wr(2), dt);
  output.throttle = reference.throttle;

  return output;
}

}  // namespace mrs_multirotor_simulator
