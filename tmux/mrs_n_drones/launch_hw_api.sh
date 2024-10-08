#!/bin/bash

# Script to launch automatic start for a custom range of UAVs

# Check if both lower and upper bounds are provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <lower_bound> <upper_bound>"
  exit 1
fi

LOWER_BOUND=$1
UPPER_BOUND=$2

for i in $(seq $LOWER_BOUND $UPPER_BOUND); do
  waitForTime
  export UAV_NAME="uav$i"
  roslaunch mrs_multirotor_simulator hw_api.launch custom_config:=./config/hw_api.yaml nodelet_manager_name:="${UAV_NAME}\_nodelet_manager" &
  sleep 2  # Add a delay between each launch to prevent overloading the system
done
