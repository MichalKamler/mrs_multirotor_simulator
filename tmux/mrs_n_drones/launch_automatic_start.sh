#!/bin/bash

# Script to launch automatic start for a custom range of UAVs

# Check if both lower and upper bounds are provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <lower_bound> <upper_bound>"
  exit 1
fi

LOWER_BOUND=$1
UPPER_BOUND=$2

# Loop from LOWER_BOUND to UPPER_BOUND
for i in $(seq $LOWER_BOUND $UPPER_BOUND); do
  export UAV_NAME="uav$i"
  waitForTime
  roslaunch mrs_uav_autostart automatic_start.launch custom_config:=./config/automatic_start.yaml &
  sleep 2  # Add a delay between each launch to prevent overloading the system
done
