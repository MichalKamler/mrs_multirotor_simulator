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
  UAV_NAME="uav$i"
  waitForControl
  rosservice call /$UAV_NAME/hw_api/arming 1  # Arm the UAV
  sleep 2  # Wait for arming
  rosservice call /$UAV_NAME/hw_api/offboard  # Switch to offboard mode
  sleep 2  # Wait before next UAV
done
