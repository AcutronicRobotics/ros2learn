#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"

# run gzserver
gzserver &

# run it in gazebo
# gz model --model-name double_pendulum --spawn-file double_pendulum.sdf

exec "$@"
