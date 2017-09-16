#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"

# fetch a gazebo model
curl -o double_pendulum.sdf http://models.gazebosim.org/double_pendulum_with_base/model-1_4.sdf

# run gzserver
gzserver &

# run it in gazebo
gz model --model-name double_pendulum --spawn-file double_pendulum.sdf

exec "$@"
