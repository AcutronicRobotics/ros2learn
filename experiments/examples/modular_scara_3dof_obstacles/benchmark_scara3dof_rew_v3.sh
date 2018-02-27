# # !/bin/sh
# SLOWNESS_UNIT='nsec'
# #train for 1ms
# SLOWNESS=1000000
# python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
#
# killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver
#
# SLOWNESS_UNIT='nsec'
# #train for 10ms
# SLOWNESS=10000000
# python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
#
# killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT

killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

# SLOWNESS_UNIT='sec'
# #train for 1s
# SLOWNESS=1
# python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
#
# killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
#train for 100ms
SLOWNESS=100000000
python3 train_ppo2.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

# SLOWNESS_UNIT='sec'
# #train for 2s
# SLOWNESS=2
# python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
#
# SLOWNESS_UNIT='sec'
# #train for 3s
# SLOWNESS=3
# python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
#
# killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver
#
# SLOWNESS_UNIT='sec'
# #train for 4s
# SLOWNESS=4
# python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
#
# killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver
#
# SLOWNESS_UNIT='sec'
# #train for 5s
# SLOWNESS=5
# python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
#
# killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

# SLOWNESS_UNIT='sec'
# #train for 6s
# SLOWNESS=6
# python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
# killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver
#
# SLOWNESS_UNIT='sec'
# #train for 7s
# SLOWNESS=7
# python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
#
#
# killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver
#
# SLOWNESS_UNIT='sec'
#
# #train for 8s
# SLOWNESS=8
# python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
#
#
# killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver
#
# SLOWNESS_UNIT='sec'
# #train for 9s
# SLOWNESS=9
# python3 train_ppo1.py --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
#
# killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver
#
# SLOWNESS_UNIT='sec'
# #train for 10s
# SLOWNESS=10
# python3 train_ppo1.py --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
