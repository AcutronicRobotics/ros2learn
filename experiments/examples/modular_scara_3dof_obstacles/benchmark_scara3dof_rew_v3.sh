#!/bin/sh
SLOWNESS_UNIT='nsec'
TARGET=1
PENALIZED_REWARD=1
#train for 100ms
SLOWNESS=1000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=1
PENALIZED_REWARD=2
#train for 100ms
SLOWNESS=10000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=1
PENALIZED_REWARD=3
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT

killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='sec'
TARGET=1
PENALIZED_REWARD=4
#train for 100ms
SLOWNESS=1
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='sec'
TARGET=1
PENALIZED_REWARD=5
#train for 100ms
SLOWNESS=2
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT


SLOWNESS_UNIT='sec'
TARGET=2
PENALIZED_REWARD=1
#train for 100ms
SLOWNESS=3
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='sec'
TARGET=2
PENALIZED_REWARD=2
#train for 100ms
SLOWNESS=4
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='sec'
TARGET=2
PENALIZED_REWARD=3
#train for 100ms
SLOWNESS=5
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='sec'
TARGET=2
PENALIZED_REWARD=4
#train for 100ms
SLOWNESS=6
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT

killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='sec'
TARGET=2
PENALIZED_REWARD=5
#train for 100ms
SLOWNESS=7
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT



killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='sec'
TARGET=2
PENALIZED_REWARD=3
#train for 100ms
SLOWNESS=8
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT



killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='sec'
TARGET=1
PENALIZED_REWARD=3
#train for 100ms
SLOWNESS=9
python3 train_ppo1.py --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT

killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='sec'
TARGET=1
PENALIZED_REWARD=3
#train for 100ms
SLOWNESS=10
python3 train_ppo1.py --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
