#!/bin/sh





SLOWNESS_UNIT='nsec'
TARGET=1
PENALIZED_REWARD=1
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT --target $TARGET --penalization $PENALIZED_REWARD


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=1
PENALIZED_REWARD=2
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT --target $TARGET --penalization $PENALIZED_REWARD


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=1
PENALIZED_REWARD=3
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT --target $TARGET --penalization $PENALIZED_REWARD


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=1
PENALIZED_REWARD=4
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT --target $TARGET --penalization $PENALIZED_REWARD


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=1
PENALIZED_REWARD=5
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT --target $TARGET --penalization $PENALIZED_REWARD


SLOWNESS_UNIT='nsec'
TARGET=2
PENALIZED_REWARD=1
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT --target $TARGET --penalization $PENALIZED_REWARD


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=2
PENALIZED_REWARD=2
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT --target $TARGET --penalization $PENALIZED_REWARD


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=2
PENALIZED_REWARD=3
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT --target $TARGET --penalization $PENALIZED_REWARD


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=2
PENALIZED_REWARD=4
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT --target $TARGET --penalization $PENALIZED_REWARD


killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=2
PENALIZED_REWARD=5
#train for 100ms
SLOWNESS=100000000
python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT --target $TARGET --penalization $PENALIZED_REWARD



killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=2
PENALIZED_REWARD=3
#train for 100ms
SLOWNESS=100000000
python3 train_mlsh.py --target $TARGET --penalization $PENALIZED_REWARD



killall -9 roscore rosmaster roslaunch robot_state_publisher scara_contact_plugin gzclient gzserver

SLOWNESS_UNIT='nsec'
TARGET=1
PENALIZED_REWARD=3
#train for 100ms
SLOWNESS=100000000
python3 train_mlsh.py --target $TARGET --penalization $PENALIZED_REWARD
