#!/bin/sh


ENVIRONMENT="GazeboModularScara3DOF-v3"

# SLOWNESS=1
# SLOWNESS_UNIT='sec'
# python3 train_ppo2.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT


# MACRO_DURATION=5 #opt value
# WARMUP_TIME=5
# TRAIN_TIME=200
# SAVEDIR='/media/erle/RL/networks/GazeboModularScara3DOFv3Env/two_random_targets_parameters/macro_opt_warmup_5_train_200'
# python3 train_mlsh.py  --macro_duration $MACRO_DURATION   --warmup_time $WARMUP_TIME --train_time $TRAIN_TIME --savedir $SAVEDIR
#
# MACRO_DURATION=5 #opt value
# WARMUP_TIME=10
# TRAIN_TIME=200
# SAVEDIR='/media/erle/RL/networks/GazeboModularScara3DOFv3Env/two_random_targets_parameters/macro_opt_warmup_10_train_200'
# python3 train_mlsh.py  --macro_duration $MACRO_DURATION   --warmup_time $WARMUP_TIME --train_time $TRAIN_TIME --savedir $SAVEDIR
#
#
# MACRO_DURATION=5 #opt value
# WARMUP_TIME=10 #opt value
# TRAIN_TIME=50
# SAVEDIR='/media/erle/RL/networks/GazeboModularScara3DOFv3Env/two_random_targets_parameters/macro_opt_warmup_opt_train_50'
# python3 train_mlsh.py  --macro_duration $MACRO_DURATION   --warmup_time $WARMUP_TIME --train_time $TRAIN_TIME --savedir $SAVEDIR
#
#
# MACRO_DURATION=5 #opt value
# WARMUP_TIME=2 #opt value
# TRAIN_TIME=100
# SAVEDIR='/media/erle/RL/networks/GazeboModularScara3DOFv3Env/two_random_targets_parameters/macro_opt_warmup_opt_train_100'
# python3 train_mlsh.py  --macro_duration $MACRO_DURATION   --warmup_time $WARMUP_TIME --train_time $TRAIN_TIME --savedir $SAVEDIR

MACRO_DURATION=5 #opt value
WARMUP_TIME=5 #opt value
TRAIN_TIME=10
SAVEDIR='/tmp/rosrl/GazeboModularScara3DOFv3Env/two_random_targets_parameters/macro_5_warmup_5_train_10'
python3 train_mlsh.py  --macro_duration $MACRO_DURATION   --warmup_time $WARMUP_TIME --train_time $TRAIN_TIME --savedir $SAVEDIR

MACRO_DURATION=5 #opt value
WARMUP_TIME=0 #opt value
TRAIN_TIME=200
SAVEDIR='/tmp/rosrl/GazeboModularScara3DOFv3Env/two_random_targets_parameters/macro_5_warmup_0_train_200'
python3 train_mlsh.py  --macro_duration $MACRO_DURATION   --warmup_time $WARMUP_TIME --train_time $TRAIN_TIME --savedir $SAVEDIR
