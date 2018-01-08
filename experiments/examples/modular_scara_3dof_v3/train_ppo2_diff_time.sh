#!/bin/sh

TIMESTEPS=1e6
ENVIRONMENT="GazeboModularScara3DOF-v3"
SLOWNESS=1
SLOWNESS_UNIT='sec'
# SLOWNESS_UNIT= 'nsec'
# PPO1 benchmark
# python3 train_ppo1.py  --slowness $SLOWNESS --slowness_unit $SLOWNESS_UNIT
# PPO2 benchmark
python3 train_ppo2.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT

SLOWNESS_UNIT='nsec'
#train for 100ms
SLOWNESS=100000000
python3 train_ppo2.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT

#train for 10ms
SLOWNESS=10000000
python3 train_ppo2.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT

#train for 1ms
SLOWNESS=1000000
python3 train_ppo2.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT

# ACKTR benchmark
# python3 train_acktr.py --slowness $SLOWNESS --slowness_unit $SLOWNESS_UNIT
# VPG benchmark
# python3 train_vpg.py --environment $ENVIRONMENT --num_timesteps $TIMESTEPS
# TRPO benchmark
# python3 train_trpo.py #--environment $ENVIRONMENT --num_timesteps $TIMESTEPS
# DDPG benchmark
# python3 train_ddpg.py #--environment $ENVIRONMENT --num_timesteps $TIMESTEPS
