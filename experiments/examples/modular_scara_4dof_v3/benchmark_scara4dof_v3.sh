#!/bin/sh

TIMESTEPS=1e6
ENVIRONMENT="GazeboModularScara4DOF-v3"
SLOWNESS=1
SLOWNESS_UNIT='sec'
# Train all algorithm in 1 max_seconds

python3 train_ppo1.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT

python3 train_ppo2.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT

python3 train_acktr.py  --slowness $SLOWNESS  --slowness-unit $SLOWNESS_UNIT
