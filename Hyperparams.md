## Env. specific Hyperparameters

Each environment ([gym-gazebo2 Env](https://github.com/AcutronicRobotics/gym-gazebo2/tree/master/gym_gazebo2/envs/MARA)) has it's optimal hyperparameters that allow the agent to learn faster and achieve a better policy. In the following table we present our best parameters.

Please open a new issue and share your results if you found better parameters!

### PPO2 MLP - MARA
Content: [baselines/ppo2/defaults.py](https://github.com/AcutronicRobotics/baselines/blob/91eef3578b63ba32c2e17251d3116405fb9e9cc3/baselines/ppo2/defaults.py#L26-L54).

| Environment  | num_layers | num_hidden | nsteps | nminibatches | lr | cliprange |
| ------------ | ---------- | ---------- | ------ | ------------ | -- | --------- |
| MARA  | 2 | 16 | 1024 | 4 | lambda f: 3e-3 * math.e**(-0.001918*update) | 0.25 |
| MARA Collision |  |  |  |  |  |  |
| MARA Orient |  |  |  |  |  |  |
| MARA Collision Orient |  |  |  |  |  |  |
