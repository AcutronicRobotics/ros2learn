<a href="http://www.acutronicrobotics.com"><img src="https://github.com/erlerobot/gym-gazebo-ros2/blob/master/imgs/alr_logo.png" align="left" width="190"></a>

This repository contains a number of ROS/ROS 2 enabled Artificial Intelligence (AI)
and Reinforcement Learning (RL) [algorithms](algorithms/) that run in selected [environments](environments/). 

The repository contains the following:
- [algorithms](algorithms/): techniques used for training and teaching robots.
- [environments](environments/): pre-built environments of interest to train selected robots.
- [optimization](optimization/): techniques for hyperparameter optimization.
- [experiments](experiments/): experiments and examples of the different utilities that this repository provides.

---

## Installation

Please refer to [Install.md](/Install.md).

## Usage

### Train an agent
You will find all available examples at */experiments/examples/*. Although the algorithms are complex, the way to execute them is really simple. For instance, if you want to train MARA using ppo2 and mlp, you should execute the following command:

```sh
cd /experiments/examples/MARA
python3 train_ppo2_mlp.py
```

Note that you can add the command line arguments provided by the environment, which in this case are provided by the gym-gazebo2 Env. Use `-h` to get all the available commands.

If you want to modify the algorith itself, or the environment being executed, you must update it directly in the corresponding algorithm settings file. For this example, we are using *ppo2* from *baselines*, so you can edit the `mara_mpl()` function inside */baselines/baselines/ppo2/defaults.py*.

### Run a trained policy
Once you are done with the training, or if you want to test some specific checkpoint of it, you can run that using one of the running-scripts available. This time, we are going to run a saved ppo2 policy. 

First we will edit the */experiments/examples/MARA/run_ppo2_mlp.py* file and edit the `load_path` to the checkpoint we want. Now we are ready to launch the script.

Since we want to visualize it in real conditions, we are also going to set some flags:

```sh
cd /experiments/examples/MARA
python3 run_ppo2_mlp.py -g -r -v 0.5
```

This will launch the simulation with the visual interface, real time physics (no speed up) and 0.5 rad/sec velocity in each servomotor.

### Visualize training data

Go to the /tmp folder indicated at the beginning of the execution, which in this casi is */tmp/ros_rl2/MARACollision-v0/ppo2_mlp*. Now you just have to execute Tensorboard and open the link it will provide in your web browser. You will find many useful graphs the like reward (eprewmean) plot there.

```sh
cd /tmp/ros_rl2/MARACollision-v0/ppo2_mlp
tensorboard --logdir=$(pwd)
```
