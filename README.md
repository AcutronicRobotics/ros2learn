<a href="http://www.acutronicrobotics.com"><img src="https://github.com/erlerobot/gym-gazebo2/blob/master/imgs/alr_logo.png" align="left" width="190"></a>

This repository contains a number of ROS and ROS 2.0 enabled Artificial Intelligence (AI)
and Reinforcement Learning (RL) [algorithms](algorithms/) that run in selected [environments](environments/).

The repository contains the following:
- [algorithms](algorithms/): techniques used for training and teaching robots.
- [environments](environments/): pre-built environments of interest to train selected robots.
- [experiments](experiments/): experiments and examples of the different utilities that this repository provides.

---

## Installation

Please refer to [Install.md](/Install.md).

## Usage

### Train an agent
You will find all available examples at */experiments/examples/*. Although the algorithms are complex, the way to execute them is really simple. For instance, if you want to train MARA robot using ppo2_mlp, you should execute the following command:

```sh
cd experiments/examples/MARA
python3 train_ppo2_mlp.py
```

Note that you can add the command line arguments provided by the environment, which in this case are provided by the gym-gazebo2 Env. Use `-h` to get all the available commands.

If you want to test your own trained neural networks, or train with different environment form gym-gazebo2, or play with the hyperparametes, you must update the values of the dictionary directly in the corresponding algorithm itself. For this example, we are using *ppo2_mlp* from *baselines* submodule, so you can edit the `mara_mpl()` function inside [baselines/ppo2/defaults.py](https://github.com/erlerobot/baselines/blob/8396ea2dc4d19cabb7478f6c3df0119660f0ab18/baselines/ppo2/defaults.py#L28-L53).

![Example Train](https://github.com/erlerobot/gym-gazebo2/blob/master/imgs/example_train.gif)

### Run a trained policy
Once you are done with the training, or if you want to test some specific checkpoint of it, you can run that using one of the running-scripts available. This time, to follow with the example, we are going to run a saved ppo2_mlp policy.

First, we will edit the already mentioned `mara_mpl()` dictionary, in particular the `trained_path` value, in [baselines/ppo2/defaults.py](https://github.com/erlerobot/baselines/blob/8396ea2dc4d19cabb7478f6c3df0119660f0ab18/baselines/ppo2/defaults.py#L53) to the checkpoint we want (checkpoints placed by default in /tmp/ros2learn). Now we are ready to launch the script.

Since we want to visualize it in real conditions, we are also going to set some flags:

```sh
cd experiments/examples/MARA
python3 run_ppo2_mlp.py -g -r -v 0.3
```

This will launch the simulation with the visual interface, real time physics (no speed up) and 0.3 rad/sec velocity in each servomotor.

![Example Run](https://github.com/erlerobot/gym-gazebo2/blob/master/imgs/example_run.gif)

### Visualize training data on tensorboard

The logdir path will change according to the used environment ID and the used algorithm in training.
Now you just have to execute Tensorboard and open the link it will provide (or localhost:port_number) in your web browser. You will find many useful graphs like the reward (eprewmean) plotted there.

You can also set a specific port number in case you want to visualize more than one tensorboard file from different paths.

```sh
tensorboard --logdir=/tmp/ros2learn/MARACollision-v0/ppo2_mlp --port 8008
```
![Example Tensorboard](https://github.com/erlerobot/gym-gazebo2/blob/master/imgs/example_tensorboard.gif)

