### Get the code
```
cd ~ && git clone https://github.com/AcutronicRobotics/ros2learn.git
cd ros2learn
git submodule update --init --recursive
```
#### Useful info
- Pull all the new commits: `git pull --recurse-submodules && git submodule update --recursive --remote`
- Quick reference for submodules ([1](http://www.vogella.com/tutorials/GitSubmodules/article.html), [2](https://chrisjean.com/git-submodules-adding-using-removing-and-updating/), [3](https://git-scm.com/book/en/v2/Git-Tools-Submodules))

### Install each module
This repository contains various modules that need to be installed independently:

- **gym-gazebo2**: is a toolkit for developing and comparing reinforcement learning algorithms using ROS 2 and Gazebo. Follow the [instructions](https://github.com/AcutronicRobotics/gym-gazebo2/blob/crystal/INSTALL.md) to install it.

- **baselines**: is a slightly adapted version of OpenAI's baselines repository to address robotics use cases with a set of high-quality implementations of reinforcement learning algorithms. To install it:

  ```sh
  cd ~/ros2learn/algorithms/baselines
  pip3 install -e .
  ```

### Dependent tools

```bash
pip3 install matplotlib
sudo apt install python3-tk
```
