### Get the code
```
git clone http://github.com/erlerobot/ros_rl
cd ros_rl
git submodule update --init --recursive # update the submodules
```
#### Useful info
- Update all submodules: `git pull --recurse-submodules && git submodule update --recursive --remote`
- Quick reference for submodules ([1](http://www.vogella.com/tutorials/GitSubmodules/article.html), [2](https://chrisjean.com/git-submodules-adding-using-removing-and-updating/), [3](https://git-scm.com/book/en/v2/Git-Tools-Submodules))

### Install each module
This repository contains various modules that need to be installed independently. You will find a link to the installation instructions of each repository below. Make sure you select the **correct path** (refer to this repository) to each module when following each installation guide, specially when installing modules via pip3 or sourcing provisioning scripts. 

- **Gym-gazebo2** [REQUIRED]. [Link](https://github.com/erlerobot/gym-gazebo-ros2/blob/master/INSTALL.md). 
  - Note 1: Skip [Baselines installation](https://github.com/erlerobot/gym-gazebo-ros2/blob/master/INSTALL.md#baselines), we will install it later from `ros_rl`. 
  - Note 2: In the [Install Gym-Gazebo2](https://github.com/erlerobot/gym-gazebo-ros2/blob/master/INSTALL.md#gym-gazebo2) step, you need to do the install the module already located inside `ros_rl`:

    ```bash
    cd ~/ros_rl/environments/gym-gazebo-ros2
    pip3 install -e .
    ```
  - Note 3: In the [Provisioning](https://github.com/erlerobot/gym-gazebo-ros2/blob/master/INSTALL.md#provisioning) step, you need to source the provisioning script located inside `ros_rl`:
  
    ```bash
    echo "source ~/ros_rl/environments/gym-gazebo-ros2/provision/mara_setup.sh" >> ~/.bashrc
    source ~/.bashrc
    ```
- **Baselines** [REQUIRED].

  ```bash
  cd cd ~/ros_rl/algorithms/baselines
  pip3 install -e .
    ```
- **Guided Policy Search** - deprecated [Optional]. [Link](https://github.com/erlerobot/gps/blob/01a4e108e1dc4fb126c0f2677662af0b15f3560a/README.md).
- **Meta-Learning Shared Hierarchies** [Optional]. [Link](https://github.com/erlerobot/mlsh/blob/a295419d9b7f3f7ae2f4846b9d8171da32e8e921/README.md). 
- **tensorflow-vgg** [Optional]. [Link](https://github.com/erlerobot/tensorflow-vgg/blob/aba1200b52b40042ba73c4a08a89f0e9cc095aef/README.md).