# gym-gazebo2 Docker container usage

## Pull container from docker hub

WIP

## Install Nvidia docker container
Follow the instructions provided [here](https://github.com/NVIDIA/nvidia-docker)

## Build the container

```shell
cd ~ && git clone https://github.com/AcutronicRobotics/ros2learn
cd ~/ros2learn/docker
docker build -t r2l .
```
#### First run, gazebo model setup.

```shell
cd ~/ros2learn/docker
docker rm r2l || true && docker run -it --name=r2l -h ros2learn -v `pwd`/tmp:/tmp/ros2learn r2l

#Inside the docker container, used to load visual models
cp -r /root/ros2_mara_ws /tmp/ros2learn
```

Outside the docker container, copy `ros2_mara_ws` folder to a permanent location so that you don't need to repeat this process in the future.
```
cd ~/ros2learn/docker
cp -r tmp/ros2_mara_ws .
```

## Run the container

```shell
cd ~/ros2learn/docker
# Clean the existing $PWD/tmp directory. You might need `sudo`.
rm -rf `pwd`/tmp/*
# Run a new r2l container
docker rm r2l || true && docker run -it --name=r2l -h ros2learn -v `pwd`/tmp:/tmp/ros2learn r2l
```

### Development/Research mode
You can install new software such as file editors (e.g. `apt install nano`), which would be useful if you are trying to find the optimal parameters for a network for instance.

```shell
# Inside Docker
apt update
apt install nano
```

Make sure you save the state of your docker container before exiting it by opening a new terminal and executing:

```shell
# Get the CONTAINER ID from the IMAGE called r2l.
docker ps
# Commit changes to the r2l container.
docker commit XXXXXXX r2l
```

Example:
```shell
$ docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
b0d8de35f133        r2l                 "bash"              2 minutes ago       Up 2 minutes        11597/tcp           wizardly_lamarr

$ docker commit b0d8de35f133 r2l
```

Next time you want to run the container you will need to launch the existing one:

```shell
docker run -it r2l
```

## Run a training script

```shell
# inside the docker container
cd ~/ros2learn/experiments/examples/MARA
python3 train_ppo2_mlp.py
```

#### Launch gzclient (GUI)

Make sure you have gazebo already installed in your main Ubuntu system and you are in the same path from which you executed the `docker run` command. If you are already running the simulation in the default port, you can access the visual interface the following way opening a new terminal:
```shell
# Do not use -g --gzclient flag
cd ~ && git clone https://github.com/AcutronicRobotics/ros2learn
cd ~/ros2learn/docker
sh gzclient.sh
```

#### Visualize tensorboard

From your main OS, launch tensorboard pointing it to the files shared from the docker container. You can use the absolute path to any specific file or folder available in that directory.

```shell
cd ~ && git clone https://github.com/AcutronicRobotics/ros2learn
cd ~/ros2learn/docker
sudo tensorboard --logdir=`pwd`
```
