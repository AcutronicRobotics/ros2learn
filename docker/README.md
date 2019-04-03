# gym-gazebo2 Docker container usage

## Pull container from docker hub

WIP

## Build the container

```shell
cd ~ && git clone https://github.com/AcutronicRobotics/ros2learn
cd ~/ros2learn/docker
docker build -t r2l .
```

## Run the container

```shell
cd ~/ros2learn/docker
docker rm r2l || true && docker run -it --name=r2l -h ros2learn -v `pwd`:/tmp r2l
cp -r /root/ros2_mara_ws /tmp #Inside the docker container, used to load visual models
```

## Run a training script

```shell 
# inside the docker container
cd ~/ros2learn/experiments/examples/MARA
python3 train_ppo2_mlp.py
```

### Launch gzclient (GUI)

Make sure you have gazebo already installed in your main Ubuntu system and you are in the same path from which you executed the `docker run` command. If you are already running the simulation in the default port, you can access the visual interface the following way opening a new terminal:
```shell
# Do not use -g --gzclient flag
cd ~ && git clone https://github.com/AcutronicRobotics/ros2learn
cd ~/ros2learn/docker
sh gzclient.sh
```

### Visualize tensorboard

From your main OS, launch tensorboard pointing it to the files shared from the docker container. You can use the absolute path to any specific file or folder available in that directory.

```shell
cd ~ && git clone https://github.com/AcutronicRobotics/ros2learn
cd ~/ros2learn/docker
sudo tensorboard --logdir=`pwd`
```
