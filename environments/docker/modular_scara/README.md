Modular SCARA environment built out of H-ROS modular components

### Build the docker container
```
docker build -t scaraworld .
```

### Launch it, get a bash shell on it
```
docker run -it -v="/tmp/.gazebo/:/root/.gazebo/" --name test scaraworld
```

### Launch the simulation
```
roslaunch scara_e1_gazebo scara_e1_4joints.launch
```
TODO: No screen available, need to find the way to use gzserver instead


### Visualize it remotely (outside of the container)
```
export GAZEBO_MASTER_IP=$(sudo docker inspect --format '{{ .NetworkSettings.IPAddress }}' test)
export GAZEBO_MASTER_URI=$GAZEBO_MASTER_IP:11345
gzclient --verbose
```
