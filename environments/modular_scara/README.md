Modular SCARA environment built out of H-ROS modular components

### Build the docker container
```
docker build -t scaraworld .
```

### Launch it
```
docker run -it -v="/tmp/.gazebo/:/root/.gazebo/" --name test scaraworld
```

### Visualize it remotely (outside of the container)
```
export GAZEBO_MASTER_IP=$(sudo docker inspect --format '{{ .NetworkSettings.IPAddress }}' test)
export GAZEBO_MASTER_URI=$GAZEBO_MASTER_IP:11345
gzclient --verbose
```
