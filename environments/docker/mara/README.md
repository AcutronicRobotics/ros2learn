# modular articulated arm - mara

#### Adding the gripper

clone this repo:

```
git clone https://github.com/erlerobot/modular_articulated_arm
```

## MoveIt! with a simulated mara

#### Gazebo

```
roslaunch mara_gazebo mara.launch
roslaunch mara_gazebo mara_demo_camera_top.launch
roslaunch mara_gazebo mara_demo_camera_side.launch
```

#### MoveIT

```
roslaunch mara_moveit_config mara_moveit_planning_execution.launch sim:=true
```

#### RVIZ
```
roslaunch mara_moveit_config moveit_rviz.launch config:=true
```
