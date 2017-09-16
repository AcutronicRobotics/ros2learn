FROM gazebo:gzserver8

# ROS (Kinetic)
# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2 \
&& rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO kinetic
RUN apt-get update && apt-get install -y \
    ros-kinetic-ros-core=1.3.1-0* \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl git
RUN apt-get install -y libsdformat5 libignition-math3 libgazebo8
RUN apt-get install -y libgazebo8-dev

# setup entrypoints
COPY ./entrypoint_pendulum.sh /
COPY ./entrypoint_scara.sh /
# Copy scara simulation files
COPY ./scara_e1 /ros_ws/src

# Clone dependencies for the simulation
RUN git clone https://github.com/ros-simulation/gazebo_ros_pkgs /ros_ws/src/gazebo_ros_pkgs
# Compile the ROS workspace with all the corresponding dependencies
RUN apt-get install -y ros-kinetic-nodelet
RUN apt-get install -y ros-kinetic-angles
RUN apt-get install -y ros-kinetic-urdf
RUN apt-get install -y ros-kinetic-tf
RUN apt-get install -y ros-kinetic-dynamic-reconfigure
RUN apt-get install -y ros-kinetic-image-transport
RUN apt-get install -y ros-kinetic-cv-bridge
RUN apt-get install -y ros-kinetic-polled-camera
RUN apt-get install -y ros-kinetic-diagnostic-updater
RUN apt-get install -y ros-kinetic-camera-info-manager
RUN apt-get install -y ros-kinetic-control-toolbox
RUN apt-get install -y ros-kinetic-controller-manager
RUN apt-get install -y ros-kinetic-transmission-interface
RUN apt-get install -y ros-kinetic-joint-limits-interface
RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash; cd /ros_ws/; catkin_make_isolated --install"

RUN apt-get install -y ros-kinetic-xacro
RUN apt-get install -y ros-kinetic-robot-state-publisher
RUN apt-get install -y ros-kinetic-joint-state-controller
RUN apt-get install -y ros-kinetic-joint-trajectory-controller
RUN apt-get install -y ros-kinetic-position-controllers

# ENTRYPOINT ["/entrypoint.sh"]
# CMD [ "gzserver", "my-gazebo-app-args" ]
CMD ["bash"]
