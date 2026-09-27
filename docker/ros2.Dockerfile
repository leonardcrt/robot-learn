# Image ROS 2 Jazzy : simulateur + nœud de politique.
#   docker build -f docker/ros2.Dockerfile -t robot-learn-ros2 .
#   docker run --rm robot-learn-ros2                                      # 10 épisodes avec PPO
#   docker run --rm robot-learn-ros2 ros2 launch robot_learn_ros demo.launch.py \
#       policy:=/robot-learn/checkpoints/dagger.pt episodes:=20
FROM ros:jazzy-ros-base

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip python3-numpy \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install "gymnasium>=0.29"

# Le nœud n'a besoin que de l'environnement (construction de l'observation), du réseau et de l'expert.
WORKDIR /robot-learn
COPY env ./env
COPY expert ./expert
COPY models ./models
COPY checkpoints ./checkpoints
ENV PYTHONPATH=/robot-learn

COPY ros2/robot_learn_ros /ros2_ws/src/robot_learn_ros
RUN . /opt/ros/jazzy/setup.sh && cd /ros2_ws && colcon build --symlink-install

COPY docker/ros2_entrypoint.sh /ros2_entrypoint.sh
RUN chmod +x /ros2_entrypoint.sh
ENTRYPOINT ["/ros2_entrypoint.sh"]
CMD ["ros2", "launch", "robot_learn_ros", "demo.launch.py", "episodes:=10"]
