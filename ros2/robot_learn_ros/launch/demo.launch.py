"""Démo en boucle fermée : simulateur + politique apprise.

    ros2 launch robot_learn_ros demo.launch.py policy:=/robot-learn/checkpoints/ppo.pt episodes:=10

Quand `episodes` > 0, tout s'arrête après le dernier épisode du simulateur.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim = Node(
        package="robot_learn_ros",
        executable="sim_node",
        name="sim_node",
        output="screen",
        parameters=[{"episodes": LaunchConfiguration("episodes"), "seed": LaunchConfiguration("seed")}],
    )
    policy = Node(
        package="robot_learn_ros",
        executable="policy_node",
        name="policy_node",
        output="screen",
        parameters=[{"policy": LaunchConfiguration("policy")}],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("policy", default_value="/robot-learn/checkpoints/ppo.pt"),
            DeclareLaunchArgument("episodes", default_value="0", description="0 = boucle infinie"),
            DeclareLaunchArgument("seed", default_value="1000000"),
            sim,
            policy,
            RegisterEventHandler(OnProcessExit(target_action=sim, on_exit=[EmitEvent(event=Shutdown())])),
        ]
    )
