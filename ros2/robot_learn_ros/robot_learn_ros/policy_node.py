"""Nœud ROS2 qui exécute la politique apprise.

Entrées :
    /odom        nav_msgs/Odometry                 pose du robot
    /goal_pose   geometry_msgs/PoseStamped         cible
    /obstacles   visualization_msgs/MarkerArray    obstacles circulaires (CYLINDER : centre, scale.x = diamètre)
Sortie :
    /cmd_vel     geometry_msgs/Twist               (linear.x = v, angular.z = omega)

Sur un vrai robot, /obstacles viendrait d'un module de perception (lidar + clustering) ;
ici, `sim_node` les publie directement.

Paramètres :
    policy          "expert" ou chemin d'un checkpoint (.pt)   [défaut : checkpoints/ppo.pt]
    rate_hz         fréquence de contrôle, égale à 1/dt de l'entraînement   [10.0]
    goal_tolerance  distance d'arrêt (m)   [0.3]
    odom_timeout    arrêt de sécurité si l'odométrie est plus vieille (s)   [0.5]
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from env.config import EnvConfig
from robot_learn_ros.bridge import PolicyController, load_policy, yaw_from_quaternion


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        self.declare_parameter("policy", "checkpoints/ppo.pt")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("goal_tolerance", 0.3)
        self.declare_parameter("odom_timeout", 0.5)

        cfg = EnvConfig()
        spec = self.get_parameter("policy").value
        self.controller = PolicyController(load_policy(spec, cfg), cfg, self.get_parameter("goal_tolerance").value)
        self.odom_timeout = self.get_parameter("odom_timeout").value
        self.get_logger().info(f"Politique chargée : {spec}")

        self.pose = None
        self.last_odom = None
        self.goal = None
        self.obstacles = np.zeros((0, 3))
        self.reached = False

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, "odom", self.on_odom, 10)
        self.create_subscription(PoseStamped, "goal_pose", self.on_goal, latched)
        self.create_subscription(MarkerArray, "obstacles", self.on_obstacles, latched)
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.on_timer)

    def on_odom(self, msg: Odometry):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = np.array([p.x, p.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)])
        self.last_odom = self.get_clock().now()

    def on_goal(self, msg: PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.reached = False
        self.get_logger().info(f"Nouvelle cible : ({self.goal[0]:.2f}, {self.goal[1]:.2f})")

    def on_obstacles(self, msg: MarkerArray):
        cylinders = [m for m in msg.markers if m.type == Marker.CYLINDER and m.action == Marker.ADD]
        self.obstacles = np.array(
            [[m.pose.position.x, m.pose.position.y, m.scale.x / 2.0] for m in cylinders], dtype=np.float64
        ).reshape(-1, 3)

    def on_timer(self):
        cmd = Twist()
        stale = self.last_odom is None or (self.get_clock().now() - self.last_odom).nanoseconds * 1e-9 > self.odom_timeout
        if self.pose is None or self.goal is None or stale:
            self.cmd_pub.publish(cmd)  # arrêt tant qu'on n'a pas d'information fraîche
            return
        v, omega, reached = self.controller.command(self.pose, self.goal, self.obstacles)
        if reached and not self.reached:
            self.get_logger().info("Cible atteinte")
        self.reached = reached
        cmd.linear.x, cmd.angular.z = v, omega
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():  # sur SIGINT, rclpy a déjà invalidé le contexte
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
