"""Simulateur minimal : le même modèle cinématique que l'environnement d'entraînement, exposé en ROS2.

Permet de faire tourner `policy_node` en boucle fermée sans Gazebo. Enchaîne des
scénarios tirés des graines de test et publie l'issue de chaque épisode.

Publie : /odom, /tf (odom -> base_link), /goal_pose, /obstacles, /path, /episode_status
Écoute : /cmd_vel
Paramètres : seed [1000000], episodes [0 = infini], rate_hz [10.0]
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from env.config import EnvConfig
from env.robot_env import NavigationEnv, integrate_unicycle
from env.scenario import surface_distances
from robot_learn_ros.bridge import quaternion_from_yaw

FRAME = "odom"


class SimNode(Node):
    def __init__(self):
        super().__init__("sim_node")
        self.declare_parameter("seed", 1_000_000)
        self.declare_parameter("episodes", 0)
        self.declare_parameter("rate_hz", 10.0)
        self.cfg = EnvConfig()
        self.env = NavigationEnv(self.cfg)
        self.seed = self.get_parameter("seed").value
        self.max_episodes = self.get_parameter("episodes").value
        self.dt = 1.0 / self.get_parameter("rate_hz").value

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "goal_pose", latched)
        self.obstacles_pub = self.create_publisher(MarkerArray, "obstacles", latched)
        self.path_pub = self.create_publisher(Path, "path", 10)
        self.status_pub = self.create_publisher(String, "episode_status", latched)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)

        self.cmd = (0.0, 0.0)
        self.last_cmd = None
        self.episode = 0
        self.pause_ticks = 0
        self.results: list[str] = []
        self.start_episode()
        self.create_timer(self.dt, self.on_timer)

    # --- épisodes ----------------------------------------------------------------------

    def start_episode(self):
        self.env.reset(seed=self.seed + self.episode)
        sc = self.env.scenario
        self.pose = np.array([*sc.start, sc.start_theta])
        self.steps = 0
        self.path = Path()
        self.path.header.frame_id = FRAME
        self.publish_scenario()
        self.get_logger().info(
            f"Épisode {self.episode} (graine {self.seed + self.episode}) : "
            f"départ ({sc.start[0]:.1f}, {sc.start[1]:.1f}) -> cible ({sc.goal[0]:.1f}, {sc.goal[1]:.1f}), "
            f"{len(sc.obstacles)} obstacles"
        )

    def end_episode(self, outcome: str):
        self.results.append(outcome)
        rate = self.results.count("succès") / len(self.results)
        msg = f"Épisode {self.episode} : {outcome} en {self.steps * self.dt:.1f} s (succès cumulé {rate:.0%})"
        self.get_logger().info(msg)
        self.status_pub.publish(String(data=msg))
        self.episode += 1
        if self.max_episodes and self.episode >= self.max_episodes:
            self.get_logger().info(f"Terminé : {self.results.count('succès')}/{len(self.results)} succès")
            raise SystemExit
        self.pause_ticks = int(1.0 / self.dt)
        self.start_episode()

    # --- callbacks ---------------------------------------------------------------------

    def on_cmd(self, msg: Twist):
        v = float(np.clip(msg.linear.x, 0.0, self.cfg.v_max))
        w = float(np.clip(msg.angular.z, -self.cfg.omega_max, self.cfg.omega_max))
        self.cmd = (v, w)
        self.last_cmd = self.get_clock().now()

    def on_timer(self):
        if self.pause_ticks > 0:
            self.pause_ticks -= 1
            self.publish_state()
            return
        fresh = self.last_cmd is not None and (self.get_clock().now() - self.last_cmd).nanoseconds * 1e-9 < 0.5
        v, w = self.cmd if fresh else (0.0, 0.0)
        self.pose = integrate_unicycle(self.pose, v, w, self.dt)
        self.steps += 1
        self.publish_state()

        sc, cfg = self.env.scenario, self.cfg
        clearance = surface_distances(self.pose[:2], sc.obstacles, cfg.robot_radius)
        lo, hi = -cfg.boundary_margin, cfg.arena_size + cfg.boundary_margin
        if len(clearance) and clearance.min() <= 0:
            self.end_episode("collision")
        elif np.linalg.norm(sc.goal - self.pose[:2]) <= cfg.goal_radius:
            self.end_episode("succès")
        elif not (lo <= self.pose[0] <= hi and lo <= self.pose[1] <= hi):
            self.end_episode("sortie")
        elif self.steps >= cfg.max_steps:
            self.end_episode("timeout")

    # --- publication -------------------------------------------------------------------

    def publish_state(self):
        now = self.get_clock().now().to_msg()
        x, y, yaw = self.pose
        qx, qy, qz, qw = quaternion_from_yaw(yaw)

        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = now, FRAME, "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = float(x), float(y)
        o = odom.pose.pose.orientation
        o.x, o.y, o.z, o.w = qx, qy, qz, qw
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = self.cmd
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = now, FRAME, "base_link"
        tf.transform.translation.x, tf.transform.translation.y = float(x), float(y)
        r = tf.transform.rotation
        r.x, r.y, r.z, r.w = qx, qy, qz, qw
        self.tf.sendTransform(tf)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.path.poses.append(pose)
        self.path.header.stamp = now
        self.path_pub.publish(self.path)

    def publish_scenario(self):
        sc = self.env.scenario
        now = self.get_clock().now().to_msg()
        goal = PoseStamped()
        goal.header.stamp, goal.header.frame_id = now, FRAME
        goal.pose.position.x, goal.pose.position.y = float(sc.goal[0]), float(sc.goal[1])
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for i, (ox, oy, radius) in enumerate(sc.obstacles):
            m = Marker()
            m.header.stamp, m.header.frame_id = now, FRAME
            m.ns, m.id, m.type, m.action = "obstacles", i, Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(ox), float(oy), 0.25
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = float(2 * radius)
            m.scale.z = 0.5
            m.color.r, m.color.g, m.color.b, m.color.a = 0.76, 0.76, 0.72, 1.0
            markers.markers.append(m)
        self.obstacles_pub.publish(markers)


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
