from glob import glob

from setuptools import setup

package_name = "robot_learn_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="leonardcrt",
    maintainer_email="leonardcourt4@gmail.com",
    description="Nœud ROS2 exposant une politique de navigation apprise.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_node = robot_learn_ros.policy_node:main",
            "sim_node = robot_learn_ros.sim_node:main",
        ],
    },
)
