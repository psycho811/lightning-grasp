import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
if repo_root.as_posix() not in sys.path:
    sys.path.insert(0, repo_root.as_posix())

from lygra.robot import build_robot
from lygra.utils.robot_visualizer import RobotVisualizer
from lygra.utils.vis_utils import get_box_lineset_visual
import numpy as np 
import argparse


def get_wuji_joint_configuration(robot):
    joint_pos = {
        "right_finger1_joint1": 1.27,
        "right_finger1_joint2": -0.13,
        "right_finger1_joint3": 0.18,
        "right_finger1_joint4": 0.22,
        "right_finger2_joint1": 0.74,
        "right_finger2_joint2": 0.07,
        "right_finger2_joint3": 0.72,
        "right_finger2_joint4": 0.14,
        "right_finger3_joint1": 0.66,
        "right_finger3_joint2": -0.01,
        "right_finger3_joint3": 0.53,
        "right_finger3_joint4": 0.32,
        "right_finger4_joint1": 0.81,
        "right_finger4_joint2": -0.06,
        "right_finger4_joint3": 0.41,
        "right_finger4_joint4": 0.23,
        "right_finger5_joint1": 1.24,
        "right_finger5_joint2": -0.06,
        "right_finger5_joint3": 0.21,
        "right_finger5_joint4": 0.23,
    }

    return np.array([joint_pos[joint] for joint in robot.get_active_joints()], dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', default='allegro', type=str)
    args = parser.parse_args()

    robot_name = args.robot
    robot = build_robot(robot_name)

    box_min, box_max = robot.get_canonical_space()

    print("min", box_min)
    print("max", box_max)

    robot_tree = robot.get_kinematics_tree()
    
    lower, upper = robot_tree.get_active_joint_limit()
    q = (lower + upper) / 2
    if robot_name == "wuji":
        q = get_wuji_joint_configuration(robot)

    box = get_box_lineset_visual(box_min, box_max)

    viewer = RobotVisualizer(robot)
    robot_mesh = viewer.get_mesh_fk(q, visual=False)
    viewer.show(robot_mesh + [box])
