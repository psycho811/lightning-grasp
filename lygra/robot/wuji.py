# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.base import RobotInterface
import numpy as np 


class Wuji(RobotInterface):
    def get_white_list_pairs(self):
        palm_links = ["right_palm_link"]
        finger_links = [
            "right_finger1_link1",
            "right_finger1_link2",
            "right_finger1_link3",
            "right_finger1_link4_back",
            "right_finger1_link4_front",
            "right_finger2_link1",
            "right_finger2_link2",
            "right_finger2_link3",
            "right_finger2_link4_back",
            "right_finger2_link4_front",
            "right_finger3_link1",
            "right_finger3_link2",
            "right_finger3_link3",
            "right_finger3_link4_back",
            "right_finger3_link4_front",
            "right_finger4_link1",
            "right_finger4_link2",
            "right_finger4_link3",
            "right_finger4_link4_back",
            "right_finger4_link4_front",
            "right_finger5_link1",
            "right_finger5_link2",
            "right_finger5_link3",
            "right_finger5_link4_back",
            "right_finger5_link4_front",
        ]
        palm_pairs = [[palm, link] for palm in palm_links for link in finger_links]
        split_adjacency_pairs = [
            [f"right_finger{i}_link3", f"right_finger{i}_link4_front"]
            for i in range(1, 6)
        ]
        excluded_collision_pairs = [
            ["right_palm_link", "right_finger1_link1"],
            ["right_palm_link", "right_finger1_link2"],
            ["right_palm_link", "right_finger2_link1"],
            ["right_palm_link", "right_finger2_link2"],
            ["right_palm_link", "right_finger3_link1"],
            ["right_palm_link", "right_finger3_link2"],
            ["right_palm_link", "right_finger4_link1"],
            ["right_palm_link", "right_finger4_link2"],
            ["right_palm_link", "right_finger5_link1"],
            ["right_palm_link", "right_finger5_link2"],
            ["right_finger1_link1", "right_finger1_link2"],
            ["right_finger1_link1", "right_finger1_link3"],
            ["right_finger1_link1", "right_finger1_link4_back"],
            ["right_finger1_link1", "right_finger1_link4_front"],
            ["right_finger1_link1", "right_finger2_link1"],
            ["right_finger1_link1", "right_finger2_link2"],
            ["right_finger1_link1", "right_finger2_link3"],
            ["right_finger1_link1", "right_finger2_link4_back"],
            ["right_finger1_link1", "right_finger2_link4_front"],
            ["right_finger1_link1", "right_finger3_link1"],
            ["right_finger1_link1", "right_finger3_link2"],
            ["right_finger1_link1", "right_finger3_link3"],
            ["right_finger1_link1", "right_finger3_link4_back"],
            ["right_finger1_link1", "right_finger3_link4_front"],
            ["right_finger1_link1", "right_finger4_link1"],
            ["right_finger1_link1", "right_finger4_link2"],
            ["right_finger1_link1", "right_finger4_link3"],
            ["right_finger1_link1", "right_finger4_link4_back"],
            ["right_finger1_link1", "right_finger4_link4_front"],
            ["right_finger1_link1", "right_finger5_link1"],
            ["right_finger1_link1", "right_finger5_link2"],
            ["right_finger1_link1", "right_finger5_link3"],
            ["right_finger1_link1", "right_finger5_link4_back"],
            ["right_finger1_link1", "right_finger5_link4_front"],
            ["right_finger2_link1", "right_finger1_link2"],
            ["right_finger2_link1", "right_finger1_link3"],
            ["right_finger2_link1", "right_finger1_link4_back"],
            ["right_finger2_link1", "right_finger1_link4_front"],
            ["right_finger2_link1", "right_finger2_link2"],
            ["right_finger2_link1", "right_finger2_link3"],
            ["right_finger2_link1", "right_finger2_link4_back"],
            ["right_finger2_link1", "right_finger2_link4_front"],
            ["right_finger2_link1", "right_finger3_link1"],
            ["right_finger2_link1", "right_finger3_link2"],
            ["right_finger2_link1", "right_finger3_link3"],
            ["right_finger2_link1", "right_finger3_link4_back"],
            ["right_finger2_link1", "right_finger3_link4_front"],
            ["right_finger2_link1", "right_finger4_link1"],
            ["right_finger2_link1", "right_finger4_link2"],
            ["right_finger2_link1", "right_finger4_link3"],
            ["right_finger2_link1", "right_finger4_link4_back"],
            ["right_finger2_link1", "right_finger4_link4_front"],
            ["right_finger2_link1", "right_finger5_link1"],
            ["right_finger2_link1", "right_finger5_link2"],
            ["right_finger2_link1", "right_finger5_link3"],
            ["right_finger2_link1", "right_finger5_link4_back"],
            ["right_finger2_link1", "right_finger5_link4_front"],
            ["right_finger3_link1", "right_finger1_link2"],
            ["right_finger3_link1", "right_finger1_link3"],
            ["right_finger3_link1", "right_finger1_link4_back"],
            ["right_finger3_link1", "right_finger1_link4_front"],
            ["right_finger3_link1", "right_finger2_link2"],
            ["right_finger3_link1", "right_finger2_link3"],
            ["right_finger3_link1", "right_finger2_link4_back"],
            ["right_finger3_link1", "right_finger2_link4_front"],
            ["right_finger3_link1", "right_finger3_link2"],
            ["right_finger3_link1", "right_finger3_link3"],
            ["right_finger3_link1", "right_finger3_link4_back"],
            ["right_finger3_link1", "right_finger3_link4_front"],
            ["right_finger3_link1", "right_finger4_link1"],
            ["right_finger3_link1", "right_finger4_link2"],
            ["right_finger3_link1", "right_finger4_link3"],
            ["right_finger3_link1", "right_finger4_link4_back"],
            ["right_finger3_link1", "right_finger4_link4_front"],
            ["right_finger3_link1", "right_finger5_link1"],
            ["right_finger3_link1", "right_finger5_link2"],
            ["right_finger3_link1", "right_finger5_link3"],
            ["right_finger3_link1", "right_finger5_link4_back"],
            ["right_finger3_link1", "right_finger5_link4_front"],
            ["right_finger4_link1", "right_finger1_link2"],
            ["right_finger4_link1", "right_finger1_link3"],
            ["right_finger4_link1", "right_finger1_link4_back"],
            ["right_finger4_link1", "right_finger1_link4_front"],
            ["right_finger4_link1", "right_finger2_link2"],
            ["right_finger4_link1", "right_finger2_link3"],
            ["right_finger4_link1", "right_finger2_link4_back"],
            ["right_finger4_link1", "right_finger2_link4_front"],
            ["right_finger4_link1", "right_finger3_link2"],
            ["right_finger4_link1", "right_finger3_link3"],
            ["right_finger4_link1", "right_finger3_link4_back"],
            ["right_finger4_link1", "right_finger3_link4_front"],
            ["right_finger4_link1", "right_finger4_link2"],
            ["right_finger4_link1", "right_finger4_link3"],
            ["right_finger4_link1", "right_finger4_link4_back"],
            ["right_finger4_link1", "right_finger4_link4_front"],
            ["right_finger4_link1", "right_finger5_link1"],
            ["right_finger4_link1", "right_finger5_link2"],
            ["right_finger4_link1", "right_finger5_link3"],
            ["right_finger4_link1", "right_finger5_link4_back"],
            ["right_finger4_link1", "right_finger5_link4_front"],
            ["right_finger5_link1", "right_finger1_link2"],
            ["right_finger5_link1", "right_finger1_link3"],
            ["right_finger5_link1", "right_finger1_link4_back"],
            ["right_finger5_link1", "right_finger1_link4_front"],
            ["right_finger5_link1", "right_finger2_link2"],
            ["right_finger5_link1", "right_finger2_link3"],
            ["right_finger5_link1", "right_finger2_link4_back"],
            ["right_finger5_link1", "right_finger2_link4_front"],
            ["right_finger5_link1", "right_finger3_link2"],
            ["right_finger5_link1", "right_finger3_link3"],
            ["right_finger5_link1", "right_finger3_link4_back"],
            ["right_finger5_link1", "right_finger3_link4_front"],
            ["right_finger5_link1", "right_finger4_link2"],
            ["right_finger5_link1", "right_finger4_link3"],
            ["right_finger5_link1", "right_finger4_link4_back"],
            ["right_finger5_link1", "right_finger4_link4_front"],
            ["right_finger5_link1", "right_finger5_link2"],
            ["right_finger5_link1", "right_finger5_link3"],
            ["right_finger5_link1", "right_finger5_link4_back"],
            ["right_finger5_link1", "right_finger5_link4_front"],
            ["right_finger1_link2", "right_finger1_link3"],
            ["right_finger1_link2", "right_finger1_link4_back"],
            ["right_finger1_link2", "right_finger1_link4_front"],
            ["right_finger1_link2", "right_finger2_link3"],
            ["right_finger1_link2", "right_finger2_link4_back"],
            ["right_finger1_link2", "right_finger2_link4_front"],
            ["right_finger1_link2", "right_finger3_link3"],
            ["right_finger1_link2", "right_finger3_link4_back"],
            ["right_finger1_link2", "right_finger3_link4_front"],
            ["right_finger1_link2", "right_finger4_link3"],
            ["right_finger1_link2", "right_finger4_link4_back"],
            ["right_finger1_link2", "right_finger4_link4_front"],
            ["right_finger1_link2", "right_finger5_link3"],
            ["right_finger1_link2", "right_finger5_link4_back"],
            ["right_finger1_link2", "right_finger5_link4_front"],
            ["right_finger2_link2", "right_finger1_link3"],
            ["right_finger2_link2", "right_finger1_link4_back"],
            ["right_finger2_link2", "right_finger1_link4_front"],
            ["right_finger2_link2", "right_finger2_link3"],
            ["right_finger2_link2", "right_finger2_link4_back"],
            ["right_finger2_link2", "right_finger2_link4_front"],
            ["right_finger2_link2", "right_finger3_link3"],
            ["right_finger2_link2", "right_finger3_link4_back"],
            ["right_finger2_link2", "right_finger3_link4_front"],
            ["right_finger2_link2", "right_finger4_link3"],
            ["right_finger2_link2", "right_finger4_link4_back"],
            ["right_finger2_link2", "right_finger4_link4_front"],
            ["right_finger2_link2", "right_finger5_link3"],
            ["right_finger2_link2", "right_finger5_link4_back"],
            ["right_finger2_link2", "right_finger5_link4_front"],
            ["right_finger3_link2", "right_finger1_link3"],
            ["right_finger3_link2", "right_finger1_link4_back"],
            ["right_finger3_link2", "right_finger1_link4_front"],
            ["right_finger3_link2", "right_finger2_link3"],
            ["right_finger3_link2", "right_finger2_link4_back"],
            ["right_finger3_link2", "right_finger2_link4_front"],
            ["right_finger3_link2", "right_finger3_link3"],
            ["right_finger3_link2", "right_finger3_link4_back"],
            ["right_finger3_link2", "right_finger3_link4_front"],
            ["right_finger3_link2", "right_finger4_link3"],
            ["right_finger3_link2", "right_finger4_link4_back"],
            ["right_finger3_link2", "right_finger4_link4_front"],
            ["right_finger3_link2", "right_finger5_link3"],
            ["right_finger3_link2", "right_finger5_link4_back"],
            ["right_finger3_link2", "right_finger5_link4_front"],
            ["right_finger4_link2", "right_finger1_link3"],
            ["right_finger4_link2", "right_finger1_link4_back"],
            ["right_finger4_link2", "right_finger1_link4_front"],
            ["right_finger4_link2", "right_finger2_link3"],
            ["right_finger4_link2", "right_finger2_link4_back"],
            ["right_finger4_link2", "right_finger2_link4_front"],
            ["right_finger4_link2", "right_finger3_link3"],
            ["right_finger4_link2", "right_finger3_link4_back"],
            ["right_finger4_link2", "right_finger3_link4_front"],
            ["right_finger4_link2", "right_finger4_link3"],
            ["right_finger4_link2", "right_finger4_link4_back"],
            ["right_finger4_link2", "right_finger4_link4_front"],
            ["right_finger4_link2", "right_finger5_link3"],
            ["right_finger4_link2", "right_finger5_link4_back"],
            ["right_finger4_link2", "right_finger5_link4_front"],
            ["right_finger5_link2", "right_finger1_link3"],
            ["right_finger5_link2", "right_finger1_link4_back"],
            ["right_finger5_link2", "right_finger1_link4_front"],
            ["right_finger5_link2", "right_finger2_link3"],
            ["right_finger5_link2", "right_finger2_link4_back"],
            ["right_finger5_link2", "right_finger2_link4_front"],
            ["right_finger5_link2", "right_finger3_link3"],
            ["right_finger5_link2", "right_finger3_link4_back"],
            ["right_finger5_link2", "right_finger3_link4_front"],
            ["right_finger5_link2", "right_finger4_link3"],
            ["right_finger5_link2", "right_finger4_link4_back"],
            ["right_finger5_link2", "right_finger4_link4_front"],
            ["right_finger5_link2", "right_finger5_link3"],
            ["right_finger5_link2", "right_finger5_link4_back"],
            ["right_finger5_link2", "right_finger5_link4_front"],
        ]

        pairs = []
        seen = set()
        for pair in palm_pairs + split_adjacency_pairs + excluded_collision_pairs:
            key = tuple(pair)
            reverse_key = (pair[1], pair[0])
            if key in seen or reverse_key in seen:
                continue
            pairs.append(pair)
            seen.add(key)
        return pairs

    def get_canonical_space(self):
        """
            Canonical Space for placing objects (we randomly select an object surface point and drag it into this box)
        """
        # 生成苹果比较正常
        # box_min = np.array([0.065, -0.02, 0.05], dtype=np.float32)
        # box_max = np.array([0.115, 0.02, 0.1], dtype=np.float32)
        box_min = np.array([0.06, -0.02, 0.07], dtype=np.float32)
        box_max = np.array([0.11, 0.02, 0.13], dtype=np.float32)
        return box_min, box_max 

    def get_default_urdf_path(self):
        """
            Default path to your robot URDF
        """
        return './assets/hand/wuji/mj_lab_wuji/urdf/right_mjlab_finger_split.urdf'

    def get_contact_field_config(self):
        """
            Specify which links should make contact:
            - Static Links: e.g. palms.
            - Movable Links: e.g. fingers.

            For movable links, you can restrict contact normal directions using the format below.

        """

        config = {
            "type": "v1",
            "movable_link": {},
            "static_link": {}
        }

        for link in [ 
           "right_finger1_link4_front",
            "right_finger2_link4_front",
            "right_finger3_link4_front",
            "right_finger4_link4_front",
            "right_finger5_link4_front",
        ]:
            config["movable_link"][link] = {
                "disabled_normal": [
                    (np.array([0.0, 0.0, -1.0]), 3.1415926 * 0.49999),
                    # (np.array([1.0, 0.0,  0.0]), 0.5),
                    # (np.array([0.0, 0.0, -1.0]), 3.1415926/2),
                    # (np.array([0.0, 0.0, 1.0]), 3.1415926/2),
                    # (np.array([0.0, 1.0, 0.0]), 3.1415926/4)
                ]
            }

        config["static_link"]["right_palm_link"] = {
            "allowed_normal": [
                (np.array([[1.0, 0.0, 0.0]]), 3.1415926 * 0.25)
            ]
        }
        return config


    def get_active_joints(self):
        """
            Specify the active joints (dofs).
            Our system will return active joint values in this order.
        """
        return [
            "right_finger1_joint1",
            "right_finger2_joint1",
            "right_finger3_joint1",
            "right_finger4_joint1",
            "right_finger5_joint1",

            "right_finger1_joint2",
            "right_finger2_joint2",
            "right_finger3_joint2",
            "right_finger4_joint2",
            "right_finger5_joint2",

            "right_finger1_joint3",
            "right_finger2_joint3",
            "right_finger3_joint3",
            "right_finger4_joint3",
            "right_finger5_joint3",

            "right_finger1_joint4",
            "right_finger2_joint4",
            "right_finger3_joint4",
            "right_finger4_joint4",
            "right_finger5_joint4",
        ]

    def get_base_link(self):
        return "right_palm_link"

    def get_static_links(self):
        return ["right_palm_link"]

    def get_mesh_scale(self):
        """
            Your robot mesh might be rescaled in your URDF, specify it here.
            (will be removed in the future.)
        """
        return 1.0
