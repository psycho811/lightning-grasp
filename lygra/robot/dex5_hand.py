# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.base import RobotInterface
import numpy as np 


class Dex5_hand(RobotInterface):
    def get_canonical_space(self):
        """
            Canonical Space for placing objects (we randomly select an object surface point and drag it into this box)
        """
        box_min = np.array([0.04, -0.05, 0.04], dtype=np.float32)
        box_max = np.array([0.1, 0.06, 0.12], dtype=np.float32)
        return box_min, box_max 

    def get_white_list_pairs(self):
        palm_chain = ["root", "wrist_R", "palm_R", "base_link00"]
        finger_links = [
            "Link_11R", "Link_12R", "Link_13R", "Link_14R", "TH_Tip_R",
            "Link_21R", "Link_22R", "Link_23R", "Link_24R", "FF_Tip_R",
            "Link_31R", "Link_32R", "Link_33R", "Link_34R", "MF_Tip_R",
            "Link_41R", "Link_42R", "Link_43R", "Link_44R", "RF_Tip_R",
            "Link_51R", "Link_52R", "Link_53R", "Link_54R", "LF_Tip_R",
        ]
        return [[base, link] for base in palm_chain for link in finger_links]

    def get_default_urdf_path(self):
        """
            Default path to your robot URDF
        """
        return './assets/hand/dex5_hand/Dex5-URDF-R_teleop-manus_wocolor_mimic_limit.urdf'

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

        distal_links = [
            "Link_14R",
            "Link_24R",
            "Link_34R",
            "Link_44R",
            "Link_54R",
        ]
        front_poking_links = {} #{"Link_24R", "Link_34R", "Link_44R", "Link_54R"}

        for link in distal_links:
            disabled_normals = [
                (np.array([0.0, 0.0, -1.0]), 3.1415926 * 0.49999)
            ]

            if link in front_poking_links:
                # Filter the front tip-cap so the index finger cannot "poke"
                # the object with the fingertip and pass as a grasp contact.
                disabled_normals.append((np.array([1.0, 0.0, 0.0]), 0.55))

            config["movable_link"][link] = {
                "disabled_normal": disabled_normals
            }

        config["static_link"]["base_link00"] = {
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
        joint_names = [
            'Yaw_11R', 'Roll_12R', 'Pitch_13R', 'Pitch_14R',
            'Roll_21R', 'Pitch_22R', 'Pitch_23R', 'Pitch_24R',
            'Roll_31R', 'Pitch_32R', 'Pitch_33R', 'Pitch_34R',
            'Roll_41R', 'Pitch_42R', 'Pitch_43R', 'Pitch_44R',
            'Roll_51R', 'Pitch_52R', 'Pitch_53R', 'Pitch_54R',
        ]
        return joint_names

    def get_base_link(self):
        return "root"

    def get_static_links(self):
        return ["base_link00"]

    def get_mesh_scale(self):
        """
            Your robot mesh might be rescaled in your URDF, specify it here.
            (will be removed in the future.)
        """
        return 1.0


