# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Lygra Common
from lygra.robot import build_robot
from lygra.contact_set import get_dependency_matrix, get_link_dependency_matrix
from lygra.kinematics import build_kinematics_tree
from lygra.mesh import get_urdf_mesh, get_urdf_mesh_decomposed, get_urdf_mesh_for_projection, trimesh_to_open3d
from lygra.mesh_analyzer import get_support_point_mask
from lygra.utils.geom_utils import MeshObject
from lygra.memory import IKGPUBufferPool
from lygra.utils.robot_visualizer import RobotVisualizer
from lygra.utils.transform_utils import batch_object_transform

# Lygra Pipeline
from lygra.pipeline.module.object_placement import sample_object_pose, get_object_pose_sampling_args
from lygra.pipeline.module.contact_query import batch_object_all_contact_fields_interaction
from lygra.pipeline.module.contact_collection import sample_pose_and_contact_from_interaction
from lygra.pipeline.module.contact_optimization import search_contact_point
from lygra.pipeline.module.kinematics import batch_ik, batch_contact_adjustment
from lygra.pipeline.module.collision import batch_filter_collision
from lygra.pipeline.module.postprocess import batch_assign_free_finger_and_filter

# Common 
import torch 
import trimesh
from tqdm import tqdm 
import numpy as np 
import open3d as o3d
import argparse
import time 
import random
import sys


def get_args():
    parser = argparse.ArgumentParser(description="Script configuration")
    parser.add_argument('--robot', type=str, default="allegro", help='Robot Name')
    parser.add_argument('--batch_size_outer', type=int, default=128, help='Outer batch size (Object Pose)')
    parser.add_argument('--batch_size_inner', type=int, default=128, help='Inner batch size (Contact Domain Variants)')
    parser.add_argument('--n_contact', type=int, default=3, help='Number of non-static contacts to optimize')
    parser.add_argument('--n_sample_point', type=int, default=2048, help='Number of sampled object points')
    parser.add_argument('--ik_finetune_iter', type=int, default=5, help='Number of IK finetune iterations')
    parser.add_argument('--zo_lr_sigma', type=float, default=5, help='Sigma of the Zeroth-order Optimizer')

    parser.add_argument('--cf_accel', type=str, default='lbvhs2', help='Contact Field Acceleration Structure')
    parser.add_argument('--object_pose_sampling_strategy', type=str, default='canonical', help='Object pose sampling strategy')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--object_mesh_path', type=str, default="./assets/object/ycb/013_apple/textured.obj", help='Path to the object mesh')
    


    args = parser.parse_args()
    return args


# In gratitude to the NJU ICS-PA, HKUST RI, and Berkeley CS267 teaching teams.
# This program embodies lessons and memories that came alive during its creation.
#                                                               -- Zhao-Heng Yin  
#                                                                       Nov 2025
def main(args):
    batch_size_outer = args.batch_size_outer
    batch_size_inner = args.batch_size_inner
    n_contact = args.n_contact
    n_sample_point = args.n_sample_point
    ik_finetune_iter = args.ik_finetune_iter
    cf_accel = args.cf_accel
    object_pose_sampling_strategy = args.object_pose_sampling_strategy
    visualize = args.visualize
    object_mesh_path = args.object_mesh_path
    zo_lr_sigma = args.zo_lr_sigma

    # -----------------
    # Preparation Stage 
    # -----------------
    robot = build_robot(args.robot)

    # Robot Structure.
    tree = build_kinematics_tree(
        urdf_path=robot.urdf_path,
        active_joint_names=robot.get_active_joints()
    )

   

    

    # Object Data.
    object = MeshObject(object_mesh_path)
    object_area = object.get_area()
    zo_lr = ((object_area / n_sample_point) ** 0.5) * zo_lr_sigma
    points, normals = object.sample_point_and_normal(count=n_sample_point)
    points_all = torch.from_numpy(points).cuda().float()
    normals_all = torch.from_numpy(normals).cuda().float()

   
   

    # ---------------
    # Inference Stage 
    # ---------------
    # TODO: Refactor Args/Returns (Nov 10)
    # I should have a class to wrap these arg/return values below as people did for professional graphics engines.
    # But I am too lazy to move, python dict is so comforting for prototyping.

    # result = np.load(f"./grasp_results/{args.robot}/1779344930/grasp_solutions.npz", allow_pickle=True)
    # result = np.load(f"./grasp_results/{args.robot}/20260521_064310/grasp_solutions.npz", allow_pickle=True)
    result = np.load(f"./grasp_results/{args.robot}/20260528_3_demo/grasp_solutions.npz", allow_pickle=True)

    n_result = len(result['q'])
    print(f"Found {n_result} grasping solutions.")

    # -----------------
    # Visualize Results
    # -----------------
    if not visualize:
        sys.exit(0)

    viewer = RobotVisualizer(robot)

    while True:
        idx = random.randint(0, n_result - 1)
        robot_mesh = viewer.get_mesh_fk(result['q'][idx:idx+1], visual=True)

        object_mesh = object.mesh.copy()
        object_mesh.apply_transform(result['object_pose'][idx])
        object_mesh_o3d = trimesh_to_open3d(object_mesh)
        
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color = [245 / 256, 162 / 256, 98 / 256, 0.8]
        material.base_metallic = 0.0
        material.base_roughness = 1.0
        object_mesh = {"name": 'object', "geometry": object_mesh_o3d, "material": material}
        viewer.show(robot_mesh + [object_mesh])

        if input("Continue? (Y/n)") in ['n', 'N']:
            break


if __name__ == '__main__':
    args = get_args()
    main(args)