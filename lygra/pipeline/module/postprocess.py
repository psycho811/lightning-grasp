# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm 
from lygra.pipeline.module.collision import batch_filter_collision
import torch


def batch_filter_collision_chunked(
    tree,
    q,
    object_pose,
    object_point,
    self_collision_link_pairs,
    decomposed_mesh_data,
    chunk_size=512
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    masks = []
    for start in range(0, len(q), chunk_size):
        end = min(start + chunk_size, len(q))
        mask = batch_filter_collision(
            tree,
            q[start:end],
            object_pose[start:end],
            object_point,
            self_collision_link_pairs,
            decomposed_mesh_data,
            ret_mask_only=True
        )
        masks.append(mask.bool())

    if len(masks) == 0:
        return torch.zeros((0,), dtype=torch.bool, device=q.device)

    return torch.cat(masks, dim=0)


def batch_assign_free_finger_and_filter(
    tree,
    result,
    object_point,
    self_collision_link_pairs,
    decomposed_mesh_data,
    n_assign_retry=5,
    collision_batch_size=512
):
    if len(result["q"]) == 0:
        return result

    assigned_results = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            assigned_results[k] = []
        else:
            assigned_results[k] = v 

    joint_limit_lower, joint_limit_upper = tree.get_actuated_joint_limit()
    joint_limit_lower = torch.from_numpy(joint_limit_lower).to(object_point.device)
    joint_limit_upper = torch.from_numpy(joint_limit_upper).to(object_point.device)
    
    tmp_result = {k: v for k, v in result.items()}

    n_initial = 0

    for i in tqdm(range(n_assign_retry), desc="Postprocessing"):
        q = tmp_result["q"]
        q_mask = tmp_result["q_mask"].float()
       
        free_q = torch.rand_like(q) * (joint_limit_upper - joint_limit_lower) + joint_limit_lower
        tmp_result["q"] = free_q * (1 - q_mask) + q * q_mask 
        
        success_mask = batch_filter_collision_chunked(
            tree,
            tmp_result["q"],
            tmp_result["object_pose"],
            object_point,
            self_collision_link_pairs,
            decomposed_mesh_data,
            chunk_size=collision_batch_size
        )
       
        for k, v in tmp_result.items():
            if isinstance(v, torch.Tensor):
                assigned_results[k].append(tmp_result[k][torch.where(success_mask)])
                tmp_result[k] = tmp_result[k][torch.where(success_mask.logical_not())]

        if len(tmp_result["q"]) == 0:
            break

        if q_mask.bool().all():
            break 

        n_success = success_mask.int().sum().item()
        if i == 0:
            n_initial = n_success
        else:
            if n_success < 0.1 * n_initial:
                # benefit from extra rounds is marginal.
                break

    for k, v in assigned_results.items():
        if isinstance(result[k], torch.Tensor):
            assigned_results[k] = torch.cat(v, dim=0) if len(v) > 0 else result[k][:0]

    return assigned_results
