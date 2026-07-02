# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 
import torch.nn.functional as F
from lygra.kinematics import batch_contact_ik, batch_fk
from lygra import gem
from tqdm import tqdm


def _empty_ik_result(
    tree,
    contact_link_ids,
    contact_pos_in_linkf,
    contact_normal_in_linkf,
    target_contact_pos,
    target_contact_normal,
    object_pose,
    q_mask=None
):
    n_dof = tree.n_actuated_dof()

    if q_mask is None:
        q_mask = torch.zeros((0, n_dof), dtype=torch.bool, device=object_pose.device)

    return {
        "q": torch.zeros((0, n_dof), dtype=contact_pos_in_linkf.dtype, device=contact_pos_in_linkf.device),
        "q_mask": q_mask[:0],
        "object_pose": object_pose[:0],
        "target_pos": target_contact_pos[:0],
        "target_normal": target_contact_normal[:0],
        "contact_pos": contact_pos_in_linkf[:0],
        "contact_normal": contact_normal_in_linkf[:0],
        "contact_link_id": contact_link_ids[:0],
    }


def _concat_result_chunks(chunks):
    if len(chunks) == 0:
        return {}

    result = {}
    keys = chunks[0].keys()
    for k in keys:
        values = [chunk[k] for chunk in chunks if k in chunk]
        if len(values) == 0:
            continue

        if isinstance(values[0], torch.Tensor):
            result[k] = torch.cat(values, dim=0)
        else:
            result[k] = values[0]

    return result


def batch_ik(
    tree,
    contact_ids,             # [batch, n_contact]
    contact_parent_ids, 
    contact_pos_in_linkf,    # [batch, n_contact, 3]
    contact_normal_in_linkf, # [batch, n_contact, 3]
    target_contact_pos,      # [batch, n_contact, 3]
    target_contact_normal,   # [batch, n_contact, 3]
    object_pose,
    gpu_memory_pool,
    n_dof=16,
    n_retry=10,
    regularization=1e-4,
    err_thresh=0.02,
    max_iter=12,
    step_size=0.4,
    filter=True,
    ik_batch_size=None
):
    batch_size = contact_ids.shape[0]
    contact_link_ids = contact_parent_ids[contact_ids]  # [batch, n_contact]

    if batch_size == 0:
        return _empty_ik_result(
            tree,
            contact_link_ids,
            contact_pos_in_linkf,
            contact_normal_in_linkf,
            target_contact_pos,
            target_contact_normal,
            object_pose
        )

    if ik_batch_size is not None and batch_size > ik_batch_size:
        if ik_batch_size <= 0:
            raise ValueError("ik_batch_size must be positive.")

        chunks = []
        for start in tqdm(range(0, batch_size, ik_batch_size), desc="Contact IK Chunks"):
            end = min(start + ik_batch_size, batch_size)
            chunks.append(batch_ik(
                tree=tree,
                contact_ids=contact_ids[start:end],
                contact_parent_ids=contact_parent_ids,
                contact_pos_in_linkf=contact_pos_in_linkf[start:end],
                contact_normal_in_linkf=contact_normal_in_linkf[start:end],
                target_contact_pos=target_contact_pos[start:end],
                target_contact_normal=target_contact_normal[start:end],
                object_pose=object_pose[start:end],
                gpu_memory_pool=gpu_memory_pool,
                n_dof=n_dof,
                n_retry=n_retry,
                regularization=regularization,
                err_thresh=err_thresh,
                max_iter=max_iter,
                step_size=step_size,
                filter=filter,
                ik_batch_size=None
            ))
        return _concat_result_chunks(chunks)

    ik_result = batch_contact_ik(
        tree,
        target_contact_pos, 
        target_contact_normal, 
        contact_pos_in_linkf, 
        contact_normal_in_linkf, 
        contact_link_ids,   
        n_retry=n_retry, 
        step_size=step_size, 
        max_iter=max_iter, 
        regularization=regularization,
        err_thresh=err_thresh,
        alpha_contact_matching=1.0,
        all_joint_result_buffer=gpu_memory_pool.get_ik_joint_buffer(batch_size, n_retry),
        all_link_result_buffer=gpu_memory_pool.get_ik_link_buffer(batch_size, n_retry),
        jac_result_buffer=gpu_memory_pool.get_ik_jac_result_buffer(batch_size, n_retry),
        J_err_buffer=gpu_memory_pool.get_ik_jac_error_buffer(batch_size, n_retry, contact_link_ids.size(-1))
    )

    q = ik_result["q"]
    success_idx = ik_result["success"]
    q_mask = ik_result["q_mask"]

    # q:             [batch, n_dof]
    # success_idx:   [batch]
    if filter:
        result = {
            "q":                q[success_idx], 
            "q_mask":           q_mask[success_idx], 
            "object_pose":      object_pose[success_idx],
            "target_pos":       target_contact_pos[success_idx],
            "target_normal":    target_contact_normal[success_idx],
            "contact_pos":      contact_pos_in_linkf[success_idx],
            "contact_normal":   contact_normal_in_linkf[success_idx],
            "contact_link_id":  contact_link_ids[success_idx]
        }
    else:
        result = {
            "q":                q, 
            "q_mask":           q_mask, 
            "object_pose":      object_pose,
            "target_pos":       target_contact_pos,
            "target_normal":    target_contact_normal,
            "contact_pos":      contact_pos_in_linkf,
            "contact_normal":   contact_normal_in_linkf,
            "contact_link_id":  contact_link_ids
        }
    return result


def batch_contact_adjustment(
    tree,
    mesh,
    q_init,
    q_mask,
    contact_ids,
    contact_link_ids,        # [batch, n_contact]
    contact_pos_in_linkf,    # [batch, n_contact, 3]
    contact_normal_in_linkf, # [batch, n_contact, 3]
    target_contact_pos,      # [batch, n_contact, 3]
    target_contact_normal,   # [batch, n_contact, 3]
    object_pose,             # [batch, 4, 4]
    gpu_memory_pool,
    n_iter=4,
    project_per_n_step=3,
    ret_mesh_buffer=False,
    projection_orientation_weight=0.01,
    error_tolerance=0.004,
    ik_step_size=0.4,
    ik_regularization=2e-4,
    adjustment_batch_size=None
):
    batch_size = q_init.shape[0]

    if batch_size == 0:
        print("No solution found after coarse IK.")
        result = _empty_ik_result(
            tree,
            contact_link_ids,
            contact_pos_in_linkf,
            contact_normal_in_linkf,
            target_contact_pos,
            target_contact_normal,
            object_pose,
            q_mask=q_mask
        )
        if ret_mesh_buffer:
            result["mesh_buffer"] = None
        return result

    if adjustment_batch_size is not None and batch_size > adjustment_batch_size:
        if adjustment_batch_size <= 0:
            raise ValueError("adjustment_batch_size must be positive.")

        chunks = []
        for start in tqdm(range(0, batch_size, adjustment_batch_size), desc="Kinematics Finetuning Chunks"):
            end = min(start + adjustment_batch_size, batch_size)
            chunk_contact_ids = contact_ids[start:end] if isinstance(contact_ids, torch.Tensor) else contact_ids
            chunks.append(batch_contact_adjustment(
                tree=tree,
                mesh=mesh,
                q_init=q_init[start:end],
                q_mask=q_mask[start:end],
                contact_ids=chunk_contact_ids,
                contact_link_ids=contact_link_ids[start:end],
                contact_pos_in_linkf=contact_pos_in_linkf[start:end],
                contact_normal_in_linkf=contact_normal_in_linkf[start:end],
                target_contact_pos=target_contact_pos[start:end],
                target_contact_normal=target_contact_normal[start:end],
                object_pose=object_pose[start:end],
                gpu_memory_pool=gpu_memory_pool,
                n_iter=n_iter,
                project_per_n_step=project_per_n_step,
                ret_mesh_buffer=False,
                projection_orientation_weight=projection_orientation_weight,
                error_tolerance=error_tolerance,
                ik_step_size=ik_step_size,
                ik_regularization=ik_regularization,
                adjustment_batch_size=None
            ))

        result = _concat_result_chunks(chunks)
        if ret_mesh_buffer:
            result["mesh_buffer"] = {
                "v": torch.from_numpy(mesh['v']).cuda().float(),
                "f": torch.from_numpy(mesh['f']).cuda().int(),
                "n": torch.from_numpy(mesh['n']).cuda().float(),
                "vi": torch.from_numpy(mesh['vi']).cuda().int(),
                "fi": torch.from_numpy(mesh['fi']).cuda().int()
            }
        return result

    v_tensor = torch.from_numpy(mesh['v']).cuda().float()
    f_tensor = torch.from_numpy(mesh['f']).cuda().int()
    n_tensor = torch.from_numpy(mesh['n']).cuda().float()
    v_idx_tensor = torch.from_numpy(mesh['vi']).cuda().int()
    f_idx_tensor = torch.from_numpy(mesh['fi']).cuda().int()

    contact_queries = target_contact_pos
    contact_queries_normal = F.normalize(target_contact_normal, dim=-1)
    q = q_init 

    for i in tqdm(range(n_iter * project_per_n_step), desc="Kinematics Finetuning"):
        mesh_pose = batch_fk(tree, q)["link"]

        if i % project_per_n_step == 0:
            refined_contact = gem.batch_solve_contact_with_normal(
                mesh_pose,
                contact_link_ids.int(),
                v_tensor,
                v_idx_tensor,
                f_tensor,
                n_tensor,
                f_idx_tensor,
                contact_queries,
                contact_queries_normal,
                True,    # to link frames.
                projection_orientation_weight
            )

            next_contact_pos_in_linkf = refined_contact[:, :, :3]
            next_contact_normal_in_linkf = refined_contact[:, :, 3:]

            contact_pos_in_linkf = next_contact_pos_in_linkf
            contact_normal_in_linkf = next_contact_normal_in_linkf
 
        ik_result = batch_contact_ik(
            tree,
            target_contact_pos, 
            target_contact_normal, 
            contact_pos_in_linkf, 
            contact_normal_in_linkf, 
            contact_link_ids,   
            n_retry=1, 
            step_size=ik_step_size, 
            max_iter=1, 
            regularization=ik_regularization,
            single_update_mode=True,
            q_init=q,
            all_joint_result_buffer=gpu_memory_pool.get_ik_joint_buffer(q_init.size(0), 1),
            all_link_result_buffer=gpu_memory_pool.get_ik_link_buffer(q_init.size(0), 1),
            jac_result_buffer=gpu_memory_pool.get_ik_jac_result_buffer(q_init.size(0), 1),
            J_err_buffer=gpu_memory_pool.get_ik_jac_error_buffer(q_init.size(0), 1, contact_link_ids.size(-1))
        )
        q = ik_result["q"]

    mesh_pose = batch_fk(tree, q)["link"]
    refined_contact = gem.batch_solve_contact_with_normal(
        mesh_pose,
        contact_link_ids.int(),
        v_tensor,
        v_idx_tensor,
        f_tensor,
        n_tensor,
        f_idx_tensor,
        contact_queries,
        contact_queries_normal,
        True,    # to link frames.
        0.0
    )

    contact_pos_in_linkf = refined_contact[:, :, :3]
    contact_normal_in_linkf = refined_contact[:, :, 3:]

    # Filter.
    err = ik_result["err"]
    success_idx = (err < error_tolerance ** 2).all(dim=-1)
 
    # q:            [batch, n_dof]
    # success_idx:  [batch]
    result = {
        "q":                q[success_idx], 
        "q_mask":           q_mask[success_idx],
        "object_pose":      object_pose[success_idx],
        "target_pos":       target_contact_pos[success_idx],
        "target_normal":    target_contact_normal[success_idx],
        "contact_pos":      contact_pos_in_linkf[success_idx],
        "contact_normal":   contact_normal_in_linkf[success_idx],
        "contact_link_id":  contact_link_ids[success_idx],
        
    }

    if ret_mesh_buffer:
        result["mesh_buffer"] = {
            "v": v_tensor,
            "f": f_tensor,
            "n": n_tensor,
            "vi": v_idx_tensor,
            "fi": f_idx_tensor
        }
    return result
