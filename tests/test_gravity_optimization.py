import unittest

import torch

from lygra.batch_grasp_optimizer import compute_wrench_score_vectorized
from lygra.pipeline.module.contact_optimization import build_gravity_condition


class GravityConditionTest(unittest.TestCase):
    def test_builds_force_in_object_frame(self):
        object_poses = torch.eye(4).repeat(2, 1, 1)
        object_poses[1, :3, :3] = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        condition = build_gravity_condition(
            object_poses,
            center_of_mass=torch.tensor([0.1, 0.2, 0.3]),
            gravity_direction=(0.0, 1.0, 0.0),
            gravity_scale=0.2
        )

        expected_force = torch.tensor([
            [0.0, 0.2, 0.0],
            [0.2, 0.0, 0.0]
        ])
        self.assertTrue(torch.allclose(condition["gravity_force"], expected_force))
        self.assertEqual(condition["center_of_mass"].shape, (2, 3))

    def test_rejects_zero_direction(self):
        with self.assertRaises(ValueError):
            build_gravity_condition(
                torch.eye(4).unsqueeze(0),
                center_of_mass=torch.zeros(3),
                gravity_direction=(0.0, 0.0, 0.0)
            )

    def test_gravity_changes_wrench_score(self):
        contact_pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        contact_normal = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])

        score_without_gravity, _ = compute_wrench_score_vectorized(contact_pos, contact_normal)
        score_with_gravity, _ = compute_wrench_score_vectorized(
            contact_pos,
            contact_normal,
            gravity_force=torch.tensor([[0.0, 0.2, 0.0]]),
            center_of_mass=torch.zeros(1, 3)
        )

        self.assertGreater(score_with_gravity.item(), score_without_gravity.item())


if __name__ == "__main__":
    unittest.main()
