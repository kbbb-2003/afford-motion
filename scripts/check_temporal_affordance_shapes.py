import argparse
import tempfile
from pathlib import Path

import numpy as np


CONTACT_JOINTS = [0, 10, 11, 12, 20, 21]


def extract_contact(contact: np.ndarray, joint_ids) -> np.ndarray:
    """Match the dataset joint selection logic for contact_cont_joints."""
    return contact[..., joint_ids]


def flatten_temporal_contact(contact_seq: np.ndarray) -> np.ndarray:
    """Flatten [K, N, J] to the ADM-facing shape [N, K * J]."""
    if contact_seq.ndim != 3:
        raise ValueError(f"Expected [K, N, J], but got {contact_seq.shape}.")
    return contact_seq.transpose(1, 0, 2).reshape(contact_seq.shape[1], -1)


def recover_temporal_contact(flat_contact: np.ndarray, num_phases: int, contact_dim: int) -> np.ndarray:
    """Recover [K, N, J] from the ADM-facing shape [N, K * J]."""
    if flat_contact.ndim != 2:
        raise ValueError(f"Expected [N, K * J], but got {flat_contact.shape}.")
    num_points = flat_contact.shape[0]
    return flat_contact.reshape(num_points, num_phases, contact_dim).transpose(1, 0, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-phases", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    num_phases = args.num_phases
    num_points = args.num_points
    num_samples = args.num_samples
    full_contact_dim = 22
    selected_contact_dim = len(CONTACT_JOINTS)

    # Synthetic temporal GT contact saved in contacts/*.npz.
    dist = np.random.rand(num_points, full_contact_dim).astype(np.float32)
    dist_seq = np.random.rand(num_phases, num_points, full_contact_dim).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        gt_path = tmpdir / "contact_gt.npz"
        np.savez(gt_path, dist=dist, dist_seq=dist_seq)

        gt_npz = np.load(gt_path)
        gt_contact_seq = extract_contact(gt_npz["dist_seq"], CONTACT_JOINTS)
        assert gt_contact_seq.shape == (num_phases, num_points, selected_contact_dim)

        # Stage 1 ADM target shape.
        adm_target = flatten_temporal_contact(gt_contact_seq)
        assert adm_target.shape == (num_points, num_phases * selected_contact_dim)

        # Stage 1 evaluator save format.
        recovered_seq = recover_temporal_contact(
            adm_target, num_phases=num_phases, contact_dim=selected_contact_dim
        )
        assert np.allclose(recovered_seq, gt_contact_seq)

        pred_contact = np.stack(
            [recovered_seq + sample_idx for sample_idx in range(num_samples)], axis=0
        ).astype(np.float32)
        pred_path = tmpdir / "pred_contact.npy"
        np.save(pred_path, pred_contact)

        reloaded_pred = np.load(pred_path)
        assert reloaded_pred.shape == (
            num_samples,
            num_phases,
            num_points,
            selected_contact_dim,
        )

        # Stage 2 test-time expectation: dataset returns [num_samples, K, N, J].
        stage2_test_contact = reloaded_pred
        assert stage2_test_contact.shape == (
            num_samples,
            num_phases,
            num_points,
            selected_contact_dim,
        )

        # Stage 2 mix-training expectation: one sample squeezed to [K, N, J].
        mixed_contact = reloaded_pred[:1].squeeze(0)
        assert mixed_contact.shape == (num_phases, num_points, selected_contact_dim)

        # test.py CMDM sampling now indexes batch contact as [:, k, ...].
        batch_pred_contact = np.stack([reloaded_pred, reloaded_pred + 100], axis=0)
        selected_sample = batch_pred_contact[:, 0, ...]
        assert selected_sample.shape == (2, num_phases, num_points, selected_contact_dim)

        print("Temporal affordance shape smoke test passed.")
        print(f"GT temporal contact shape: {gt_contact_seq.shape}")
        print(f"ADM flattened target shape: {adm_target.shape}")
        print(f"Recovered ADM output shape: {recovered_seq.shape}")
        print(f"Saved pred_contact shape: {reloaded_pred.shape}")
        print(f"Stage2 mixed-train contact shape: {mixed_contact.shape}")
        print(f"Stage2 sampled batch contact shape: {selected_sample.shape}")


if __name__ == "__main__":
    main()
