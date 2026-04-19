"""Data loading and symmetry augmentation for SkyZero_V4 training."""

import os
import numpy as np
import torch
import _numpy_compat  # noqa: F401 -- installs header-parse compat for legacy npz


def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor: Tensor with shape (..., W, W)
        symm: 0-3 = rotation by symm * pi/2; 4-7 = mirror + rotation
    """
    assert tensor.shape[-1] == tensor.shape[-2]
    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)


def apply_symmetry_flat_policy(tensor, symm, pos_len):
    """Apply symmetry to flat policy tensor [B, board_area]."""
    batch_size = tensor.shape[0]
    spatial = tensor.view(batch_size, 1, pos_len, pos_len)
    transformed = apply_symmetry(spatial, symm)
    return transformed.reshape(batch_size, pos_len * pos_len).contiguous()


def read_npz_training_data(npz_files, batch_size, pos_len, device, randomize_symmetries=True):
    """
    Generator yielding training batches from shuffled NPZ files.

    Yields dicts with keys:
        encodedInputNCHW:        [B, C, H, W] float32
        policyTargetsN:          [B, board_area] float32
        opponentPolicyTargetsN:  [B, board_area] float32
        valueTargetsN:           [B, 3] float32
        sampleWeightsN:          [B] float32
    """
    rand = np.random.default_rng(seed=list(os.urandom(12)))

    for npz_file in npz_files:
        with np.load(npz_file) as npz:
            encoded = npz["encodedInputNCHW"].astype(np.float32)
            policy = npz["policyTargetsN"].astype(np.float32)
            opp_policy = npz["opponentPolicyTargetsN"].astype(np.float32)
            value = npz["valueTargetsN"].astype(np.float32)
            weights = npz["sampleWeightsN"].astype(np.float32)
            if "oppPolicyWeightsN" in npz:
                opp_policy_weights = npz["oppPolicyWeightsN"].astype(np.float32)
            else:
                opp_policy_weights = np.ones_like(weights)

        num_samples = encoded.shape[0]
        num_batches = num_samples // batch_size

        for n in range(num_batches):
            start = n * batch_size
            end = start + batch_size

            batch_encoded = torch.from_numpy(encoded[start:end]).to(device)
            batch_policy = torch.from_numpy(policy[start:end]).to(device)
            batch_opp_policy = torch.from_numpy(opp_policy[start:end]).to(device)
            batch_value = torch.from_numpy(value[start:end]).to(device)
            batch_weights = torch.from_numpy(weights[start:end]).to(device)
            batch_opp_policy_weights = torch.from_numpy(opp_policy_weights[start:end]).to(device)

            if randomize_symmetries:
                symm = int(rand.integers(0, 8))
                batch_encoded = apply_symmetry(batch_encoded, symm).contiguous()
                batch_policy = apply_symmetry_flat_policy(batch_policy, symm, pos_len)
                batch_opp_policy = apply_symmetry_flat_policy(batch_opp_policy, symm, pos_len)

            yield {
                "encodedInputNCHW": batch_encoded,
                "policyTargetsN": batch_policy,
                "opponentPolicyTargetsN": batch_opp_policy,
                "valueTargetsN": batch_value,
                "sampleWeightsN": batch_weights,
                "oppPolicyWeightsN": batch_opp_policy_weights,
            }


def collect_npz_files(data_dir):
    """Collect all .npz files from a directory, sorted by name."""
    files = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".npz"):
            files.append(os.path.join(data_dir, fname))
    return files
