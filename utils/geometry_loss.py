import torch


def qinv(q: torch.Tensor) -> torch.Tensor:
    """Invert a quaternion with shape (..., 4)."""
    if q.shape[-1] != 4:
        raise ValueError(f"Quaternion shape should end with 4, but got {q.shape}.")
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors with quaternions.

    Args:
        q: quaternion tensor with shape (..., 4)
        v: vector tensor with shape (..., 3)
    """
    if q.shape[-1] != 4 or v.shape[-1] != 3:
        raise ValueError(f"Unexpected q/v shapes: {q.shape}, {v.shape}.")
    if q.shape[:-1] != v.shape[:-1]:
        raise ValueError(f"Quaternion/vector leading dims must match: {q.shape}, {v.shape}.")

    original_shape = v.shape
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def recover_root_rot_pos(data: torch.Tensor):
    """Recover root quaternion and root position from HumanML3D h3d features."""
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data: torch.Tensor, joints_num: int) -> torch.Tensor:
    """Recover joint positions from HumanML3D relative coordinates."""
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def motion_repr_to_joints(motion: torch.Tensor, motion_type: str, joints_num: int = 22) -> torch.Tensor:
    """Convert supported motion representations to joint positions.

    Args:
        motion: [B, L, D]
        motion_type: one of `pos`, `pos_rot`, `h3d`
    """
    if motion_type == 'pos':
        return motion[..., :joints_num * 3].reshape(*motion.shape[:2], joints_num, 3)
    if motion_type == 'pos_rot':
        return motion[..., :joints_num * 3].reshape(*motion.shape[:2], joints_num, 3)
    if motion_type == 'h3d':
        return recover_from_ric(motion, joints_num)
    raise NotImplementedError(f'Unsupported motion type for geometry loss: {motion_type}')


def sample_valid_frame_indices(x_mask: torch.Tensor, num_samples: int):
    """Sample evenly spaced valid frame indices from each sequence.

    Args:
        x_mask: [B, L], True for invalid/padded frames
        num_samples: sampled frames per sequence

    Returns:
        sampled_indices: [B, S]
        sampled_valid_mask: [B, S], True for valid sampled entries
    """
    batch_size, _ = x_mask.shape
    device = x_mask.device

    sampled_indices = []
    sampled_valid_mask = []
    for batch_idx in range(batch_size):
        valid_idx = torch.nonzero(~x_mask[batch_idx], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            sampled_indices.append(torch.zeros(num_samples, dtype=torch.long, device=device))
            sampled_valid_mask.append(torch.zeros(num_samples, dtype=torch.bool, device=device))
            continue

        use_samples = min(num_samples, valid_idx.numel())
        linspace_idx = torch.linspace(
            0,
            valid_idx.numel() - 1,
            steps=use_samples,
            device=device,
        ).round().long()
        current_indices = valid_idx[linspace_idx]
        current_mask = torch.ones(use_samples, dtype=torch.bool, device=device)

        if use_samples < num_samples:
            pad_count = num_samples - use_samples
            pad_indices = current_indices[-1:].repeat(pad_count)
            pad_mask = torch.zeros(pad_count, dtype=torch.bool, device=device)
            current_indices = torch.cat([current_indices, pad_indices], dim=0)
            current_mask = torch.cat([current_mask, pad_mask], dim=0)

        sampled_indices.append(current_indices)
        sampled_valid_mask.append(current_mask)

    return torch.stack(sampled_indices, dim=0), torch.stack(sampled_valid_mask, dim=0)


def batched_index_select(sequence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather sequence frames with per-batch indices.

    Args:
        sequence: [B, L, ...]
        indices: [B, S]
    """
    batch_size = sequence.shape[0]
    batch_indices = torch.arange(batch_size, device=sequence.device).unsqueeze(1).expand_as(indices)
    return sequence[batch_indices, indices]


def sample_scene_points(scene_xyz: torch.Tensor, num_points: int) -> torch.Tensor:
    """Sample scene points independently for each batch item.

    Args:
        scene_xyz: [B, N, 3]
        num_points: sampled point count
    """
    batch_size, total_points, _ = scene_xyz.shape
    device = scene_xyz.device

    sampled = []
    for batch_idx in range(batch_size):
        if total_points <= num_points:
            point_idx = torch.arange(total_points, device=device)
            if total_points < num_points:
                pad_idx = point_idx[-1:].repeat(num_points - total_points)
                point_idx = torch.cat([point_idx, pad_idx], dim=0)
        else:
            point_idx = torch.randperm(total_points, device=device)[:num_points]
        sampled.append(scene_xyz[batch_idx, point_idx])

    return torch.stack(sampled, dim=0)


def nearest_point_distances(query_points: torch.Tensor, scene_points: torch.Tensor) -> torch.Tensor:
    """Compute nearest scene-point distance for each query point.

    Args:
        query_points: [B, Q, 3]
        scene_points: [B, N, 3]

    Returns:
        distances: [B, Q]
    """
    pairwise_dist = torch.cdist(query_points, scene_points)
    return pairwise_dist.min(dim=-1).values


def smplx_signed_distance(
    object_points: torch.Tensor,
    smplx_vertices: torch.Tensor,
    smplx_face,
):
    """Approximate signed distance from scene points to a SMPL-X mesh.

    Positive values indicate the query point is inside the human mesh according
    to the closest-vertex normal heuristic used by the existing evaluator.
    """
    if not torch.is_tensor(smplx_face):
        smplx_face = torch.as_tensor(smplx_face, dtype=torch.long, device=smplx_vertices.device)
    else:
        smplx_face = smplx_face.to(device=smplx_vertices.device, dtype=torch.long)

    smplx_face_vertices = smplx_vertices[:, smplx_face]
    e1 = smplx_face_vertices[:, :, 1] - smplx_face_vertices[:, :, 0]
    e2 = smplx_face_vertices[:, :, 2] - smplx_face_vertices[:, :, 0]

    e1 = e1 / torch.norm(e1, dim=-1, keepdim=True).clamp_min(1e-8)
    e2 = e2 / torch.norm(e2, dim=-1, keepdim=True).clamp_min(1e-8)
    smplx_face_normal = torch.cross(e1, e2, dim=-1)

    smplx_vertex_normals = torch.zeros_like(smplx_vertices)
    smplx_vertex_normals.index_add_(1, smplx_face[:, 0], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:, 1], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:, 2], smplx_face_normal)
    smplx_vertex_normals = smplx_vertex_normals / torch.norm(
        smplx_vertex_normals, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    pairwise_distance = torch.cdist(object_points, smplx_vertices)
    distance_to_human, closest_human_points_idx = pairwise_distance.min(dim=2)

    closest_human_point = smplx_vertices.gather(
        1, closest_human_points_idx.unsqueeze(-1).expand(-1, -1, 3)
    )
    query_to_surface = closest_human_point - object_points
    query_to_surface = query_to_surface / torch.norm(
        query_to_surface, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    closest_vertex_normals = smplx_vertex_normals.gather(
        1, closest_human_points_idx.unsqueeze(-1).expand(-1, -1, 3)
    )
    same_direction = torch.sum(query_to_surface * closest_vertex_normals, dim=-1)
    signed_distance_to_human = same_direction.sign() * distance_to_human
    return signed_distance_to_human, closest_human_point
