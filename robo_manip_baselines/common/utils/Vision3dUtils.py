import numpy as np
import pytorch3d.ops as torch3d_ops
import torch


def crop_pointcloud_bb(pointcloud: np.ndarray, min_bound=None, max_bound=None):
    """Crop the point cloud using a bounding box."""
    if min_bound is not None:
        mask = np.all(pointcloud[:, :3] > min_bound, axis=1)
        pointcloud = pointcloud[mask]
    if max_bound is not None:
        mask = np.all(pointcloud[:, :3] < max_bound, axis=1)
        pointcloud = pointcloud[mask]
    return pointcloud


def downsample_pointcloud_fps(pointcloud: np.ndarray, num_points: int = 512):
    """Downsample the point cloud with farthest point sampling (FPS)."""
    pointcloud_tensor = torch.from_numpy(pointcloud).unsqueeze(0)
    num_points_tensor = torch.tensor([num_points])
    _, sampled_indices = torch3d_ops.sample_farthest_points(
        pointcloud_tensor, K=num_points_tensor
    )
    pointcloud = pointcloud[sampled_indices.squeeze(0).numpy()]
    return pointcloud
