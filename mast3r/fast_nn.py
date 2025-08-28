# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R Fast Nearest Neighbor
# --------------------------------------------------------
import torch
import numpy as np
import math
import time
from contextlib import contextmanager
from scipy.spatial import KDTree

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.device import to_numpy, todevice  # noqa


# Profiling infrastructure for fast_nn module
_fast_nn_profile_data = {}

@contextmanager
def _profile_timer(operation_name, enabled=True):
    """Context manager for timing operations in fast_nn module."""
    if not enabled:
        yield
        return
        
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if operation_name not in _fast_nn_profile_data:
            _fast_nn_profile_data[operation_name] = {'total_time': 0, 'count': 0, 'times': []}
        _fast_nn_profile_data[operation_name]['total_time'] += elapsed
        _fast_nn_profile_data[operation_name]['count'] += 1
        _fast_nn_profile_data[operation_name]['times'].append(elapsed)


def get_fast_nn_profile_data():
    """Get profiling data for fast_nn operations."""
    return _fast_nn_profile_data.copy()


def clear_fast_nn_profile_data():
    """Clear profiling data for fast_nn operations."""
    global _fast_nn_profile_data
    _fast_nn_profile_data = {}


@torch.no_grad()
def bruteforce_reciprocal_nns(A, B, device='cuda', block_size=None, dist='l2'):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    A = A.to(device)
    B = B.to(device)

    if dist == 'l2':
        dist_func = torch.cdist
        argmin = torch.min
    elif dist == 'dot':
        def dist_func(A, B):
            return A @ B.T

        def argmin(X, dim):
            sim, nn = torch.max(X, dim=dim)
            return sim.neg_(), nn
    else:
        raise ValueError(f'Unknown {dist=}')

    if block_size is None or len(A) * len(B) <= block_size**2:
        dists = dist_func(A, B)
        _, nn_A = argmin(dists, dim=1)
        _, nn_B = argmin(dists, dim=0)
    else:
        dis_A = torch.full((A.shape[0],), float('inf'), device=device, dtype=A.dtype)
        dis_B = torch.full((B.shape[0],), float('inf'), device=device, dtype=B.dtype)
        nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
        nn_B = torch.full((B.shape[0],), -1, device=device, dtype=torch.int64)
        number_of_iteration_A = math.ceil(A.shape[0] / block_size)
        number_of_iteration_B = math.ceil(B.shape[0] / block_size)

        for i in range(number_of_iteration_A):
            A_i = A[i * block_size:(i + 1) * block_size]
            for j in range(number_of_iteration_B):
                B_j = B[j * block_size:(j + 1) * block_size]
                dists_blk = dist_func(A_i, B_j)  # A, B, 1
                # dists_blk = dists[i * block_size:(i+1)*block_size, j * block_size:(j+1)*block_size]
                min_A_i, argmin_A_i = argmin(dists_blk, dim=1)
                min_B_j, argmin_B_j = argmin(dists_blk, dim=0)

                col_mask = min_A_i < dis_A[i * block_size:(i + 1) * block_size]
                line_mask = min_B_j < dis_B[j * block_size:(j + 1) * block_size]

                dis_A[i * block_size:(i + 1) * block_size][col_mask] = min_A_i[col_mask]
                dis_B[j * block_size:(j + 1) * block_size][line_mask] = min_B_j[line_mask]

                nn_A[i * block_size:(i + 1) * block_size][col_mask] = argmin_A_i[col_mask] + (j * block_size)
                nn_B[j * block_size:(j + 1) * block_size][line_mask] = argmin_B_j[line_mask] + (i * block_size)
    nn_A = nn_A.cpu().numpy()
    nn_B = nn_B.cpu().numpy()
    return nn_A, nn_B


class cdistMatcher:
    def __init__(self, db_pts, device='cuda'):
        self.db_pts = db_pts.to(device)
        self.device = device

    def query(self, queries, k=1, **kw):
        assert k == 1
        if queries.numel() == 0:
            return None, []
        nnA, nnB = bruteforce_reciprocal_nns(queries, self.db_pts, device=self.device, **kw)
        dis = None
        return dis, nnA


def merge_corres(idx1, idx2, shape1=None, shape2=None, ret_xy=True, ret_index=False):
    assert idx1.dtype == idx2.dtype == np.int32

    # unique and sort along idx1
    corres = np.unique(np.c_[idx2, idx1].view(np.int64), return_index=ret_index)
    if ret_index:
        corres, indices = corres
    xy2, xy1 = corres[:, None].view(np.int32).T

    if ret_xy:
        assert shape1 and shape2
        xy1 = np.unravel_index(xy1, shape1)
        xy2 = np.unravel_index(xy2, shape2)
        if ret_xy != 'y_x':
            xy1 = xy1[0].base[:, ::-1]
            xy2 = xy2[0].base[:, ::-1]

    if ret_index:
        return xy1, xy2, indices
    return xy1, xy2


def fast_reciprocal_NNs(pts1, pts2, subsample_or_initxy1=8, ret_xy=True, pixel_tol=0, ret_basin=False,
                        device='cuda', enable_profiling=True, **matcher_kw):
    with _profile_timer("Setup and Tensor Preparation", enable_profiling):
        H1, W1, DIM1 = pts1.shape
        H2, W2, DIM2 = pts2.shape
        assert DIM1 == DIM2

        pts1 = pts1.reshape(-1, DIM1)
        pts2 = pts2.reshape(-1, DIM2)

        if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
            S = subsample_or_initxy1
            y1, x1 = np.mgrid[S // 2:H1:S, S // 2:W1:S].reshape(2, -1)
            max_iter = 10
        else:
            x1, y1 = subsample_or_initxy1
            if isinstance(x1, torch.Tensor):
                x1 = x1.cpu().numpy()
            if isinstance(y1, torch.Tensor):
                y1 = y1.cpu().numpy()
            max_iter = 1

        xy1 = np.int32(np.unique(x1 + W1 * y1))  # make sure there's no doublons
        xy2 = np.full_like(xy1, -1)
        old_xy1 = xy1.copy()
        old_xy2 = xy2.copy()

    with _profile_timer("Tree Building and Device Setup", enable_profiling):
        if 'dist' in matcher_kw or 'block_size' in matcher_kw \
                or (isinstance(device, str) and device.startswith('cuda')) \
                or (isinstance(device, torch.device) and device.type.startswith('cuda')):
            pts1 = pts1.to(device)
            pts2 = pts2.to(device)
            tree1 = cdistMatcher(pts1, device=device)
            tree2 = cdistMatcher(pts2, device=device)
        else:
            pts1, pts2 = to_numpy((pts1, pts2))
            tree1 = KDTree(pts1)
            tree2 = KDTree(pts2)

    with _profile_timer("Iterative Reciprocal Matching", enable_profiling):
        notyet = np.ones(len(xy1), dtype=bool)
        basin = np.full((H1 * W1 + 1,), -1, dtype=np.int32) if ret_basin else None

        niter = 0
        # n_notyet = [len(notyet)]
        while notyet.any():
            _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))
            if not ret_basin:
                notyet &= (old_xy2 != xy2)  # remove points that have converged

            _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))
            if ret_basin:
                basin[old_xy1[notyet]] = xy1[notyet]
            notyet &= (old_xy1 != xy1)  # remove points that have converged

            # n_notyet.append(notyet.sum())
            niter += 1
            if niter >= max_iter:
                break

            old_xy2[:] = xy2
            old_xy1[:] = xy1

        # print('notyet_stats:', ' '.join(map(str, (n_notyet+[0]*10)[:max_iter])))

    with _profile_timer("Post-processing and Merging", enable_profiling):
        if pixel_tol > 0:
            # in case we only want to match some specific points
            # and still have some way of checking reciprocity
            old_yx1 = np.unravel_index(old_xy1, (H1, W1))[0].base
            new_yx1 = np.unravel_index(xy1, (H1, W1))[0].base
            dis = np.linalg.norm(old_yx1 - new_yx1, axis=-1)
            converged = dis < pixel_tol
            if not isinstance(subsample_or_initxy1, int):
                xy1 = old_xy1  # replace new points by old ones
        else:
            converged = ~notyet  # converged correspondences

        # keep only unique correspondences, and sort on xy1
        xy1, xy2 = merge_corres(xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy)
    
    if ret_basin:
        return xy1, xy2, basin
    return xy1, xy2


@torch.no_grad()
def fast_reciprocal_NNs_rectified(pts1, pts2, subsample_or_initxy1=8, ret_xy=True, pixel_tol=0,
                                  device='cuda', enable_profiling=True, max_disparity=None, **matcher_kw):
    """
    Fast reciprocal nearest neighbors for rectified stereo pairs.
    Takes advantage of the fact that corresponding points lie on the same horizontal lines (epipolar constraint).
    
    Args:
        pts1, pts2: Feature descriptors (H, W, DIM)
        subsample_or_initxy1: Subsampling factor or initial coordinates
        ret_xy: Whether to return xy coordinates
        pixel_tol: Pixel tolerance for convergence
        device: Computing device
        enable_profiling: Whether to enable profiling
        max_disparity: Maximum disparity to search (optional constraint)
        **matcher_kw: Additional matcher arguments
    
    Returns:
        xy1, xy2: Corresponding coordinates
    """
    with _profile_timer("Rectified Setup and Tensor Preparation", enable_profiling):
        H1, W1, DIM1 = pts1.shape
        H2, W2, DIM2 = pts2.shape
        assert DIM1 == DIM2
        assert H1 == H2, "Rectified images must have same height"
        
        pts1 = pts1.reshape(-1, DIM1)
        pts2 = pts2.reshape(-1, DIM2)
        
        if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
            S = subsample_or_initxy1
            y1, x1 = np.mgrid[S // 2:H1:S, S // 2:W1:S].reshape(2, -1)
            max_iter = 10
        else:
            x1, y1 = subsample_or_initxy1
            if isinstance(x1, torch.Tensor):
                x1 = x1.cpu().numpy()
            if isinstance(y1, torch.Tensor):
                y1 = y1.cpu().numpy()
            max_iter = 1
        
        # Group points by y-coordinate for epipolar constraint
        unique_y = np.unique(y1)
        y_groups = {}
        for y in unique_y:
            mask = y1 == y
            y_groups[y] = {
                'x1': x1[mask],
                'indices1': np.where(mask)[0],
                'xy1_indices': x1[mask] + W1 * y  # Linear indices in image 1
            }
    
    with _profile_timer("Rectified Tree Building and Device Setup", enable_profiling):
        if 'dist' in matcher_kw or 'block_size' in matcher_kw \
                or (isinstance(device, str) and device.startswith('cuda')) \
                or (isinstance(device, torch.device) and device.type.startswith('cuda')):
            pts1 = pts1.to(device)
            pts2 = pts2.to(device)
            use_cuda = True
        else:
            pts1, pts2 = to_numpy((pts1, pts2))
            use_cuda = False
    
    with _profile_timer("Rectified Epipolar Constrained Matching", enable_profiling):
        all_xy1 = []
        all_xy2 = []
        
        for y in unique_y:
            group = y_groups[y]
            if len(group['x1']) == 0:
                continue
            
            # Extract features for this horizontal line in both images
            line1_indices = group['xy1_indices']
            line2_indices = np.arange(y * W2, (y + 1) * W2)  # Full horizontal line in image 2
            
            # Apply disparity constraint if specified
            if max_disparity is not None:
                # For each x1, only consider x2 in range [x1-max_disparity, x1+max_disparity]
                valid_line2_indices = []
                x1_to_line2_range = {}
                
                for i, x1_val in enumerate(group['x1']):
                    x2_start = max(0, x1_val - max_disparity)
                    x2_end = min(W2, x1_val + max_disparity + 1)
                    x2_range = np.arange(x2_start, x2_end)
                    x1_to_line2_range[i] = len(valid_line2_indices) + np.arange(len(x2_range))
                    valid_line2_indices.extend(y * W2 + x2_range)
                
                line2_indices = np.array(valid_line2_indices)
            
            if len(line2_indices) == 0:
                continue
            
            # Extract feature vectors for this horizontal line
            line1_features = pts1[line1_indices]
            line2_features = pts2[line2_indices]
            
            if use_cuda:
                # Use CUDA-based matching
                if 'dist' not in matcher_kw:
                    matcher_kw['dist'] = 'dot'
                
                # Compute similarity/distance matrix
                if matcher_kw.get('dist') == 'dot':
                    similarities = line1_features @ line2_features.T
                    _, nn_1to2 = torch.max(similarities, dim=1)
                    _, nn_2to1 = torch.max(similarities, dim=0)
                else:  # L2 distance
                    distances = torch.cdist(line1_features, line2_features)
                    _, nn_1to2 = torch.min(distances, dim=1)
                    _, nn_2to1 = torch.min(distances, dim=0)
                
                nn_1to2 = nn_1to2.cpu().numpy()
                nn_2to1 = nn_2to1.cpu().numpy()
            else:
                # Use CPU-based matching with KDTree
                from scipy.spatial import KDTree
                tree2 = KDTree(line2_features)
                tree1 = KDTree(line1_features)
                
                # Find nearest neighbors
                _, nn_1to2 = tree2.query(line1_features)
                _, nn_2to1 = tree1.query(line2_features)
            
            # Check reciprocal consistency
            reciprocal_mask = []
            for i in range(len(line1_indices)):
                j = nn_1to2[i]
                if j < len(nn_2to1) and nn_2to1[j] == i:
                    reciprocal_mask.append(True)
                else:
                    reciprocal_mask.append(False)
            
            reciprocal_mask = np.array(reciprocal_mask)
            
            if reciprocal_mask.any():
                # Convert back to image coordinates
                valid_i = np.where(reciprocal_mask)[0]
                for i in valid_i:
                    x1_coord = group['x1'][i]
                    y_coord = y
                    
                    j = nn_1to2[i]
                    if max_disparity is not None:
                        # Map back from constrained indices to full image coordinates
                        x2_start = max(0, x1_coord - max_disparity)
                        x2_coord = x2_start + (j % (2 * max_disparity + 1))
                        x2_coord = min(x2_coord, W2 - 1)
                    else:
                        x2_coord = j  # j is already the x-coordinate in the line
                    
                    all_xy1.append([x1_coord, y_coord])
                    all_xy2.append([x2_coord, y_coord])
    
    with _profile_timer("Rectified Post-processing", enable_profiling):
        if len(all_xy1) == 0:
            # No matches found
            if ret_xy:
                return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
            else:
                return np.array([]), np.array([])
        
        xy1 = np.array(all_xy1)
        xy2 = np.array(all_xy2)
        
        if not ret_xy:
            # Convert to linear indices
            idx1 = xy1[:, 0] + W1 * xy1[:, 1]
            idx2 = xy2[:, 0] + W2 * xy2[:, 1]
            return idx1.astype(np.int32), idx2.astype(np.int32)
    
    return xy1, xy2


def extract_correspondences_nonsym(A, B, confA, confB, subsample=8, device=None, ptmap_key='pred_desc', pixel_tol=0):
    if '3d' in ptmap_key:
        opt = dict(device='cpu', workers=32)
    else:
        opt = dict(device=device, dist='dot', block_size=2**13)

    # matching the two pairs
    idx1 = []
    idx2 = []
    # merge corres from opposite pairs
    HA, WA = A.shape[:2]
    HB, WB = B.shape[:2]
    if pixel_tol == 0:
        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample, ret_xy=False, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample, ret_xy=False, **opt)
    else:
        S = subsample
        yA, xA = np.mgrid[S // 2:HA:S, S // 2:WA:S].reshape(2, -1)
        yB, xB = np.mgrid[S // 2:HB:S, S // 2:WB:S].reshape(2, -1)

        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=(xA, yA), ret_xy=False, pixel_tol=pixel_tol, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=(xB, yB), ret_xy=False, pixel_tol=pixel_tol, **opt)

    idx1 = np.r_[nn1to2[0], nn2to1[1]]
    idx2 = np.r_[nn1to2[1], nn2to1[0]]

    c1 = confA.ravel()[idx1]
    c2 = confB.ravel()[idx2]

    xy1, xy2, idx = merge_corres(idx1, idx2, (HA, WA), (HB, WB), ret_xy=True, ret_index=True)
    conf = np.minimum(c1[idx], c2[idx])
    corres = (xy1.copy(), xy2.copy(), conf)
    return todevice(corres, device)
