'''
-----------------------
| Blake Troutman 2025 |
-----------------------


This script is a hackathon project to densify a COLMAP SFM reconstruction using MASt3R.

It is designed to be a simple script that can be run on a single machine with a GPU.

It is not designed to be a production-ready script.

Since this script is a hackathon project, manage your expectations accordingly when parsing through the code.


Any bad code you find here is the result of Cursor; all the good code is mine. 

'''

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import pycolmap
from pathlib import Path
import json
import random
import torch
from PIL import Image
import cv2
import time
import argparse

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r 
from dust3r.utils.image import load_images
from dust3r.inference import inference

NO_POINT = 18446744073709551615
DEFAULT_PAIRS = 'pairs.json'

def compute_robust_bounding_box(reconstruction, min_visibility=3, padding_factor=0.1):
    """
    Compute a robust bounding box from COLMAP 3D points with good visibility.
    
    Args:
        reconstruction: COLMAP reconstruction object
        min_visibility: Minimum number of images a point must be visible in
        padding_factor: Additional padding as fraction of bounding box size
    
    Returns:
        bbox_min, bbox_max: 3D coordinates of bounding box corners
    """
    robust_points = []
    
    # Collect 3D points with sufficient visibility
    for point_id, point3D in reconstruction.points3D.items():
        if len(point3D.track.elements) >= min_visibility:
            robust_points.append(point3D.xyz)
    
    if len(robust_points) == 0:
        print(f"Warning: No points found with visibility >= {min_visibility}")
        # Fallback to all points
        robust_points = [point3D.xyz for point3D in reconstruction.points3D.values()]
    
    if len(robust_points) == 0:
        print("Warning: No 3D points found in reconstruction")
        return None, None
    
    robust_points = np.array(robust_points)
    
    # Compute percentile-based bounding box to be robust to outliers
    bbox_min = np.percentile(robust_points, 5, axis=0)   # 5th percentile
    bbox_max = np.percentile(robust_points, 95, axis=0)  # 95th percentile
    
    # Add padding
    bbox_size = bbox_max - bbox_min
    padding = bbox_size * padding_factor
    bbox_min -= padding
    bbox_max += padding
    
    print(f"Computed robust bounding box from {len(robust_points)} points (min_visibility={min_visibility})")
    print(f"  Min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
    print(f"  Max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
    print(f"  Size: [{bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f}]")
    
    return bbox_min, bbox_max


def filter_points_by_bounding_box(points, colors, bbox_min, bbox_max):
    """
    Filter points and colors to only include those within the bounding box.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        bbox_min, bbox_max: 3D coordinates of bounding box corners
    
    Returns:
        filtered_points, filtered_colors: Arrays containing only points inside bbox
    """
    if bbox_min is None or bbox_max is None:
        return points, colors
    
    # Check which points are inside the bounding box
    inside_mask = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
    
    filtered_points = points[inside_mask]
    filtered_colors = colors[inside_mask]
    
    print(f"Filtered {len(points)} points to {len(filtered_points)} points inside bounding box")
    
    return filtered_points, filtered_colors


def optimized_fast_matching(desc1, desc2, subsample_factor=8, device='cuda', 
                           block_size=2**13):
    """
    Wrapper around fast_reciprocal_NNs with simple optimizations.
    Focus on practical speed improvements without replacing the core algorithm.
    """
    # Use the existing highly optimized fast_reciprocal_NNs
    # Just add some simple optimizations like better block sizes
    
    # Determine optimal block size based on GPU memory and descriptor size
    H1, W1, D = desc1.shape
    H2, W2, _ = desc2.shape
    
    # Adaptive block size based on descriptor dimensions and GPU memory
    if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
        # Larger block size for better GPU utilization
        optimal_block_size = min(block_size, 2**14)  # Increase from 2**13 to 2**14
    else:
        optimal_block_size = block_size
    
    # Use the optimized fast_reciprocal_NNs with better parameters
    from mast3r.fast_nn import fast_reciprocal_NNs
    
    matches_img1, matches_img2 = fast_reciprocal_NNs(
        desc1, desc2, 
        subsample_or_initxy1=subsample_factor, 
        device=device, 
        dist='dot', 
        block_size=optimal_block_size
    )
    
    return matches_img1, matches_img2


def compute_depth_from_pair(reconstruction, img1_id, img2_id, matches_img1, matches_img2, img_dir):
    """
    Compute depth values for matches in the reference frame (img1) from a pair.
    Returns depths and 3D points in world coordinates.
    """
    # get triangulated points
    K1 = reconstruction.images[img1_id].camera.calibration_matrix()
    K2 = reconstruction.images[img2_id].camera.calibration_matrix()
    P1 = K1 @ reconstruction.images[img1_id].cam_from_world().matrix()
    P2 = K2 @ reconstruction.images[img2_id].cam_from_world().matrix()

    m1_undistort = matches_img1.astype(np.float64).T
    m2_undistort = matches_img2.astype(np.float64).T

    # get distortion parameters
    dist_keys_1 = reconstruction.images[img1_id].camera.params_info.split(', ')
    dist_keys_2 = reconstruction.images[img2_id].camera.params_info.split(', ')
    dist1 = {}
    dist2 = {}
    for j in range(len(dist_keys_1)):
        dist1[dist_keys_1[j]] = reconstruction.images[img1_id].camera.params[j]
    for j in range(len(dist_keys_2)):
        dist2[dist_keys_2[j]] = reconstruction.images[img2_id].camera.params[j]

    dist_coeffs_1 = np.array([dist1.get('k1', 0), dist1.get('k2', 0), dist1.get('p1', 0), dist1.get('p2', 0), 
                             dist1.get('k3', 0), dist1.get('k4', 0), dist1.get('k5', 0), dist1.get('k6', 0)])
    dist_coeffs_2 = np.array([dist2.get('k1', 0), dist2.get('k2', 0), dist2.get('p1', 0), dist2.get('p2', 0), 
                             dist2.get('k3', 0), dist2.get('k4', 0), dist2.get('k5', 0), dist2.get('k6', 0)])

    # undistort the matches
    m1_undistort = cv2.undistortPoints(m1_undistort, K1, dist_coeffs_1, P=K1)
    m2_undistort = cv2.undistortPoints(m2_undistort, K2, dist_coeffs_2, P=K2)

    triangulated_points = cv2.triangulatePoints(P1, P2, m1_undistort, m2_undistort)
    triangulated_points = triangulated_points / triangulated_points[3, :]
    triangulated_points = triangulated_points[:3, :]
    triangulated_points = triangulated_points.transpose()

    # Compute depths in camera 1's reference frame
    img1_cam = reconstruction.images[img1_id].cam_from_world()
    img1_R = img1_cam.rotation.matrix()
    img1_t = img1_cam.translation
    
    # Transform points to camera 1's coordinate system
    points_cam1 = (img1_R @ triangulated_points.T + img1_t.reshape(-1, 1)).T
    depths = points_cam1[:, 2]  # Z coordinate is depth
    
    return depths, triangulated_points


def check_depth_consistency(pixel_depths, threshold=0.05, min_validations=2):
    """
    Find the depth value with maximum number of validations (robust to outliers).
    For each depth, count how many other depths are within threshold% of it.
    Returns the average of all depths consistent with the most validated depth.
    
    pixel_depths: list of depth values for the same pixel from different pairings
    threshold: maximum allowed relative depth variation (e.g., 0.05 for 5%)
    min_validations: minimum number of validations required
    Returns: (avg_consistent_depth, is_consistent, num_validations)
    """
    if len(pixel_depths) < 2:
        if len(pixel_depths) == 1:
            return pixel_depths[0], True, 1
        else:
            return None, False, 0
    
    pixel_depths = np.array(pixel_depths)
    # Remove invalid depths (negative or very small)
    valid_depths = pixel_depths[pixel_depths > 0.1]
    
    if len(valid_depths) < 2:
        if len(valid_depths) == 1:
            return valid_depths[0], True, 1
        else:
            return None, False, 0
    
    # For each depth, count how many other depths are within threshold% of it
    best_candidate_depth = None
    max_validations = 0
    
    for i, candidate_depth in enumerate(valid_depths):
        # Count validations for this candidate depth
        validations = 0
        for j, other_depth in enumerate(valid_depths):
            if i != j:  # Don't count the depth against itself
                relative_error = abs(other_depth - candidate_depth) / candidate_depth
                if relative_error <= threshold:
                    validations += 1
        
        # Add 1 to count the candidate depth itself
        validations += 1
        
        if validations > max_validations:
            max_validations = validations
            best_candidate_depth = candidate_depth
    
    # Now collect all depths that are consistent with the best candidate depth
    # and return their average for smoother results
    if best_candidate_depth is not None:
        consistent_depths = []
        for depth in valid_depths:
            relative_error = abs(depth - best_candidate_depth) / best_candidate_depth
            if relative_error <= threshold:
                consistent_depths.append(depth)
        
        # Return average of all consistent depths
        avg_consistent_depth = np.mean(consistent_depths)
        is_consistent = max_validations >= min_validations
        return avg_consistent_depth, is_consistent, max_validations
    else:
        return None, False, 0


def densify_with_consistency_check(reconstruction, img_dir, pairs, batch_size=20, sampling_factor=8, 
                                 force_cpu=False, verbose=False, min_consistent_pairs=3, 
                                 depth_consistency_threshold=0.05, enable_fast_matching=False,
                                 block_size_power=14):
    """
    Main function that implements multi-pairing consistency checking for each frame.
    """
    # fixed params for MASt3R
    model_w = 512
    model_h = 384
    size = 512
    model_path = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'

    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = AsymmetricMASt3R.from_pretrained(model_path, verbose=verbose).to(device)

    # Dictionary to store final consistent point clouds per frame
    frame_clouds = {}
    
    # Process each frame
    for frame_id, partner_ids in pairs.items():
        if not partner_ids:  # Skip frames with no partners
            continue
            
        print(f"Processing frame {frame_id} with {len(partner_ids)} partners...")
        
        # Store depth maps from different pairings for this frame
        pixel_depth_maps = {}  # pixel_coord -> [depth1, depth2, ...]
        pixel_points_maps = {}  # pixel_coord -> [point1, point2, ...]
        pixel_colors = {}      # pixel_coord -> color
        
        # Process each pairing for this frame
        for partner_id in partner_ids:
            if frame_id not in reconstruction.images or partner_id not in reconstruction.images:
                continue
                
            # Load images for this pair
            img1_name = reconstruction.images[frame_id].name
            img2_name = reconstruction.images[partner_id].name
            img1_path = os.path.join(img_dir, img1_name)
            img2_path = os.path.join(img_dir, img2_name)
            
            try:
                images = load_images([img1_path, img2_path], size=size)
                image_pair = tuple([images[0], images[1]])
                
                # Get predictions
                output = inference([image_pair], model=model, device=device, batch_size=1, verbose=verbose)
                
                view1, pred1 = output['view1'], output['pred1']
                view2, pred2 = output['view2'], output['pred2']
                
                desc1, desc2 = pred1['desc'][0].squeeze(0).detach(), pred2['desc'][0].squeeze(0).detach()
                
                # Use optimized matching
                if enable_fast_matching:
                    matches_img1, matches_img2 = optimized_fast_matching(
                        desc1, desc2, 
                        subsample_factor=sampling_factor, 
                        device=device,
                        block_size=2**block_size_power
                    )
                else:
                    matches_img1, matches_img2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=sampling_factor, device=device, dist='dot', block_size=2**13)
                
                # Filter valid matches
                H1, W1 = view1['true_shape'][0]
                valid_matches_img1 = (matches_img1[:, 0] >= 3) & (matches_img1[:, 1] >= 3) & (matches_img1[:, 0] < int(W1) - 3) & (matches_img1[:, 1] < int(H1) - 3)
                
                H2, W2 = view2['true_shape'][0]
                valid_matches_img2 = (matches_img2[:, 0] >= 3) & (matches_img2[:, 1] >= 3) & (matches_img2[:, 0] < int(W2) - 3) & (matches_img2[:, 1] < int(H2) - 3)
                
                valid_matches = valid_matches_img1 & valid_matches_img2
                
                matches_img1 = matches_img1[valid_matches]
                matches_img2 = matches_img2[valid_matches]
                
                if matches_img1.shape[0] < 1:
                    continue
                
                # Rescale matches to original image size
                cam1 = reconstruction.images[frame_id].camera
                cam2 = reconstruction.images[partner_id].camera
                w1_scale = cam1.width / model_w
                h1_scale = cam1.height / model_h
                w2_scale = cam2.width / model_w
                h2_scale = cam2.height / model_h
                
                matches_img1[:, 0] = matches_img1[:, 0] * w1_scale
                matches_img1[:, 1] = matches_img1[:, 1] * h1_scale
                matches_img2[:, 0] = matches_img2[:, 0] * w2_scale
                matches_img2[:, 1] = matches_img2[:, 1] * h2_scale
                
                # Compute depths and 3D points
                depths, points_3d = compute_depth_from_pair(reconstruction, frame_id, partner_id, matches_img1, matches_img2, img_dir)
                
                # Store depth information per pixel location
                image = Image.open(img1_path)
                img_width, img_height = image.size
                
                for i, (depth, point_3d) in enumerate(zip(depths, points_3d)):
                    if depth <= 0:  # Skip invalid depths
                        continue
                        
                    x, y = matches_img1[i]
                    # Round to nearest pixel
                    pixel_coord = (int(round(x)), int(round(y)))
                    
                    # Skip out-of-bounds pixels
                    if pixel_coord[0] < 0 or pixel_coord[0] >= img_width or pixel_coord[1] < 0 or pixel_coord[1] >= img_height:
                        continue
                    
                    if pixel_coord not in pixel_depth_maps:
                        pixel_depth_maps[pixel_coord] = []
                        pixel_points_maps[pixel_coord] = []
                        # Get color for this pixel
                        pixel_colors[pixel_coord] = np.array(image.getpixel(pixel_coord), dtype=np.float32) / 255.0
                    
                    pixel_depth_maps[pixel_coord].append(depth)
                    pixel_points_maps[pixel_coord].append(point_3d)
                    
            except Exception as e:
                print(f"Error processing pair {frame_id}-{partner_id}: {e}")
                continue
        
        # Now check consistency and build final point cloud for this frame
        consistent_points = []
        consistent_colors = []
        
        for pixel_coord, depths in pixel_depth_maps.items():
            if len(depths) >= min_consistent_pairs:
                avg_depth, is_consistent, num_validations = check_depth_consistency(
                    depths, depth_consistency_threshold, min_consistent_pairs)
                
                if is_consistent and avg_depth is not None:
                    # Find the point closest to the average validated depth
                    best_idx = np.argmin(np.abs(np.array(depths) - avg_depth))
                    best_point = pixel_points_maps[pixel_coord][best_idx]
                    
                    consistent_points.append(best_point)
                    consistent_colors.append(pixel_colors[pixel_coord])
        
        if len(consistent_points) > 0:
            frame_clouds[frame_id] = {
                'points': np.array(consistent_points),
                'colors': np.array(consistent_colors)
            }
            print(f"Frame {frame_id}: Generated {len(consistent_points)} consistent points from {len(pixel_depth_maps)} total pixel matches")
        else:
            print(f"Frame {frame_id}: No consistent points found")
    
    return frame_clouds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene_dir', type=str, required=True, help='Path to the scene directory containing sparse/ and images/ subdirectories')
    parser.add_argument('-o', '--output_dir', type=str, required=False, help='Output directory name (relative to scene_dir) for all output files', default='densified_output')
    parser.add_argument('-u', '--use_existing_pairs', action='store_true', help='Use existing pairs from JSON file if it exists')
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='Batch size for inference. 1 will always work, but will be slower. 24GB GPU can handle batch size of 30', default=1)
    parser.add_argument('-c', '--force_cpu', action='store_true', help='Force CPU inference instead of CUDA')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-f', '--sampling_factor', type=int, required=False, help='Sampling factor for point triangulation. Lower = denser. User powers of 2. Default = 8', default=8)
    parser.add_argument('-m', '--min_feature_coverage', type=float, required=False, help='Minimum proportion of image that must be covered by shared points to be considered a good match. Default = 0.6', default=0.6)
    # New parameters for multi-pairing consistency
    parser.add_argument('--max_pairs_per_image', type=int, required=False, help='Maximum number of pairs to compute per image for consistency checking. Default = 7', default=7)
    parser.add_argument('--min_consistent_pairs', type=int, required=False, help='Minimum number of consistent pairings required to keep a point. Default = 3', default=3)
    parser.add_argument('--depth_consistency_threshold', type=float, required=False, help='Depth consistency threshold as percentage (e.g., 0.05 for 5%%). Default = 0.05', default=0.05)
    parser.add_argument('--enable_consistency_check', action='store_true', help='Enable multi-pairing consistency checking')
    # New parameters for bounding box filtering
    parser.add_argument('--enable_bbox_filter', action='store_true', help='Enable bounding box filtering based on COLMAP 3D points')
    parser.add_argument('--min_point_visibility', type=int, required=False, help='Minimum visibility (number of images) for COLMAP points used in bounding box computation. Default = 3', default=3)
    parser.add_argument('--bbox_padding_factor', type=float, required=False, help='Additional padding for bounding box as fraction of size. Default = 0.1', default=0.1)
    # New parameters for matching optimization
    parser.add_argument('--enable_fast_matching', action='store_true', help='Enable optimized fast matching with better block sizes')
    parser.add_argument('--block_size_power', type=int, required=False, help='Block size as power of 2 (e.g., 14 for 2^14). Default = 14', default=14)
    args = parser.parse_args()

    scene_dir = args.scene_dir
    reconstruction_path = os.path.join(scene_dir, 'sparse')
    img_dir = os.path.join(scene_dir, 'images')
    
    # Create output directory structure
    output_dir = os.path.join(scene_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Set up output file paths
    output_path = os.path.join(output_dir, 'dense.ply')
    pairs_path = os.path.join(output_dir, DEFAULT_PAIRS)
    
    # Validate scene directory structure
    if not os.path.exists(scene_dir):
        raise ValueError(f"Scene directory does not exist: {scene_dir}")
    if not os.path.exists(reconstruction_path):
        raise ValueError(f"Sparse reconstruction directory does not exist: {reconstruction_path}")
    if not os.path.exists(img_dir):
        raise ValueError(f"Images directory does not exist: {img_dir}")
    
    use_existing_pairs = args.use_existing_pairs
    batch_size = args.batch_size
    force_cpu = args.force_cpu
    verbose = args.verbose
    sampling_factor = args.sampling_factor
    min_feature_coverage = args.min_feature_coverage
    # New parameters
    max_pairs_per_image = args.max_pairs_per_image
    min_consistent_pairs = args.min_consistent_pairs
    depth_consistency_threshold = args.depth_consistency_threshold
    enable_consistency_check = args.enable_consistency_check
    # Bounding box filtering parameters
    enable_bbox_filter = args.enable_bbox_filter
    min_point_visibility = args.min_point_visibility
    bbox_padding_factor = args.bbox_padding_factor
    # Fast matching parameters
    enable_fast_matching = args.enable_fast_matching
    block_size_power = args.block_size_power

    if use_existing_pairs and not os.path.exists(pairs_path):
        print(f"Pairs file {pairs_path} does not exist. Falling back to generating new pairs.")
        use_existing_pairs = False

    # Print configuration summary
    print(f"Scene directory: {scene_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dense point cloud will be saved to: {output_path}")
    print(f"Pairs file: {pairs_path}")
    print()

    torch.cuda.empty_cache()
    start_time = time.time()

    print(f"Loading reconstruction from {reconstruction_path}...")
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    
    # Compute robust bounding box if filtering is enabled
    bbox_min, bbox_max = None, None
    if enable_bbox_filter:
        bbox_min, bbox_max = compute_robust_bounding_box(
            reconstruction, 
            min_visibility=min_point_visibility, 
            padding_factor=bbox_padding_factor
        )
    
    # Save configuration for debugging/reproducibility
    config_path = os.path.join(output_dir, 'config.json')
    config = {
        'scene_dir': scene_dir,
        'output_dir': args.output_dir,
        'batch_size': batch_size,
        'sampling_factor': sampling_factor,
        'min_feature_coverage': min_feature_coverage,
        'max_pairs_per_image': max_pairs_per_image,
        'min_consistent_pairs': min_consistent_pairs,
        'depth_consistency_threshold': depth_consistency_threshold,
        'enable_consistency_check': enable_consistency_check,
        'enable_bbox_filter': enable_bbox_filter,
        'min_point_visibility': min_point_visibility,
        'bbox_padding_factor': bbox_padding_factor,
        'enable_fast_matching': enable_fast_matching,
        'block_size_power': block_size_power,
        'force_cpu': force_cpu,
        'verbose': verbose
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to {config_path}")

    if not use_existing_pairs:
        if enable_consistency_check:
            pairs = get_multiple_pairs_per_image(reconstruction, max_pairs_per_image=max_pairs_per_image, min_feature_coverage=min_feature_coverage)
        else:
            pairs = get_best_pairs(reconstruction, min_feature_coverage=min_feature_coverage)
        with open(pairs_path, 'w') as f:
            print(f"Saving pairs to {pairs_path}...")
            json.dump(pairs, f, indent=4)
    else:
        pairs = json.load(open(pairs_path))
        print(f"Loaded pairs from {pairs_path}...")

    if enable_consistency_check:
        densified_frames = densify_with_consistency_check(
            reconstruction, img_dir, pairs, 
            batch_size=batch_size, 
            sampling_factor=sampling_factor, 
            force_cpu=force_cpu, 
            verbose=verbose,
            min_consistent_pairs=min_consistent_pairs,
            depth_consistency_threshold=depth_consistency_threshold,
            enable_fast_matching=enable_fast_matching,
            block_size_power=block_size_power
        )
    else:
        densified_pairs = densify_pairs_mast3r_batch(
            reconstruction, img_dir, pairs, 
            batch_size=batch_size, 
            sampling_factor=sampling_factor, 
            force_cpu=force_cpu, 
            verbose=verbose,
            enable_fast_matching=enable_fast_matching,
            block_size_power=block_size_power
        )
        # Convert to the expected format for backwards compatibility
        densified_frames = {}
        for img1, data in densified_pairs.items():
            densified_frames[int(img1)] = data

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

    # Combine all points from different frames
    all_points = None
    all_colors = None
    total_frames_processed = 0
    
    for frame_id, frame_data in densified_frames.items():
        total_frames_processed += 1
        if all_points is None:
            all_points = frame_data['points']
            all_colors = frame_data['colors']
        else:
            all_points = np.concatenate([all_points, frame_data['points']], axis=0)
            all_colors = np.concatenate([all_colors, frame_data['colors']], axis=0)

    # Apply bounding box filtering if enabled
    if enable_bbox_filter and bbox_min is not None and bbox_max is not None:
        all_points, all_colors = filter_points_by_bounding_box(all_points, all_colors, bbox_min, bbox_max)

    # Check if we have any points left after filtering
    if all_points is None or len(all_points) == 0:
        print("No points remaining after filtering. Exiting.")
        return

    # Create and save point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Save point cloud
    o3d.io.write_point_cloud(output_path, pcd)
    
    # Save processing summary
    summary_path = os.path.join(output_dir, 'processing_summary.json')
    summary = {
        'total_points': len(all_points),
        'total_frames_processed': total_frames_processed,
        'processing_time_seconds': end_time - start_time,
        'bounding_box_filtering_enabled': enable_bbox_filter,
        'consistency_checking_enabled': enable_consistency_check
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print final summary
    print()
    print("=" * 60)
    print("DENSIFICATION COMPLETED")
    print("=" * 60)
    print(f"Dense point cloud: {output_path}")
    print(f"Total points: {len(all_points):,}")
    print(f"Frames processed: {total_frames_processed}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print()
    print("Output files saved to:")
    print(f"  - Dense point cloud: {os.path.relpath(output_path, scene_dir)}")
    print(f"  - Pairs file: {os.path.relpath(pairs_path, scene_dir)}")
    print(f"  - Configuration: {os.path.relpath(config_path, scene_dir)}")
    print(f"  - Summary: {os.path.relpath(summary_path, scene_dir)}")
    print("=" * 60)

    # Display the point cloud
    print("Opening point cloud visualization...")
    o3d.visualization.draw_geometries([pcd])


def densify_pairs_mast3r_batch(reconstruction, img_dir, pairs, batch_size=20, sampling_factor=8, force_cpu=False, verbose=False, enable_fast_matching=False, block_size_power=14) -> dict[int, np.ndarray]:

    # fixed params for MASt3R
    model_w = 512
    model_h = 384
    size = 512
    model_path = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'

    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = AsymmetricMASt3R.from_pretrained(model_path, verbose=verbose).to(device)

    # initialize empty dict of image_id to np.ndarray of point3D_ids
    densified_pairs = {}

    def densify_batch(pair_batch):
        image_batch = []
        image_batch_sizes = []
        img_id_to_idx = {}
        image_paths = []

        # get all image indices that appear in pair_batch
        image_indices = set()
        for img1, img2 in pair_batch.items():
            if int(img1) not in reconstruction.images or int(img2) not in reconstruction.images:
                continue
            image_indices.add(int(img1))
            image_indices.add(int(img2))

        # Skip this batch if no valid images found
        if not image_indices:
            print("Skipping batch - no valid images found in reconstruction...")
            return

        # get the image paths for the image indices
        for image_idx in image_indices:
            img_name = reconstruction.images[image_idx].name
            img_path = os.path.join(img_dir, img_name)
            image_paths.append(img_path)
            img_id_to_idx[image_idx] = len(image_paths) - 1

        # load all images in the batch
        print("loading images...")
        images = load_images(image_paths, size=size)

        # pair up the images
        for img1, img2 in pair_batch.items():
            if int(img1) not in reconstruction.images or int(img2) not in reconstruction.images:
                continue
            img1_idx = img_id_to_idx[int(img1)]
            img2_idx = img_id_to_idx[int(img2)]
            image_batch.append(tuple([images[img1_idx], images[img2_idx]]))
            cam_1 = reconstruction.images[int(img1)].camera
            cam_2 = reconstruction.images[int(img2)].camera
            image_batch_sizes.append([tuple([cam_1.width, cam_1.height]), tuple([cam_2.width, cam_2.height])])

        # get predictions for each image pair
        output = inference(image_batch, model=model, device=device, batch_size=batch_size, verbose=verbose)

        # triangulate each image pair
        print("triangulating...")
        for i in range(len(image_batch)):
            img1, img2 = list(pair_batch.items())[i]

            if int(img1) not in reconstruction.images or int(img2) not in reconstruction.images:
                continue

            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']

            desc1, desc2 = pred1['desc'][i].squeeze(0).detach(), pred2['desc'][i].squeeze(0).detach()

            # Use optimized matching
            if enable_fast_matching:
                matches_img1, matches_img2 = optimized_fast_matching(
                    desc1, desc2, 
                    subsample_factor=sampling_factor, 
                    device=device,
                    block_size=2**block_size_power
                )
            else:
                matches_img1, matches_img2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=sampling_factor, device=device, dist='dot', block_size=2**13)

            # ignore small border around the edge
            H1, W1 = view1['true_shape'][i]
            valid_matches_img1 = (matches_img1[:, 0] >= 3) & (matches_img1[:, 1] >= 3) & (matches_img1[:, 0] < int(W1) - 3) & (matches_img1[:, 1] < int(H1) - 3)

            H2, W2 = view2['true_shape'][i]
            valid_matches_img2 = (matches_img2[:, 0] >= 3) & (matches_img2[:, 1] >= 3) & (matches_img2[:, 0] < int(W2) - 3) & (matches_img2[:, 1] < int(H2) - 3)

            valid_matches = valid_matches_img1 & valid_matches_img2

            matches_img1 = matches_img1[valid_matches]
            matches_img2 = matches_img2[valid_matches]

            if matches_img1.shape[0] < 1 or matches_img2.shape[0] < 1:
                continue

            # rescale matches to original image size
            w1, h1 = image_batch_sizes[i][0]
            w1_scale = w1 / model_w
            h1_scale = h1 / model_h
            w2, h2 = image_batch_sizes[i][1]
            w2_scale = w2 / model_w
            h2_scale = h2 / model_h

            matches_img1[:, 0] = matches_img1[:, 0] * w1_scale
            matches_img1[:, 1] = matches_img1[:, 1] * h1_scale
            matches_img2[:, 0] = matches_img2[:, 0] * w2_scale
            matches_img2[:, 1] = matches_img2[:, 1] * h2_scale

            # get colors from image 1
            img1_path = os.path.join(img_dir, reconstruction.images[int(img1)].name)
            img1_color = np.zeros((len(matches_img1), 3))
            image = Image.open(img1_path)
            img_width, img_height = image.size  # Get actual image dimensions
            
            for j in range(len(matches_img1)):
                x, y = matches_img1[j]
                # Convert to integers and clamp to image bounds
                x = int(np.clip(x, 0, img_width - 1))
                y = int(np.clip(y, 0, img_height - 1))
                img1_color[j] = np.array(image.getpixel((x, y)), dtype=np.float32) / 255.0

            # get triangulated points
            K1 = reconstruction.images[int(img1)].camera.calibration_matrix()
            K2 = reconstruction.images[int(img2)].camera.calibration_matrix()
            P1 = K1 @ reconstruction.images[int(img1)].cam_from_world().matrix()
            P2 = K2 @ reconstruction.images[int(img2)].cam_from_world().matrix()

            m1_undistort = matches_img1.astype(np.float64).T
            m2_undistort = matches_img2.astype(np.float64).T

            # get distortion parameters
            dist_keys_1 = reconstruction.images[int(img1)].camera.params_info.split(', ')
            dist_keys_2 = reconstruction.images[int(img2)].camera.params_info.split(', ')
            dist1 = {}
            dist2 = {}
            for j in range(len(dist_keys_1)):
                dist1[dist_keys_1[j]] = reconstruction.images[int(img1)].camera.params[j]
            for j in range(len(dist_keys_2)):
                dist2[dist_keys_2[j]] = reconstruction.images[int(img2)].camera.params[j]

            dist_coeffs_1 = np.array([dist1.get('k1', 0), dist1.get('k2', 0), dist1.get('p1', 0), dist1.get('p2', 0), dist1.get('k3', 0), dist1.get('k4', 0), dist1.get('k5', 0), dist1.get('k6', 0)])
            dist_coeffs_2 = np.array([dist2.get('k1', 0), dist2.get('k2', 0), dist2.get('p1', 0), dist2.get('p2', 0), dist2.get('k3', 0), dist2.get('k4', 0), dist2.get('k5', 0), dist2.get('k6', 0)])

            # undistort the matches
            m1_undistort = cv2.undistortPoints(m1_undistort, K1, dist_coeffs_1, P=K1)
            m2_undistort = cv2.undistortPoints(m2_undistort, K2, dist_coeffs_2, P=K2)

            triangulated_points = cv2.triangulatePoints(P1, P2, m1_undistort, m2_undistort)
            triangulated_points = triangulated_points / triangulated_points[3, :]
            triangulated_points = triangulated_points[:3, :]
            triangulated_points = triangulated_points.transpose()

            # filter out points that are too far from the cameras
            img1_cam = reconstruction.images[int(img1)].cam_from_world()
            img1_R = img1_cam.rotation.matrix()
            img1_t = img1_cam.translation

            img2_cam = reconstruction.images[int(img2)].cam_from_world()
            img2_R = img2_cam.rotation.matrix()
            img2_t = img2_cam.translation

            # image center of img1
            img1_C = -img1_R.T @ img1_t

            # image center of img2
            img2_C = -img2_R.T @ img2_t

            baseline = np.linalg.norm(img1_C - img2_C)

            # filter out points that are too far from the cameras (parallax stand-in)
            dist_to_cameras = np.linalg.norm(triangulated_points - img1_C, axis=1)
            triangulated_points = triangulated_points[dist_to_cameras < 20 * baseline, :]
            img1_color = img1_color[dist_to_cameras < 20 * baseline]

            # filter out points that are behind the camera
            behind_camera = (img1_R @ triangulated_points.T - img1_t.reshape(-1, 1))[2] < 0
            triangulated_points = triangulated_points[~behind_camera, :]
            img1_color = img1_color[~behind_camera]

            # TODO: filter out points that violate epipolar constraint

            # # [OPTIONAL] add the image point to the triangulated points
            # triangulated_points = np.concatenate([triangulated_points, img1_C.reshape(-1, 3)], axis=0)
            # img1_color = np.concatenate([img1_color, np.array([1, 0, 0]).reshape(1, 3)], axis=0)

            densified_pairs[img1] = {
                'points': triangulated_points,
                'colors': img1_color
            }

    num_batches = len(pairs) // batch_size
    batches = []
    for i in range(num_batches):
        batch = {k: v for k, v in pairs.items() if k in list(pairs.keys())[i*batch_size:(i+1)*batch_size]}
        batches.append(batch)

    for batch in batches:
        densify_batch(batch)
    
    return densified_pairs


def get_multiple_pairs_per_image(reconstruction, max_pairs_per_image=7, min_points=100, parallax_sample_size=100, min_feature_coverage=0.6):
    """
    Find multiple pairs for each image instead of just one best pair.
    Returns a dictionary where each key is an image_id and value is a list of partner image_ids.
    """
    
    class MatchCandidate:
        def __init__(self, image_id, shared_points):
            self.image_id = image_id
            self.shared_points = shared_points
            self.avg_parallax_angle = 0
            self.x_coverage = 0
            self.y_coverage = 0

    # initialize empty pair map (image ids to list of partner ids)
    pairs = {}
    for image in reconstruction.images.values():
        pairs[image.image_id] = []
    
    # generate dict of image_id to set of point3D_ids
    image_point3D_ids = {}
    image_point3D_xy = {}
    for image in reconstruction.images.values():
        image_point3D_xy[image.image_id] = {}
        for point2D in image.points2D:
            if point2D.has_point3D():
                image_point3D_xy[image.image_id][point2D.point3D_id] = point2D.xy
                if image_point3D_ids.get(image.image_id, None) is None:
                    image_point3D_ids[image.image_id] = set()
                image_point3D_ids[image.image_id].add(point2D.point3D_id)

    # iterate over all images
    for i, image in enumerate(reconstruction.images.values()):
        print(f"Getting multiple corresponding images for {image.name}... {i}/{len(reconstruction.images) - 1}")
        # identify other images that share at least min_points points
        other_images = [other_image for other_image in reconstruction.images.values() if other_image.image_id != image.image_id]
        match_candidates = []
        for other_image in other_images:
            # get the points that the two images share
            shared_points = image_point3D_ids[image.image_id] & image_point3D_ids[other_image.image_id]
            if len(shared_points) >= min_points:
                match_candidates.append(MatchCandidate(other_image.image_id, shared_points))
        
        # if there are no frames with a good number of shared points, skip this one
        if len(match_candidates) == 0:
            continue
        
        # for each match candidate, compute the feature coverage of the shared points
        for match_candidate in match_candidates:
            shared_points = match_candidate.shared_points
            xs = [image_point3D_xy[image.image_id][point_id][0] for point_id in shared_points]
            ys = [image_point3D_xy[image.image_id][point_id][1] for point_id in shared_points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_range = x_max - x_min
            y_range = y_max - y_min
            match_candidate.x_coverage = x_range / image.camera.width
            match_candidate.y_coverage = y_range / image.camera.height

        # for each match candidate, record the average parallax angle between the two images and each shared point
        for match_candidate in match_candidates:
            other_image = reconstruction.images[match_candidate.image_id]
            shared_points = match_candidate.shared_points
            parallax_angles = []
            sample_size = min(parallax_sample_size, len(shared_points))
            for point_id in random.sample(list(shared_points), sample_size):
                point = reconstruction.points3D[point_id]
                img1_cam = reconstruction.images[image.image_id].cam_from_world()
                img1_R = img1_cam.rotation.matrix()
                img1_t = img1_cam.translation
                img2_cam = reconstruction.images[other_image.image_id].cam_from_world()
                img2_R = img2_cam.rotation.matrix()
                img2_t = img2_cam.translation
                img1_C = -img1_R.T @ img1_t
                img2_C = -img2_R.T @ img2_t
                u = img1_C - point.xyz
                v = img2_C - point.xyz
                parallax_angle = np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))
                parallax_angles.append(parallax_angle)
            
            avg_parallax_angle = np.mean(parallax_angles)
            match_candidate.avg_parallax_angle = avg_parallax_angle

        # Sort candidates by a combination of shared points, feature coverage, and parallax
        match_candidates.sort(key=lambda x: (
            len(x.shared_points) * 
            ((x.x_coverage + x.y_coverage) / 2) * 
            min(x.avg_parallax_angle, 1.0)  # Cap parallax contribution
        ), reverse=True)
        
        # Select top candidates with good parallax (> 0.1 radians ≈ 5.7 degrees)
        good_candidates = []
        for candidate in match_candidates:
            if candidate.avg_parallax_angle > 0.1 and len(good_candidates) < max_pairs_per_image:
                good_candidates.append(candidate.image_id)
        
        # If no good parallax candidates, just take the top ones by shared points
        if len(good_candidates) == 0:
            good_candidates = [candidate.image_id for candidate in match_candidates[:max_pairs_per_image]]
        
        pairs[image.image_id] = good_candidates

    return pairs


def get_best_pairs(reconstruction, min_points=100, parallax_sample_size=100, min_feature_coverage=0.6):
    """Original single-pair selection function"""

    class MatchCandidate:
        def __init__(self, image_id, shared_points):
            self.image_id = image_id
            self.shared_points = shared_points
            self.avg_parallax_angle = 0
            self.x_coverage = 0
            self.y_coverage = 0

    # initialize empty pair map (image ids)
    pairs = {}
    for image in reconstruction.images.values():
        pairs[image.image_id] = -1
    
    # generate dict of image_id to set of point3D_ids
    image_point3D_ids = {}
    image_point3D_xy = {}
    for image in reconstruction.images.values():
        image_point3D_xy[image.image_id] = {}
        for point2D in image.points2D:
            if point2D.has_point3D():
                image_point3D_xy[image.image_id][point2D.point3D_id] = point2D.xy
                if image_point3D_ids.get(image.image_id, None) is None:
                    image_point3D_ids[image.image_id] = set()
                image_point3D_ids[image.image_id].add(point2D.point3D_id)

    # iterate over all images
    for i, image in enumerate(reconstruction.images.values()):
        print(f"Getting best corresponding image for {image.name}... {i}/{len(reconstruction.images) - 1}")
        # identify other images that share at least 100 points
        other_images = [other_image for other_image in reconstruction.images.values() if other_image.image_id != image.image_id]
        match_candidates = []
        for other_image in other_images:
            # get the points that the two images share
            shared_points = image_point3D_ids[image.image_id] & image_point3D_ids[other_image.image_id]
            match_candidates.append(MatchCandidate(other_image.image_id, shared_points))
        
        # sort by number of shared points
        match_candidates.sort(key=lambda x: len(x.shared_points), reverse=True)
        
        # if there are no frames with a good number of shared points, skip this one
        if len(match_candidates) == 0 or len(match_candidates[0].shared_points) < min_points:
            continue
        
        # filter out match candidates that don't have enough matches
        match_candidates = [match_candidate for match_candidate in match_candidates if len(match_candidate.shared_points) >= min_points]

        # for each match candidate, compute the feature coverage of the shared points
        for match_candidate in match_candidates:
            shared_points = match_candidate.shared_points
            xs = [image_point3D_xy[image.image_id][point_id][0] for point_id in shared_points]
            ys = [image_point3D_xy[image.image_id][point_id][1] for point_id in shared_points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_range = x_max - x_min
            y_range = y_max - y_min
            match_candidate.x_coverage = x_range / image.camera.width
            match_candidate.y_coverage = y_range / image.camera.height

        # order match candidates by average xy coverage
        match_candidates.sort(key=lambda x: (x.x_coverage + x.y_coverage) / 2, reverse=True)
        
        # # filter out match candidates that don't have enough feature coverage in the base image
        # match_candidates = [match_candidate for match_candidate in match_candidates if match_candidate.x_coverage > min_feature_coverage and match_candidate.y_coverage > min_feature_coverage]

        # for each match candidate, record the average parallax angle between the two images and each shared point
        for match_candidate in match_candidates:
            other_image = reconstruction.images[match_candidate.image_id]
            shared_points = match_candidate.shared_points
            parallax_angles = []
            for point_id in random.sample(list(shared_points), parallax_sample_size):
                point = reconstruction.points3D[point_id]
                img1_cam = reconstruction.images[image.image_id].cam_from_world()
                img1_R = img1_cam.rotation.matrix()
                img1_t = img1_cam.translation
                img2_cam = reconstruction.images[other_image.image_id].cam_from_world()
                img2_R = img2_cam.rotation.matrix()
                img2_t = img2_cam.translation
                img1_C = -img1_R.T @ img1_t
                img2_C = -img2_R.T @ img2_t
                u = img1_C - point.xyz
                v = img2_C - point.xyz
                parallax_angle = np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
                parallax_angles.append(parallax_angle)
            
            avg_parallax_angle = np.mean(parallax_angles)
            match_candidate.avg_parallax_angle = avg_parallax_angle

        if len(match_candidates) == 0:
            continue
        
        # make sure the high overlap match has good parallax
        winning_index = -1
        for i in range(len(match_candidates)):
            if match_candidates[i].avg_parallax_angle > 0.1:
                winning_index = i
                break
        if winning_index == -1:
            winning_index = 0
        
        pairs[image.image_id] = match_candidates[winning_index].image_id

    return pairs

if __name__ == "__main__":
    main()