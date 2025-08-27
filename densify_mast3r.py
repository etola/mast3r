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
from contextlib import contextmanager

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs, get_fast_nn_profile_data, clear_fast_nn_profile_data

import mast3r.utils.path_to_dust3r 
from image_io import load_images, initialize_cache, print_cache_stats
from dust3r.inference import inference

import colmap_utils
from colmap_utils import ColmapReconstruction
from config import DensificationConfig, create_config_from_args


# Global profiling dictionary
profile_data = {}

@contextmanager
def profile_timer(operation_name, enabled=True):
    """Context manager for timing operations and accumulating results."""
    if not enabled:
        yield
        return
        
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if operation_name not in profile_data:
            profile_data[operation_name] = {'total_time': 0, 'count': 0, 'times': []}
        profile_data[operation_name]['total_time'] += elapsed
        profile_data[operation_name]['count'] += 1
        profile_data[operation_name]['times'].append(elapsed)


def print_profiling_summary(enabled=True):
    """Print a comprehensive profiling summary."""
    if not enabled:
        return
        
    # Merge fast_nn profiling data into main profile_data
    fast_nn_data = get_fast_nn_profile_data()
    combined_data = profile_data.copy()
    
    # Add fast_nn data with prefixes for clarity
    for operation, data in fast_nn_data.items():
        prefixed_operation = f"FastNN: {operation}"
        combined_data[prefixed_operation] = data
    
    if not combined_data:
        return
        
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)
    
    total_measured_time = sum(data['total_time'] for data in combined_data.values())
    
    # Sort by total time spent
    sorted_operations = sorted(combined_data.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    print(f"{'Operation':<45} {'Total Time':<12} {'Count':<8} {'Avg Time':<12} {'Percentage':<10}")
    print("-" * 90)
    
    for operation, data in sorted_operations:
        total_time = data['total_time']
        count = data['count']
        avg_time = total_time / count if count > 0 else 0
        percentage = (total_time / total_measured_time * 100) if total_measured_time > 0 else 0
        
        print(f"{operation:<45} {total_time:<12.3f} {count:<8} {avg_time:<12.3f} {percentage:<10.1f}%")
    
    print("-" * 90)
    print(f"{'Total measured time:':<45} {total_measured_time:<12.3f}")
    print("=" * 90)


def save_profiling_data(output_path, enabled=True):
    """Save profiling data to a JSON file for analysis."""
    if not enabled or not profile_data:
        return
        
    # Prepare data for JSON serialization (remove numpy arrays etc.)
    json_data = {}
    for operation, data in profile_data.items():
        json_data[operation] = {
            'total_time': float(data['total_time']),
            'count': int(data['count']),
            'avg_time': float(data['total_time'] / data['count']) if data['count'] > 0 else 0.0,
            'times': [float(t) for t in data['times']]
        }
    
    # Add summary statistics
    total_measured_time = sum(data['total_time'] for data in profile_data.values())
    json_data['_summary'] = {
        'total_measured_time': float(total_measured_time),
        'operation_count': len(profile_data),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Profiling data saved to {output_path}")


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
    
    # print(f"Filtered {len(points)} points to {len(filtered_points)} points inside bounding box")
    
    return filtered_points, filtered_colors



def compute_depth_from_pair(reconstruction: ColmapReconstruction, img1_id: int, img2_id: int, matches_img1: np.ndarray, matches_img2: np.ndarray):
    """
    Compute depth values for matches in the reference frame (img1) from a pair.
    Returns depths and 3D points in world coordinates.
    Assumes undistorted images.
    """
    # Get camera matrices
    P1 = reconstruction.get_camera_projection_matrix(img1_id)
    P2 = reconstruction.get_camera_projection_matrix(img2_id)

    # Convert matches to homogeneous coordinates for triangulation
    m1 = matches_img1.astype(np.float64).T
    m2 = matches_img2.astype(np.float64).T

    triangulated_points = cv2.triangulatePoints(P1, P2, m1, m2)
    triangulated_points = triangulated_points / triangulated_points[3, :]
    triangulated_points = triangulated_points[:3, :]
    triangulated_points = triangulated_points.transpose()

    # Compute depths in camera 1's reference frame
    img1_cam = reconstruction.get_image_cam_from_world(img1_id)
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


def densify_with_consistency_check(reconstruction: ColmapReconstruction, pairs: dict, config: DensificationConfig):
    """
    Main function that implements multi-pairing consistency checking for each frame.
    """
    device = config.get_device()
    
    # load the model
    with profile_timer("Model Loading (Consistency Check)", config.enable_profiling):
        model = AsymmetricMASt3R.from_pretrained(config.model_path, verbose=config.verbose).to(device)

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
            if not reconstruction.has_image(frame_id) or not reconstruction.has_image(partner_id):
                continue
                
            # Load images for this pair
            img1_name = reconstruction.get_image_name(frame_id)
            img2_name = reconstruction.get_image_name(partner_id)
            img1_path = os.path.join(config.img_dir, img1_name)
            img2_path = os.path.join(config.img_dir, img2_name)
            
            try:
                with profile_timer("Image Loading (Consistency)", config.enable_profiling):
                    images = load_images([img1_path, img2_path], size=config.size)
                    image_pair = tuple([images[0], images[1]])
                
                # Get predictions
                with profile_timer("Model Inference (Consistency)", config.enable_profiling):
                    output = inference([image_pair], model=model, device=device, batch_size=1, verbose=config.verbose)
                
                view1, pred1 = output['view1'], output['pred1']
                view2, pred2 = output['view2'], output['pred2']
                
                desc1, desc2 = pred1['desc'][0].squeeze(0).detach(), pred2['desc'][0].squeeze(0).detach()
                
                # Compute feature matches
                with profile_timer("Feature Matching (Consistency)", config.enable_profiling):
                    matches_img1, matches_img2 = fast_reciprocal_NNs(
                        desc1, desc2, 
                        subsample_or_initxy1=config.sampling_factor, 
                        device=device, 
                        dist='dot', 
                        block_size=config.get_block_size(),
                        enable_profiling=config.enable_profiling
                    )
                
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
                cam1 = reconstruction.get_image_camera(frame_id)
                cam2 = reconstruction.get_image_camera(partner_id)
                w1_scale = cam1.width / config.model_w
                h1_scale = cam1.height / config.model_h
                w2_scale = cam2.width / config.model_w
                h2_scale = cam2.height / config.model_h
                
                matches_img1[:, 0] = matches_img1[:, 0] * w1_scale
                matches_img1[:, 1] = matches_img1[:, 1] * h1_scale
                matches_img2[:, 0] = matches_img2[:, 0] * w2_scale
                matches_img2[:, 1] = matches_img2[:, 1] * h2_scale
                
                # Compute depths and 3D points
                with profile_timer("Triangulation (Consistency)", config.enable_profiling):
                    depths, points_3d = compute_depth_from_pair(reconstruction, frame_id, partner_id, matches_img1, matches_img2)
                

                
                # Store depth information per pixel location
                image = Image.open(img1_path)
                img_width, img_height = image.size
                
                # Vectorized filtering and processing
                depths = np.array(depths)
                points_3d = np.array(points_3d)
                
                # Filter valid depths
                valid_mask = depths > 0
                valid_depths = depths[valid_mask]
                valid_points = points_3d[valid_mask]
                valid_matches = matches_img1[valid_mask]
                
                if len(valid_depths) == 0:
                    continue
                
                # Round coordinates and check bounds vectorized
                pixel_x = np.round(valid_matches[:, 0]).astype(int)
                pixel_y = np.round(valid_matches[:, 1]).astype(int)
                
                # Bounds checking
                bounds_mask = (
                    (pixel_x >= 0) & (pixel_x < img_width) & 
                    (pixel_y >= 0) & (pixel_y < img_height)
                )
                
                # Apply bounds filter
                valid_depths = valid_depths[bounds_mask]
                valid_points = valid_points[bounds_mask]
                pixel_x = pixel_x[bounds_mask]
                pixel_y = pixel_y[bounds_mask]
                
                if len(valid_depths) == 0:
                    continue
                
                # Vectorized color extraction for all valid pixels
                img_array = np.array(image, dtype=np.float32) / 255.0
                pixel_colors_array = img_array[pixel_y, pixel_x]
                
                # Build dictionaries for valid pixels only
                for i in range(len(valid_depths)):
                    pixel_coord = (pixel_x[i], pixel_y[i])
                    
                    if pixel_coord not in pixel_depth_maps:
                        pixel_depth_maps[pixel_coord] = []
                        pixel_points_maps[pixel_coord] = []
                        pixel_colors[pixel_coord] = pixel_colors_array[i]
                    
                    pixel_depth_maps[pixel_coord].append(valid_depths[i])
                    pixel_points_maps[pixel_coord].append(valid_points[i])
                    
            except Exception as e:
                print(f"Error processing pair {frame_id}-{partner_id}: {e}")
                continue
        
        # Now check consistency and build final point cloud for this frame
        consistent_points = []
        consistent_colors = []
        
        for pixel_coord, depths in pixel_depth_maps.items():
            if len(depths) >= config.min_consistent_pairs:
                avg_depth, is_consistent, num_validations = check_depth_consistency(
                    depths, config.depth_consistency_threshold, config.min_consistent_pairs)
                
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
    parser.add_argument('--cache_memory_gb', type=float, default=16.0, help='Maximum memory for image cache in GB (default: 16.0)')
    parser.add_argument('--disable_profiling', action='store_true', help='Disable performance profiling output')

    parser.add_argument('-f', '--sampling_factor', type=int, required=False, help='Sampling factor for point triangulation. Lower = denser. User powers of 2. Default = 8', default=8)
    parser.add_argument('-m', '--min_feature_coverage', type=float, required=False, help='Minimum proportion of image that must be covered by shared points to be considered a good match. Default = 0.6', default=0.6)
    # New parameters for multi-pairing consistency
    parser.add_argument('--max_pairs_per_image', type=int, required=False, help='Maximum number of pairs to compute per image for consistency checking. Default = 7', default=7)
    parser.add_argument('--min_consistent_pairs', type=int, required=False, help='Minimum number of consistent pairings required to keep a point. Default = 3', default=3)
    parser.add_argument('--depth_consistency_threshold', type=float, required=False, help='Depth consistency threshold as percentage (e.g., 0.05 for 5%%). Default = 0.05', default=0.05)
    parser.add_argument('--enable_consistency_check', action='store_true', help='Enable multi-pairing consistency checking')
    # Bounding box filtering parameters
    parser.add_argument('--disable_bbox_filter', action='store_true', help='Disable bounding box filtering')
    parser.add_argument('--min_point_visibility', type=int, required=False, help='Minimum visibility (number of images) for COLMAP points used in bounding box computation. Default = 3', default=3)
    parser.add_argument('--bbox_padding_factor', type=float, required=False, help='Additional padding for bounding box as fraction of size. Default = 1.0', default=1.0)
    # Point cloud outlier removal parameters
    parser.add_argument('--enable_outlier_removal', action='store_true', help='Enable statistical outlier removal from point cloud')
    parser.add_argument('--outlier_nb_neighbors', type=int, required=False, help='Number of neighbors for outlier removal. Default = 20', default=20)
    parser.add_argument('--outlier_std_ratio', type=float, required=False, help='Standard deviation ratio for outlier removal. Default = 2.0', default=2.0)
    # Matching parameters
    parser.add_argument('--block_size_power', type=int, required=False, help='Block size as power of 2 (e.g., 14 for 2^14). Default = 14', default=14)
    args = parser.parse_args()

    # Create configuration object from arguments
    config = create_config_from_args(args)
    config.setup_output_paths()
    config.validate_paths()
    
    # Print configuration summary
    config.print_summary()
    
    # Initialize image cache for efficient loading and resizing
    cache_memory_gb = getattr(config, 'cache_memory_gb', 16.0)  # Default to 16GB if not specified
    initialize_cache(max_memory_gb=cache_memory_gb)
    
    # Clear any existing profiling data for clean measurements
    clear_fast_nn_profile_data()
    
    # Check if existing pairs file should be used
    if config.use_existing_pairs and not os.path.exists(config.pairs_path):
        print(f"Pairs file {config.pairs_path} does not exist. Falling back to generating new pairs.")
        config.use_existing_pairs = False

    torch.cuda.empty_cache()
    start_time = time.time()

    with profile_timer("Reconstruction Loading", config.enable_profiling):
        print(f"Loading reconstruction from {config.reconstruction_path}...")
        reconstruction: ColmapReconstruction = colmap_utils.load_reconstruction(config.reconstruction_path)
    
    # Compute robust bounding box for global filtering
    bbox_min, bbox_max = None, None
    if not config.disable_bbox_filter:
        with profile_timer("Bounding Box Computation", config.enable_profiling):
            bbox_min, bbox_max = reconstruction.compute_robust_bounding_box(
                min_visibility=config.min_point_visibility, 
                padding_factor=config.bbox_padding_factor
            )
    
    # Save configuration for debugging/reproducibility
    config_path = os.path.join(config.scene_dir, config.output_dir, 'config.json')
    config.save_to_file(config_path)
    print(f"Saved configuration to {config_path}")

    if not config.use_existing_pairs:
        with profile_timer("Pair Selection", config.enable_profiling):
            pairs = reconstruction.get_best_pairs(
                min_feature_coverage=config.min_feature_coverage,
                pairs_per_image=config.max_pairs_per_image
            )
        with profile_timer("Pairs Saving", config.enable_profiling):
            with open(config.pairs_path, 'w') as f:
                print(f"Saving pairs to {config.pairs_path}...")
                json.dump(pairs, f, indent=4)
    else:
        with profile_timer("Pairs Loading", config.enable_profiling):
            pairs = json.load(open(config.pairs_path))
            print(f"Loaded pairs from {config.pairs_path}...")

    if config.enable_consistency_check:
        with profile_timer("Main Densification (Consistency Check)", config.enable_profiling):
            densified_frames = densify_with_consistency_check(reconstruction, pairs, config)
    else:
        with profile_timer("Main Densification (Batch Processing)", config.enable_profiling):
            densified_pairs = densify_pairs_mast3r_batch(reconstruction, pairs, config)
        # Convert to the expected format for backwards compatibility
        densified_frames = {}
        for img1, data in densified_pairs.items():
            densified_frames[int(img1)] = data

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    # Print cache statistics
    print_cache_stats()
    
    # Print profiling summary
    print_profiling_summary(config.enable_profiling)
    
    # Save profiling data to JSON file
    profiling_path = os.path.join(config.scene_dir, config.output_dir, 'profiling_data.json')
    save_profiling_data(profiling_path, config.enable_profiling)

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

    # Apply global bounding box filtering if enabled
    if not config.disable_bbox_filter and bbox_min is not None and bbox_max is not None:
        with profile_timer("Global Bounding Box Filtering", config.enable_profiling):
            all_points, all_colors = filter_points_by_bounding_box(all_points, all_colors, bbox_min, bbox_max)

    # Check if we have any points left after filtering
    if all_points is None or len(all_points) == 0:
        print("No points remaining after filtering. Exiting.")
        return

    # Create and save point cloud
    with profile_timer("Point Cloud Creation", config.enable_profiling):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Apply statistical outlier removal if enabled
    if config.enable_outlier_removal:
        with profile_timer("Statistical Outlier Removal", config.enable_profiling):
            print(f"Removing statistical outliers (neighbors={config.outlier_nb_neighbors}, std_ratio={config.outlier_std_ratio})...")
            points_before = len(pcd.points)
            pcd, outlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=config.outlier_nb_neighbors, 
                std_ratio=config.outlier_std_ratio
            )
            points_after = len(pcd.points)
            outliers_removed = points_before - points_after
            print(f"Removed {outliers_removed:,} outlier points ({outliers_removed/points_before*100:.1f}%)")
            print(f"Final point cloud: {points_after:,} points")

    # Save point cloud
    with profile_timer("Point Cloud Saving", config.enable_profiling):
        o3d.io.write_point_cloud(config.output_path, pcd)
    
    # Save processing summary
    summary_path = os.path.join(config.scene_dir, config.output_dir, 'processing_summary.json')
    final_points = len(pcd.points)
    summary = {
        'total_points_before_filtering': len(all_points),
        'final_points_after_filtering': final_points,
        'total_frames_processed': total_frames_processed,
        'processing_time_seconds': end_time - start_time,
        'bounding_box_filtering_enabled': not config.disable_bbox_filter,
        'consistency_checking_enabled': config.enable_consistency_check,
        'outlier_removal_enabled': config.enable_outlier_removal
    }
    if config.enable_outlier_removal:
        summary.update({
            'outlier_removal_neighbors': config.outlier_nb_neighbors,
            'outlier_removal_std_ratio': config.outlier_std_ratio,
            'outliers_removed': len(all_points) - final_points
        })
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print final summary
    print()
    print("=" * 60)
    print("DENSIFICATION COMPLETED")
    print("=" * 60)
    print(f"Dense point cloud: {config.output_path}")
    print(f"Final points: {final_points:,}")
    if config.enable_outlier_removal:
        print(f"Points before outlier removal: {len(all_points):,}")
        print(f"Outliers removed: {len(all_points) - final_points:,}")
    print(f"Frames processed: {total_frames_processed}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print()
    print("Output files saved to:")
    print(f"  - Dense point cloud: {os.path.relpath(config.output_path, config.scene_dir)}")
    print(f"  - Pairs file: {os.path.relpath(config.pairs_path, config.scene_dir)}")
    print(f"  - Configuration: {os.path.relpath(config_path, config.scene_dir)}")
    print(f"  - Summary: {os.path.relpath(summary_path, config.scene_dir)}")
    print("=" * 60)

    # Display the point cloud
    print("Opening point cloud visualization...")
    o3d.visualization.draw_geometries([pcd])


def densify_pairs_mast3r_batch(reconstruction: ColmapReconstruction, pairs: dict, config: DensificationConfig) -> dict[int, np.ndarray]:

    device = config.get_device()
    
    # load the model
    with profile_timer("Model Loading (Batch)", config.enable_profiling):
        model = AsymmetricMASt3R.from_pretrained(config.model_path, verbose=config.verbose).to(device)

    # Dictionary to store final point clouds per frame (similar to consistency check)
    frame_clouds = {}

    def densify_batch(pair_batch):
        image_batch = []
        image_batch_sizes = []
        img_id_to_idx = {}
        image_paths = []
        pair_info = []  # Store (img1, img2) tuples for tracking

        # get all image indices that appear in pair_batch
        image_indices = set()
        for img1, img2_list in pair_batch.items():
            if not reconstruction.has_image(int(img1)):
                continue
            # Handle both single partner and multiple partners format
            if isinstance(img2_list, list):
                partners = img2_list
            else:
                partners = [img2_list]
            
            # if len(partners) > 1:
            #     print(f"Processing image {img1} with {len(partners)} partners: {partners}")
            
            for img2 in partners:
                if not reconstruction.has_image(int(img2)):
                    continue
                image_indices.add(int(img1))
                image_indices.add(int(img2))
                pair_info.append((int(img1), int(img2)))

        # Skip this batch if no valid images found
        if not image_indices:
            print("Skipping batch - no valid images found in reconstruction...")
            return

        # get the image paths for the image indices
        for image_idx in image_indices:
            img_name = reconstruction.get_image_name(image_idx)
            img_path = os.path.join(config.img_dir, img_name)
            image_paths.append(img_path)
            img_id_to_idx[image_idx] = len(image_paths) - 1

        # load all images in the batch
        with profile_timer("Image Loading (Batch)", config.enable_profiling):
            print("loading images...")
            images = load_images(image_paths, size=config.size)

        # pair up the images using pair_info
        with profile_timer("Batch Preparation", config.enable_profiling):
            for img1, img2 in pair_info:
                img1_idx = img_id_to_idx[img1]
                img2_idx = img_id_to_idx[img2]
                image_batch.append(tuple([images[img1_idx], images[img2_idx]]))
                cam_1 = reconstruction.get_image_camera(img1)
                cam_2 = reconstruction.get_image_camera(img2)
                image_batch_sizes.append([tuple([cam_1.width, cam_1.height]), tuple([cam_2.width, cam_2.height])])

        # get predictions for each image pair
        with profile_timer("Model Inference (Batch)", config.enable_profiling):
            print(f"Processing {len(image_batch)} pairs in this batch...")
            output = inference(image_batch, model=model, device=device, batch_size=config.batch_size, verbose=config.verbose)

        # triangulate each image pair
        print("triangulating...")
        for i in range(len(image_batch)):
            img1, img2 = pair_info[i]

            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']

            desc1, desc2 = pred1['desc'][i].squeeze(0).detach(), pred2['desc'][i].squeeze(0).detach()

            # Compute feature matches
            with profile_timer("Feature Matching (Batch)", config.enable_profiling):
                matches_img1, matches_img2 = fast_reciprocal_NNs(
                    desc1, desc2, 
                    subsample_or_initxy1=config.sampling_factor, 
                    device=device, 
                    dist='dot', 
                    block_size=config.get_block_size(),
                    enable_profiling=config.enable_profiling
                )

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
            w1_scale = w1 / config.model_w
            h1_scale = h1 / config.model_h
            w2, h2 = image_batch_sizes[i][1]
            w2_scale = w2 / config.model_w
            h2_scale = h2 / config.model_h

            matches_img1[:, 0] = matches_img1[:, 0] * w1_scale
            matches_img1[:, 1] = matches_img1[:, 1] * h1_scale
            matches_img2[:, 0] = matches_img2[:, 0] * w2_scale
            matches_img2[:, 1] = matches_img2[:, 1] * h2_scale

            # get colors from image 1
            img1_path = os.path.join(config.img_dir, reconstruction.get_image_name(img1))
            image = Image.open(img1_path)
            img_width, img_height = image.size  # Get actual image dimensions
            
            # Vectorized color extraction
            img_array = np.array(image, dtype=np.float32) / 255.0
            x_coords = np.clip(matches_img1[:, 0].astype(int), 0, img_width - 1)
            y_coords = np.clip(matches_img1[:, 1].astype(int), 0, img_height - 1)
            img1_color = img_array[y_coords, x_coords]  # Note: numpy uses [y, x] indexing

            # Use the existing triangulation function
            with profile_timer("Triangulation (Batch)", config.enable_profiling):
                _, triangulated_points = compute_depth_from_pair(reconstruction, img1, img2, matches_img1, matches_img2)
            


            # filter out points that are too far from the cameras
            img1_cam = reconstruction.get_image_cam_from_world(img1)
            img1_R = img1_cam.rotation.matrix()
            img1_t = img1_cam.translation

            img2_cam = reconstruction.get_image_cam_from_world(img2)
            img2_R = img2_cam.rotation.matrix()
            img2_t = img2_cam.translation

            # Get camera centers using helper functions
            img1_C = reconstruction.get_camera_center(img1)
            img2_C = reconstruction.get_camera_center(img2)
            
            baseline = reconstruction.compute_baseline(img1, img2)

            # filter out points that are too far from the cameras (parallax stand-in)
            dist_to_cameras = np.linalg.norm(triangulated_points - img1_C, axis=1)
            triangulated_points = triangulated_points[dist_to_cameras < 20 * baseline, :]
            img1_color = img1_color[dist_to_cameras < 20 * baseline]

            # filter out points that are behind the camera
            behind_camera = (img1_R @ triangulated_points.T - img1_t.reshape(-1, 1))[2] < 0
            triangulated_points = triangulated_points[~behind_camera, :]
            img1_color = img1_color[~behind_camera]

            # TODO: filter out points that violate epipolar constraint

            # Accumulate points for this frame (img1)
            if img1 not in frame_clouds:
                frame_clouds[img1] = {
                    'points': triangulated_points,
                    'colors': img1_color
                }
                # print(f"Created new point cloud for image {img1} with {len(triangulated_points)} points")
            else:
                # Concatenate with existing points for this frame
                prev_count = len(frame_clouds[img1]['points'])
                frame_clouds[img1]['points'] = np.concatenate([
                    frame_clouds[img1]['points'], 
                    triangulated_points
                ], axis=0)
                frame_clouds[img1]['colors'] = np.concatenate([
                    frame_clouds[img1]['colors'], 
                    img1_color
                ], axis=0)
                new_count = len(frame_clouds[img1]['points'])
                # print(f"Accumulated points for image {img1}: {prev_count} + {len(triangulated_points)} = {new_count} total")

    # Create batches more efficiently by handling all pairs, including remainder
    pair_keys = list(pairs.keys())
    batches = []
    
    for i in range(0, len(pair_keys), config.batch_size):
        batch_keys = pair_keys[i:i + config.batch_size]
        batch = {k: pairs[k] for k in batch_keys}
        batches.append(batch)

    for batch in batches:
        densify_batch(batch)
    
    return frame_clouds




if __name__ == "__main__":
    main()