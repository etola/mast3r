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
    parser.add_argument('-p', '--pairs_per_image', type=int, required=False, help='Maximum number of pairs to compute per image. Default = 4', default=4)
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
                pairs_per_image=config.pairs_per_image
            )
        with profile_timer("Pairs Saving", config.enable_profiling):
            with open(config.pairs_path, 'w') as f:
                print(f"Saving pairs to {config.pairs_path}...")
                json.dump(pairs, f, indent=4)
    else:
        with profile_timer("Pairs Loading", config.enable_profiling):
            pairs = json.load(open(config.pairs_path))
            print(f"Loaded pairs from {config.pairs_path}...")

    with profile_timer("Main Densification (Batch Processing)", config.enable_profiling):
        frame_info = densify_pairs_mast3r_batch(reconstruction, pairs, config)

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    # Print cache statistics
    print_cache_stats()
    
    # Print profiling summary
    print_profiling_summary(config.enable_profiling)
    
    # Save profiling data to JSON file
    profiling_path = os.path.join(config.scene_dir, config.output_dir, 'profiling_data.json')
    save_profiling_data(profiling_path, config.enable_profiling)

    # Load and combine all saved point clouds
    print("Loading and merging saved point clouds...")
    all_points = None
    all_colors = None
    total_frames_processed = 0
    saved_point_clouds = []
    
    with profile_timer("Point Cloud Loading and Merging", config.enable_profiling):
        for frame_id, frame_data in frame_info.items():
            if frame_data['saved_path'] and os.path.exists(frame_data['saved_path']):
                total_frames_processed += 1
                
                # Load the saved point cloud
                pcd = o3d.io.read_point_cloud(frame_data['saved_path'])
                frame_points = np.asarray(pcd.points)
                frame_colors = np.asarray(pcd.colors)
                
                saved_point_clouds.append(frame_data['saved_path'])  # Track for cleanup
                
                if all_points is None:
                    all_points = frame_points
                    all_colors = frame_colors
                else:
                    all_points = np.concatenate([all_points, frame_points], axis=0)
                    all_colors = np.concatenate([all_colors, frame_colors], axis=0)

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
    total_points_before_filtering = len(all_points) if all_points is not None else 0
    summary = {
        'total_points_before_filtering': total_points_before_filtering,
        'final_points_after_filtering': final_points,
        'total_frames_processed': total_frames_processed,
        'processing_time_seconds': end_time - start_time,
        'bounding_box_filtering_enabled': not config.disable_bbox_filter,
        'outlier_removal_enabled': config.enable_outlier_removal
    }
    if config.enable_outlier_removal:
        summary.update({
            'outlier_removal_neighbors': config.outlier_nb_neighbors,
            'outlier_removal_std_ratio': config.outlier_std_ratio,
            'outliers_removed': total_points_before_filtering - final_points
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
        print(f"Points before outlier removal: {total_points_before_filtering:,}")
        print(f"Outliers removed: {total_points_before_filtering - final_points:,}")
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


def densify_for_image(reconstruction: ColmapReconstruction, key_frame_id: int, partner_frame_ids: list, config: DensificationConfig, model, device, save_to_disk: bool = False, output_dir: str = None) -> dict:
    """
    Densify a single key frame using multi-view triangulation from collated tracks.
    
    Args:
        reconstruction: COLMAP reconstruction
        key_frame_id: ID of the key frame
        partner_frame_ids: List of partner frame IDs
        config: Densification configuration
        model: MASt3R model
        device: Computing device
    
    Returns:
        Dictionary with 'points' and 'colors' for the key frame
    """
    from collections import defaultdict
    
    if not reconstruction.has_image(key_frame_id):
        return {'points': np.array([]), 'colors': np.array([])}
    
    # Filter valid partner frames
    valid_partners = [pid for pid in partner_frame_ids if reconstruction.has_image(pid)]
    if not valid_partners:
        return {'points': np.array([]), 'colors': np.array([])}
    
    print(f"Processing key frame {key_frame_id} with {len(valid_partners)} partners...")
    
    # Load key frame
    key_img_name = reconstruction.get_image_name(key_frame_id)
    key_img_path = os.path.join(config.img_dir, key_img_name)
    
    # Load all images for this key frame
    image_paths = [key_img_path]
    for partner_id in valid_partners:
        partner_img_name = reconstruction.get_image_name(partner_id)
        partner_img_path = os.path.join(config.img_dir, partner_img_name)
        image_paths.append(partner_img_path)
    
    with profile_timer("Image Loading (Single Key Frame)", config.enable_profiling):
        images = load_images(image_paths, size=config.size)
    
    # Prepare image pairs for inference - use the same approach as densify_batch
    image_batch = []
    image_batch_sizes = []
    
    key_cam = reconstruction.get_image_camera(key_frame_id)
    key_size = (key_cam.width, key_cam.height)
    
    # Create image pairs using the loaded image dictionaries directly
    key_image_dict = images[0]  # Use the full dictionary from load_images
    
    for i, partner_id in enumerate(valid_partners):
        partner_cam = reconstruction.get_image_camera(partner_id)
        partner_size = (partner_cam.width, partner_cam.height)
        
        partner_image_dict = images[i + 1]  # Use the full dictionary from load_images
        
        # Follow the same pattern as densify_batch
        image_batch.append(tuple([key_image_dict, partner_image_dict]))
        image_batch_sizes.append([key_size, partner_size])
    
    # Run inference
    with profile_timer("Model Inference (Single Key Frame)", config.enable_profiling):
        output = inference(image_batch, model=model, device=device, batch_size=config.batch_size, verbose=config.verbose)
    
    # Collect matches for each key frame pixel
    pixel_tracks = defaultdict(list)  # pixel_coord -> [(partner_id, partner_coord), ...]
    
    for i, partner_id in enumerate(valid_partners):
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        
        desc1 = pred1['desc'][i].squeeze(0).detach()
        desc2 = pred2['desc'][i].squeeze(0).detach()
        
        # Compute feature matches
        with profile_timer("Feature Matching (Single Key Frame)", config.enable_profiling):
            matches_key, matches_partner = fast_reciprocal_NNs(
                desc1, desc2,
                subsample_or_initxy1=config.sampling_factor,
                device=device,
                dist='dot',
                block_size=config.get_block_size(),
                enable_profiling=config.enable_profiling
            )
        
        # Validate matches within image boundaries
        H1, W1 = view1['true_shape'][i]
        valid_matches_key = (matches_key[:, 0] >= 3) & (matches_key[:, 1] >= 3) & \
                           (matches_key[:, 0] < int(W1) - 3) & (matches_key[:, 1] < int(H1) - 3)
        
        H2, W2 = view2['true_shape'][i]
        valid_matches_partner = (matches_partner[:, 0] >= 3) & (matches_partner[:, 1] >= 3) & \
                               (matches_partner[:, 0] < int(W2) - 3) & (matches_partner[:, 1] < int(H2) - 3)
        
        valid_matches = valid_matches_key & valid_matches_partner
        matches_key = matches_key[valid_matches]
        matches_partner = matches_partner[valid_matches]
        
        if len(matches_key) == 0:
            continue
        
        # Scale matches to original image size immediately (same as densify_batch)
        key_w_scale = key_size[0] / config.model_w
        key_h_scale = key_size[1] / config.model_h
        partner_w_scale = image_batch_sizes[i][1][0] / config.model_w
        partner_h_scale = image_batch_sizes[i][1][1] / config.model_h
        
        # Scale all matches to original coordinates (same as densify_batch)
        matches_key_scaled = matches_key.astype(np.float64)  # Convert to float to allow scaling
        matches_key_scaled[:, 0] *= key_w_scale
        matches_key_scaled[:, 1] *= key_h_scale
        
        matches_partner_scaled = matches_partner.astype(np.float64)  # Convert to float to allow scaling
        matches_partner_scaled[:, 0] *= partner_w_scale
        matches_partner_scaled[:, 1] *= partner_h_scale
        
        # Group matches by scaled key frame pixel coordinates
        for j in range(len(matches_key_scaled)):
            # Use model coordinates (unscaled) as dictionary key for grouping
            key_pixel = tuple(matches_key[j].astype(int))  
            # Store scaled coordinates for triangulation
            key_coord_scaled = matches_key_scaled[j]
            partner_coord_scaled = matches_partner_scaled[j]
            pixel_tracks[key_pixel].append((partner_id, key_coord_scaled, partner_coord_scaled))
    
    # Triangulate tracks with multiple observations
    triangulated_points = []
    pixel_colors = []
    
    # Use cached unnormalized image for color sampling (best of both worlds!)
    # This avoids file I/O while preserving proper brightness
    key_img_array = key_image_dict['img_unnormalized']  # Already normalized [0,1] without ImgNorm
    img_height, img_width = key_img_array.shape[:2]  # Note: numpy shape is (H, W, C)
    
    with profile_timer("Multi-view Triangulation (Single Key Frame)", config.enable_profiling):
        for pixel_coord, observations in pixel_tracks.items():
            if len(observations) < 1:  # Need at least one partner observation
                continue
            
            x_key_model, y_key_model = pixel_coord  # These are in model coordinates
            
            # We'll check bounds later using scaled coordinates since we're using original image
            
            # Prepare data for multi-view triangulation using already-scaled coordinates
            camera_matrices = []
            image_points = []
            
            # Get the first observation to extract key frame scaled coordinates
            first_observation = observations[0]
            _, key_coord_scaled, _ = first_observation
            
            # Add key frame observation (using scaled coordinates)
            P_key = reconstruction.get_camera_projection_matrix(key_frame_id)
            camera_matrices.append(P_key)
            image_points.append([key_coord_scaled[0], key_coord_scaled[1]])
            
            # Add partner observations (using stored scaled coordinates)
            for partner_id, key_coord_scaled, partner_coord_scaled in observations:
                P_partner = reconstruction.get_camera_projection_matrix(partner_id)
                camera_matrices.append(P_partner)
                image_points.append([partner_coord_scaled[0], partner_coord_scaled[1]])
            
            # Convert to numpy arrays
            camera_matrices = np.array(camera_matrices)
            image_points = np.array(image_points).T  # Shape: (2, n_views)
            
            if len(camera_matrices) >= 2:  # Need at least 2 views for triangulation
                # Use OpenCV's triangulation for multi-view case
                if len(camera_matrices) == 2:
                    # Standard two-view triangulation
                    triangulated = cv2.triangulatePoints(
                        camera_matrices[0], camera_matrices[1],
                        image_points[:, [0]], image_points[:, [1]]
                    )
                else:
                    # For more than 2 views, triangulate using first two and validate with others
                    triangulated = cv2.triangulatePoints(
                        camera_matrices[0], camera_matrices[1],
                        image_points[:, [0]], image_points[:, [1]]
                    )
                
                # Convert from homogeneous coordinates
                triangulated = triangulated / triangulated[3, :]
                point_3d = triangulated[:3, 0]
                
                # Basic validation: check if point is in front of cameras
                valid_point = True
                for cam_idx in range(len(camera_matrices)):
                    cam_id = key_frame_id if cam_idx == 0 else observations[cam_idx-1][0]
                    img_cam = reconstruction.get_image_cam_from_world(cam_id)
                    points_cam = (img_cam.rotation.matrix() @ point_3d + img_cam.translation)
                    if points_cam[2] <= 0:  # Behind camera
                        valid_point = False
                        break
                
                if valid_point:
                    # Sample color from cached image using model coordinates
                    # The cached img_unnormalized is in model dimensions, so use model coords
                    x_clipped = int(np.clip(x_key_model, 0, img_width - 1))
                    y_clipped = int(np.clip(y_key_model, 0, img_height - 1))
                    color = key_img_array[y_clipped, x_clipped]
                    
                    triangulated_points.append(point_3d)
                    pixel_colors.append(color)
    
    # Convert to numpy arrays
    if len(triangulated_points) > 0:
        triangulated_points = np.array(triangulated_points)
        pixel_colors = np.array(pixel_colors)
        
        # Apply distance filtering
        key_cam_center = reconstruction.get_camera_center(key_frame_id)
        distances = np.linalg.norm(triangulated_points - key_cam_center, axis=1)
        
        # Compute typical baseline for filtering
        if valid_partners:
            typical_baseline = np.mean([
                reconstruction.compute_baseline(key_frame_id, pid) 
                for pid in valid_partners[:3]  # Use first few partners
            ])
            
            # Filter points that are too far
            valid_distance = distances < 20 * typical_baseline
            triangulated_points = triangulated_points[valid_distance]
            pixel_colors = pixel_colors[valid_distance]
        
        print(f"Generated {len(triangulated_points)} points for key frame {key_frame_id}")
        
        if save_to_disk and output_dir:
            # Save point cloud to disk immediately
            frame_filename = f"frame_{key_frame_id:06d}.ply"
            frame_path = os.path.join(output_dir, frame_filename)
            
            with profile_timer("Frame Point Cloud Saving", config.enable_profiling):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(triangulated_points)
                pcd.colors = o3d.utility.Vector3dVector(pixel_colors)
                o3d.io.write_point_cloud(frame_path, pcd)
                print(f"Saved frame {key_frame_id} point cloud to {frame_filename}")
            
            return {
                'saved_path': frame_path,
                'point_count': len(triangulated_points)
            }
        else:
            return {
                'points': triangulated_points,
                'colors': pixel_colors
            }
    else:
        triangulated_points = np.array([]).reshape(0, 3)
        pixel_colors = np.array([]).reshape(0, 3)
        print(f"No valid points generated for key frame {key_frame_id}")
        
        if save_to_disk and output_dir:
            return {
                'saved_path': None,
                'point_count': 0
            }
        else:
            return {
                'points': triangulated_points,
                'colors': pixel_colors
            }


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

    # New densification strategy: use multi-view triangulation per key frame
    # Create temporary directory for frame point clouds
    frame_clouds_dir = os.path.join(config.scene_dir, config.output_dir, 'temp_frame_clouds')
    os.makedirs(frame_clouds_dir, exist_ok=True)
    
    print(f"Processing {len(pairs)} frames with multi-view triangulation strategy...")
    frame_info = {}  # Store frame information instead of actual point clouds
    
    for key_frame_id, partner_frame_ids in pairs.items():
        # Convert to appropriate format
        key_frame_id = int(key_frame_id)
        if isinstance(partner_frame_ids, list):
            partners = [int(pid) for pid in partner_frame_ids]
        else:
            partners = [int(partner_frame_ids)]
        
        # Use the new densification function with disk saving
        frame_result = densify_for_image(
            reconstruction, key_frame_id, partners, config, model, device,
            save_to_disk=True, output_dir=frame_clouds_dir
        )
        
        if frame_result['point_count'] > 0:
            frame_info[key_frame_id] = frame_result
    
    return frame_info




if __name__ == "__main__":
    main()