"""
COLMAP Utilities for MASt3R Densification
========================================

This module contains all COLMAP-related functionality including:
- Bounding box computation from sparse reconstruction
- Image pair selection algorithms
- Camera parameter extraction
- 3D point analysis

"""

import numpy as np
import random
import pycolmap


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


def get_camera_calibration_matrix(image, reconstruction):
    """Get camera calibration matrix for an image."""
    return reconstruction.images[image.image_id].camera.calibration_matrix()


def get_camera_projection_matrix(image_id, reconstruction):
    """Get camera projection matrix (K @ [R|t])."""
    image = reconstruction.images[image_id]
    K = image.camera.calibration_matrix()
    return K @ image.cam_from_world().matrix()


def get_camera_distortion_params(image_id, reconstruction):
    """Extract camera distortion parameters as a dictionary and array."""
    image = reconstruction.images[image_id]
    camera = image.camera
    
    # Parse parameter info
    dist_keys = camera.params_info.split(', ')
    dist_dict = {}
    for i, key in enumerate(dist_keys):
        dist_dict[key] = camera.params[i]
    
    # Create standard distortion coefficient array
    dist_coeffs = np.array([
        dist_dict.get('k1', 0), dist_dict.get('k2', 0), 
        dist_dict.get('p1', 0), dist_dict.get('p2', 0), 
        dist_dict.get('k3', 0), dist_dict.get('k4', 0), 
        dist_dict.get('k5', 0), dist_dict.get('k6', 0)
    ])
    
    return dist_dict, dist_coeffs


def get_camera_center(image_id, reconstruction):
    """Get camera center in world coordinates."""
    image = reconstruction.images[image_id]
    cam = image.cam_from_world()
    R = cam.rotation.matrix()
    t = cam.translation
    return -R.T @ t


def compute_baseline(img1_id, img2_id, reconstruction):
    """Compute baseline distance between two cameras."""
    C1 = get_camera_center(img1_id, reconstruction)
    C2 = get_camera_center(img2_id, reconstruction)
    return np.linalg.norm(C1 - C2)


def compute_parallax_angle(point_3d, img1_id, img2_id, reconstruction):
    """Compute parallax angle for a 3D point viewed from two cameras."""
    C1 = get_camera_center(img1_id, reconstruction)
    C2 = get_camera_center(img2_id, reconstruction)
    
    u = C1 - point_3d
    v = C2 - point_3d
    
    # Compute angle between viewing rays
    cos_angle = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
    return np.arccos(cos_angle)


def build_image_point_maps(reconstruction):
    """
    Build dictionaries mapping image IDs to their 3D points and 2D coordinates.
    
    Returns:
        image_point3D_ids: dict[image_id] -> set of point3D_ids
        image_point3D_xy: dict[image_id] -> dict[point3D_id] -> 2D coordinates
    """
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
    
    return image_point3D_ids, image_point3D_xy


class MatchCandidate:
    """Helper class for storing image matching candidates with their metrics."""
    def __init__(self, image_id, shared_points):
        self.image_id = image_id
        self.shared_points = shared_points
        self.avg_parallax_angle = 0
        self.x_coverage = 0
        self.y_coverage = 0


def compute_feature_coverage(shared_points, image_id, image_point3D_xy, reconstruction):
    """Compute spatial coverage of features in an image."""
    image = reconstruction.images[image_id]
    xs = [image_point3D_xy[image_id][point_id][0] for point_id in shared_points]
    ys = [image_point3D_xy[image_id][point_id][1] for point_id in shared_points]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_coverage = x_range / image.camera.width
    y_coverage = y_range / image.camera.height
    
    return x_coverage, y_coverage


def compute_average_parallax(shared_points, img1_id, img2_id, reconstruction, sample_size=100):
    """Compute average parallax angle between two images for shared 3D points."""
    parallax_angles = []
    sample_size = min(sample_size, len(shared_points))
    
    for point_id in random.sample(list(shared_points), sample_size):
        point = reconstruction.points3D[point_id]
        parallax_angle = compute_parallax_angle(point.xyz, img1_id, img2_id, reconstruction)
        parallax_angles.append(parallax_angle)
    
    return np.mean(parallax_angles)


def get_multiple_pairs_per_image(reconstruction, max_pairs_per_image=7, min_points=100, 
                                parallax_sample_size=100, min_feature_coverage=0.6):
    """
    Find multiple pairs for each image instead of just one best pair.
    Returns a dictionary where each key is an image_id and value is a list of partner image_ids.
    """
    # Initialize empty pair map (image ids to list of partner ids)
    pairs = {}
    for image in reconstruction.images.values():
        pairs[image.image_id] = []
    
    # Build image-to-points mappings
    image_point3D_ids, image_point3D_xy = build_image_point_maps(reconstruction)
    
    # Process each image
    for i, image in enumerate(reconstruction.images.values()):
        print(f"Getting multiple corresponding images for {image.name}... {i}/{len(reconstruction.images) - 1}")
        
        # Find other images that share at least min_points points
        other_images = [other_image for other_image in reconstruction.images.values() 
                       if other_image.image_id != image.image_id]
        match_candidates = []
        
        for other_image in other_images:
            # Get the points that the two images share
            shared_points = image_point3D_ids[image.image_id] & image_point3D_ids[other_image.image_id]
            if len(shared_points) >= min_points:
                match_candidates.append(MatchCandidate(other_image.image_id, shared_points))
        
        # Skip if no good candidates found
        if len(match_candidates) == 0:
            continue
        
        # Compute feature coverage for each candidate
        for match_candidate in match_candidates:
            x_cov, y_cov = compute_feature_coverage(
                match_candidate.shared_points, image.image_id, image_point3D_xy, reconstruction)
            match_candidate.x_coverage = x_cov
            match_candidate.y_coverage = y_cov
        
        # Compute average parallax angle for each candidate
        for match_candidate in match_candidates:
            match_candidate.avg_parallax_angle = compute_average_parallax(
                match_candidate.shared_points, image.image_id, match_candidate.image_id, 
                reconstruction, parallax_sample_size)
        
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
    
    # Initialize empty pair map (image ids)
    pairs = {}
    for image in reconstruction.images.values():
        pairs[image.image_id] = -1
    
    # Build image-to-points mappings
    image_point3D_ids, image_point3D_xy = build_image_point_maps(reconstruction)
    
    # Process each image
    for i, image in enumerate(reconstruction.images.values()):
        print(f"Getting best corresponding image for {image.name}... {i}/{len(reconstruction.images) - 1}")
        
        # Find other images that share at least min_points points
        other_images = [other_image for other_image in reconstruction.images.values() 
                       if other_image.image_id != image.image_id]
        match_candidates = []
        
        for other_image in other_images:
            # Get the points that the two images share
            shared_points = image_point3D_ids[image.image_id] & image_point3D_ids[other_image.image_id]
            match_candidates.append(MatchCandidate(other_image.image_id, shared_points))
        
        # Sort by number of shared points
        match_candidates.sort(key=lambda x: len(x.shared_points), reverse=True)
        
        # Skip if no good candidates found
        if len(match_candidates) == 0 or len(match_candidates[0].shared_points) < min_points:
            continue
        
        # Filter out match candidates that don't have enough matches
        match_candidates = [candidate for candidate in match_candidates 
                          if len(candidate.shared_points) >= min_points]

        # Compute feature coverage for each candidate
        for match_candidate in match_candidates:
            x_cov, y_cov = compute_feature_coverage(
                match_candidate.shared_points, image.image_id, image_point3D_xy, reconstruction)
            match_candidate.x_coverage = x_cov
            match_candidate.y_coverage = y_cov

        # Order match candidates by average xy coverage
        match_candidates.sort(key=lambda x: (x.x_coverage + x.y_coverage) / 2, reverse=True)

        # Compute average parallax angle for each candidate
        for match_candidate in match_candidates:
            match_candidate.avg_parallax_angle = compute_average_parallax(
                match_candidate.shared_points, image.image_id, match_candidate.image_id, 
                reconstruction, parallax_sample_size)

        if len(match_candidates) == 0:
            continue
        
        # Make sure the high overlap match has good parallax
        winning_index = -1
        for i in range(len(match_candidates)):
            if match_candidates[i].avg_parallax_angle > 0.1:
                winning_index = i
                break
        if winning_index == -1:
            winning_index = 0
        
        pairs[image.image_id] = match_candidates[winning_index].image_id

    return pairs


def load_reconstruction(reconstruction_path):
    """Load COLMAP reconstruction with error handling."""
    try:
        reconstruction = pycolmap.Reconstruction(reconstruction_path)
        print(f"Loaded reconstruction with {len(reconstruction.images)} images and {len(reconstruction.points3D)} 3D points")
        return reconstruction
    except Exception as e:
        raise ValueError(f"Failed to load COLMAP reconstruction from {reconstruction_path}: {e}")


def validate_image_in_reconstruction(image_id, reconstruction):
    """Check if an image ID exists in the reconstruction."""
    return image_id in reconstruction.images


def get_reconstruction_summary(reconstruction):
    """Get summary statistics about the reconstruction."""
    total_observations = sum(len(point3D.track.elements) for point3D in reconstruction.points3D.values())
    avg_track_length = total_observations / len(reconstruction.points3D) if reconstruction.points3D else 0
    
    return {
        'num_images': len(reconstruction.images),
        'num_points_3d': len(reconstruction.points3D),
        'num_cameras': len(reconstruction.cameras),
        'total_observations': total_observations,
        'avg_track_length': avg_track_length
    } 