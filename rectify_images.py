"""
Image Rectification for Stereo Processing
=========================================

This module provides functionality for stereo rectification of image pairs using 
COLMAP reconstruction data. It includes functions for computing rectification 
parameters, applying rectification to images, and managing coordinate transformations.

Author: Blake Troutman 2025
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import json

# COLMAP utilities
import colmap_utils
from colmap_utils import ColmapReconstruction


def update_intrinsics_for_rotation(K: np.ndarray, orig_width: int, orig_height: int, rotation_angle: float) -> np.ndarray:
    """
    Update camera intrinsics matrix for image rotation.
    
    Args:
        K: Original intrinsics matrix (3x3)
        orig_width, orig_height: Original image dimensions
        rotation_angle: Rotation angle in degrees (0, 90, 180, 270)
    
    Returns:
        Updated intrinsics matrix
    """
    K_new = K.copy()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    if rotation_angle == 90:
        # 90° clockwise: (x, y) → (y, width-1-x)
        # New principal point: (cx, cy) → (cy, width-1-cx)
        # Focal lengths may swap depending on sensor orientation
        K_new[0, 0] = fy  # fx_new = fy_old
        K_new[1, 1] = fx  # fy_new = fx_old  
        K_new[0, 2] = cy  # cx_new = cy_old
        K_new[1, 2] = orig_width - 1 - cx  # cy_new = width-1-cx_old
        
    elif rotation_angle == -90 or rotation_angle == 270:
        # -90° (270°): (x, y) → (height-1-y, x)
        # New principal point: (cx, cy) → (height-1-cy, cx)
        K_new[0, 0] = fy  # fx_new = fy_old
        K_new[1, 1] = fx  # fy_new = fx_old
        K_new[0, 2] = orig_height - 1 - cy  # cx_new = height-1-cy_old
        K_new[1, 2] = cx  # cy_new = cx_old
        
    elif rotation_angle == 180 or rotation_angle == -180:
        # 180°: (x, y) → (width-1-x, height-1-y)
        # Focal lengths stay the same, only principal point changes
        K_new[0, 2] = orig_width - 1 - cx   # cx_new = width-1-cx_old
        K_new[1, 2] = orig_height - 1 - cy  # cy_new = height-1-cy_old
    
    # For 0° or other angles, return original matrix
    return K_new

def detect_epipolar_orientation(reconstruction: ColmapReconstruction, img1_id: int, img2_id: int):
    """
    Detect the orientation of epipolar lines and determine required rotation
    to make them horizontal for optimal rectified stereo matching.
    
    Args:
        reconstruction: COLMAP reconstruction
        img1_id: First image ID
        img2_id: Second image ID
        
    Returns:
        rotation_angle: Angle in degrees (0, 90, 180, 270) to rotate images before rectification
    """
    try:
        # Get camera parameters using the reconstruction API
        camera1 = reconstruction.get_image_camera(img1_id)
        camera2 = reconstruction.get_image_camera(img2_id)
        
        # Build camera matrices
        K1 = np.array([[camera1.params[0], 0, camera1.params[2]],
                       [0, camera1.params[1], camera1.params[3]],
                       [0, 0, 1]])
        
        K2 = np.array([[camera2.params[0], 0, camera2.params[2]],
                       [0, camera2.params[1], camera2.params[3]],
                       [0, 0, 1]])
        
        # Get camera poses (world to camera)
        img1_cam = reconstruction.get_image_cam_from_world(img1_id)
        img2_cam = reconstruction.get_image_cam_from_world(img2_id)
        
        R1 = img1_cam.rotation.matrix()
        t1 = img1_cam.translation.reshape(3, 1)
        R2 = img2_cam.rotation.matrix()
        t2 = img2_cam.translation.reshape(3, 1)
        
        # Compute relative pose (camera2 relative to camera1)
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        
        # Compute essential matrix
        t_skew = np.array([[0, -t_rel[2, 0], t_rel[1, 0]],
                          [t_rel[2, 0], 0, -t_rel[0, 0]],
                          [-t_rel[1, 0], t_rel[0, 0], 0]])
        E = t_skew @ R_rel
        
        # Compute fundamental matrix
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        
        # Sample points in the first image to compute epipolar lines
        h1, w1 = camera1.height, camera1.width
        sample_points = np.array([
            [w1/4, h1/4, 1],     # Top-left quadrant
            [3*w1/4, h1/4, 1],   # Top-right quadrant  
            [w1/4, 3*h1/4, 1],   # Bottom-left quadrant
            [3*w1/4, 3*h1/4, 1], # Bottom-right quadrant
            [w1/2, h1/2, 1]      # Center
        ]).T
        
        # Compute epipolar lines in second image
        epipolar_lines = F @ sample_points  # Each column is [a, b, c] for line ax + by + c = 0
        
        # Calculate angles of epipolar lines
        angles = []
        for i in range(epipolar_lines.shape[1]):
            a, b, c = epipolar_lines[0, i], epipolar_lines[1, i], epipolar_lines[2, i]
            if abs(b) > 1e-6:  # Avoid division by zero
                # Angle of line ax + by + c = 0 is arctan(-a/b)
                angle = np.arctan2(-a, b) * 180 / np.pi
                angles.append(angle)
        
        if not angles:
            return 0  # Default to no rotation if we can't compute angles
        
        # Find the dominant orientation
        avg_angle = np.mean(angles)
        
        # Normalize to [0, 180) since epipolar lines are undirected
        avg_angle = avg_angle % 180
        
        # Determine required rotation to make lines horizontal (0° or 180°)
        if abs(avg_angle) < 22.5 or abs(avg_angle - 180) < 22.5:
            # Already horizontal
            rotation_angle = 0
        elif abs(avg_angle - 90) < 22.5:
            # Vertical lines, rotate 90° to make horizontal
            rotation_angle = 90
        elif abs(avg_angle - 45) < 22.5:
            # Diagonal (~45°), rotate 45° to make horizontal
            rotation_angle = 45
        elif abs(avg_angle - 135) < 22.5:
            # Diagonal (~135°), rotate -45° to make horizontal  
            rotation_angle = -45
        else:
            # Default case, try to get closest to horizontal
            if avg_angle > 90:
                rotation_angle = 180 - avg_angle
            else:
                rotation_angle = -avg_angle
        
        print(f"Epipolar lines angle: {avg_angle:.1f}°, applying rotation: {rotation_angle}°")
        return rotation_angle
        
    except Exception as e:
        print(f"Warning: Could not detect epipolar orientation: {e}")
        return 0  # Default to no rotation

def rectify_stereo_pair(reconstruction: ColmapReconstruction, img1_id: int, img2_id: int, 
                       alpha: float = 0.5, enable_rotation: bool = True):
    """
    Compute stereo rectification for a pair of images.
    
    Args:
        reconstruction: COLMAP reconstruction
        img1_id, img2_id: Image IDs
        alpha: Rectification cropping parameter (0=crop to valid region, 1=keep all pixels)
        enable_rotation: Whether to apply rotation for horizontal epipolar lines
    
    Returns:
        Dictionary containing rectification matrices and parameters, or None if rectification fails
    """
    try:
        # First, detect required rotation for horizontal epipolar lines (if enabled)
        if enable_rotation:
            rotation_angle = detect_epipolar_orientation(reconstruction, img1_id, img2_id)
        else:
            rotation_angle = 0
        
        # Get camera parameters
        cam1 = reconstruction.get_image_camera(img1_id)
        cam2 = reconstruction.get_image_camera(img2_id)
        
        # Get camera poses
        img1_cam = reconstruction.get_image_cam_from_world(img1_id)
        img2_cam = reconstruction.get_image_cam_from_world(img2_id)
        
        # Convert to OpenCV format
        # Camera matrices
        K1 = np.array([[cam1.params[0], 0, cam1.params[2]],
                       [0, cam1.params[1], cam1.params[3]],
                       [0, 0, 1]], dtype=np.float64)
        
        K2 = np.array([[cam2.params[0], 0, cam2.params[2]],
                       [0, cam2.params[1], cam2.params[3]],
                       [0, 0, 1]], dtype=np.float64)
        
        # Distortion coefficients (assume no distortion for simplicity)
        D1 = np.zeros(4, dtype=np.float64)
        D2 = np.zeros(4, dtype=np.float64)
        
        # Get original image dimensions
        orig_w, orig_h = cam1.width, cam1.height
        
        # Get original rotations and translations
        R1_orig = img1_cam.rotation.matrix()
        t1_orig = img1_cam.translation
        R2_orig = img2_cam.rotation.matrix()
        t2_orig = img2_cam.translation
        
        # Apply rotation to camera poses if needed (before computing rectification)
        if enable_rotation and rotation_angle != 0:
            print(f"Applying {rotation_angle}° rotation to camera poses and intrinsics before rectification")
            
            # Create rotation matrix for the detected angle
            # Note: This rotates the camera coordinate system
            angle_rad = np.radians(rotation_angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Rotation around Z-axis (standard image rotation)
            R_img_rot = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ], dtype=np.float64)
            
            # Apply rotation to camera orientations
            # R_new = R_img_rot @ R_orig (rotate the camera reference frame)
            R1 = R_img_rot @ R1_orig
            R2 = R_img_rot @ R2_orig
            t1 = t1_orig  # Translation doesn't change with pure rotation
            t2 = t2_orig
            
            # Update intrinsics matrices for the rotated images
            print(f"  - Original intrinsics - cx: {K1[0,2]:.1f}, cy: {K1[1,2]:.1f}, fx: {K1[0,0]:.1f}, fy: {K1[1,1]:.1f}")
            K1 = update_intrinsics_for_rotation(K1, orig_w, orig_h, rotation_angle)
            K2 = update_intrinsics_for_rotation(K2, orig_w, orig_h, rotation_angle)
            print(f"  - Updated intrinsics - cx: {K1[0,2]:.1f}, cy: {K1[1,2]:.1f}, fx: {K1[0,0]:.1f}, fy: {K1[1,1]:.1f}")
            
            # Update image dimensions if needed (90° or 270° rotations swap w/h)
            if abs(rotation_angle) == 90 or abs(rotation_angle) == 270:
                image_width, image_height = orig_h, orig_w  # Swap dimensions
                print(f"  - Image dimensions after rotation: {image_width}x{image_height}")
            else:
                image_width, image_height = orig_w, orig_h
        else:
            # Use original poses and intrinsics for rectification computation
            R1 = R1_orig
            R2 = R2_orig
            t1 = t1_orig
            t2 = t2_orig
            # K1, K2 already set above
            image_width, image_height = orig_w, orig_h
        
        # Relative pose from cam1 to cam2 (using potentially rotated poses)
        R = R2 @ R1.T
        t = t2 - R @ t1
        
        # Check if baseline is too small for stable rectification
        baseline = np.linalg.norm(t)
        if baseline < 1e-6:
            print(f"Warning: Very small baseline ({baseline:.2e}) for images {img1_id}, {img2_id}")
            return None
        
        # Image sizes (use updated dimensions that account for rotation)
        image_size = (image_width, image_height)
        print(f"  - Rectification computed for image size: {image_size}")
        
        # Compute rectification
        R1_rect, R2_rect, P1_rect, P2_rect, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R, t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=alpha  # Control amount of cropping: 0=crop to valid region, 1=keep all pixels
        )
        
        # Debug: Print valid ROI information
        print(f"  - Valid ROI 1: {valid_roi1} (x, y, w, h)")
        print(f"  - Valid ROI 2: {valid_roi2} (x, y, w, h)")
        print(f"  - Image size used for rectification: {image_size}")
        
        if alpha == 0.0:
            print(f"  - Alpha=0: Cropping to valid regions only")
            # Check if valid regions are reasonable
            roi1_coverage = (valid_roi1[2] * valid_roi1[3]) / (image_size[0] * image_size[1])
            roi2_coverage = (valid_roi2[2] * valid_roi2[3]) / (image_size[0] * image_size[1])
            print(f"  - ROI coverage: {roi1_coverage:.2%} (img1), {roi2_coverage:.2%} (img2)")
            
            if roi1_coverage < 0.5 or roi2_coverage < 0.5:
                print(f"  - WARNING: Valid ROI coverage is low, this may cause excessive cropping")
        elif alpha == 1.0:
            print(f"  - Alpha=1: Keeping all pixels (may have black borders)")
        
        # Compute rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1_rect, P1_rect, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2_rect, P2_rect, image_size, cv2.CV_32FC1)
        
        # Validate that maps were created successfully
        if map1x is None or map1y is None or map2x is None or map2y is None:
            print(f"Error: Failed to create rectification maps for images {img1_id}, {img2_id}")
            return None
        
        return {
            'R1_rect': R1_rect,
            'R2_rect': R2_rect,
            'P1_rect': P1_rect,
            'P2_rect': P2_rect,
            'Q': Q,
            'map1x': map1x, 'map1y': map1y,
            'map2x': map2x, 'map2y': map2y,
            'K1': K1, 'K2': K2,
            'R1_orig': R1_orig, 'R2_orig': R2_orig,
            't1_orig': t1_orig, 't2_orig': t2_orig,
            'rotation_angle': rotation_angle,
            'image_size': (image_width, image_height),  # Updated dimensions after rotation
            'original_image_size': (cam1.width, cam1.height),  # Keep original for reference
            'baseline': baseline
        }
        
    except Exception as e:
        print(f"Error in rectify_stereo_pair for images {img1_id}, {img2_id}: {e}")
        return None


def apply_rotation_to_image(image_array: np.ndarray, rotation_angle: float) -> np.ndarray:
    """
    Apply rotation to an image array using efficient transpose operations for 90° increments.
    
    Args:
        image_array: Input image (H, W, C) in [0, 1] range
        rotation_angle: Rotation angle in degrees
    
    Returns:
        Rotated image array (dimensions may change for 90°/270° rotations)
    """
    if rotation_angle == 0:
        return image_array
    
    # Normalize angle to [0, 360)
    angle = rotation_angle % 360
    
    # Use efficient transpose operations for 90° increments (no black padding!)
    if angle == 90:
        # 90° clockwise: transpose + horizontal flip
        # (H, W, C) -> (W, H, C)
        rotated = np.transpose(image_array, (1, 0, 2))  # Swap H and W
        rotated = np.flip(rotated, axis=1)  # Horizontal flip
        print(f"Applied 90° rotation: {image_array.shape} → {rotated.shape}")
        return rotated
        
    elif angle == 180:
        # 180°: vertical + horizontal flip (no dimension change)
        rotated = np.flip(np.flip(image_array, axis=0), axis=1)
        print(f"Applied 180° rotation: {image_array.shape} → {rotated.shape}")
        return rotated
        
    elif angle == 270:
        # 270° clockwise (= -90°): transpose + vertical flip  
        # (H, W, C) -> (W, H, C)
        rotated = np.transpose(image_array, (1, 0, 2))  # Swap H and W
        rotated = np.flip(rotated, axis=0)  # Vertical flip
        print(f"Applied 270° rotation: {image_array.shape} → {rotated.shape}")
        return rotated
        
    else:
        # For non-90° increments, fall back to cv2 (with potential padding)
        print(f"Warning: Using cv2 rotation for non-90° angle: {rotation_angle}° (may introduce black padding)")
        
        # Convert to uint8 for OpenCV
        image_uint8 = (image_array * 255).astype(np.uint8)
        h, w = image_uint8.shape[:2]
        
        # Get rotation matrix
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Apply rotation
        rotated_uint8 = cv2.warpAffine(image_uint8, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        
        # Convert back to float
        return rotated_uint8.astype(np.float32) / 255.0


def apply_rectification_to_image(image_array: np.ndarray, mapx: np.ndarray, mapy: np.ndarray, 
                               rotation_angle: float = 0, rotate_first: bool = True) -> np.ndarray:
    """
    Apply rotation and rectification to an image array.
    
    Args:
        image_array: Input image (H, W, C) in [0, 1] range
        mapx, mapy: Rectification maps from cv2.initUndistortRectifyMap
        rotation_angle: Rotation angle in degrees
        rotate_first: If True, apply rotation before rectification (recommended)
    
    Returns:
        Processed image array or None if processing fails
    """
    try:
        if image_array is None:
            print("Error: Input image_array is None")
            return None
        
        if mapx is None or mapy is None:
            print("Error: Rectification maps are None")
            return None
        
        processed_image = image_array
        
        # Apply rotation first if needed and rotate_first is True
        if rotation_angle != 0 and rotate_first:
            processed_image = apply_rotation_to_image(processed_image, rotation_angle)
            if processed_image is None:
                print("Error: Pre-rectification rotation failed")
                return None
        
        # Convert to uint8 for cv2.remap
        image_uint8 = (processed_image * 255).astype(np.uint8)
        
        # Apply rectification
        rectified_uint8 = cv2.remap(image_uint8, mapx, mapy, cv2.INTER_LINEAR)
        
        if rectified_uint8 is None:
            print("Error: cv2.remap returned None")
            return None
        
        # Convert back to float
        rectified_float = rectified_uint8.astype(np.float32) / 255.0
        
        # Apply rotation after rectification if needed (old behavior, not recommended)
        if rotation_angle != 0 and not rotate_first:
            rectified_float = apply_rotation_to_image(rectified_float, rotation_angle)
            if rectified_float is None:
                print("Error: Post-rectification rotation failed")
                return None
        
        return rectified_float
        
    except Exception as e:
        print(f"Error in apply_rectification_to_image: {e}")
        return None



def transfer_matches_to_original(matches_rect1: np.ndarray, matches_rect2: np.ndarray, 
                                rect_params: dict, model_size: tuple = (512, 384)) -> tuple[np.ndarray, np.ndarray]:
    """
    Transfer matches from model coordinates back to original image coordinates.
    
    Pipeline: Model coords → Crop inverse → Resize inverse → Rectification inverse → Original coords
    
    Note: Rotation is now applied to camera poses before rectification computation,
    so the rectification inverse automatically handles the rotation transformation.
    
    Args:
        matches_rect1, matches_rect2: Matches in model coordinates (after crop)
        rect_params: Rectification parameters from rectify_stereo_pair
        model_size: (width, height) of model input size
    
    Returns:
        Matches in original image coordinates
    """
    # Get rectification matrices
    R1_rect = rect_params['R1_rect']
    R2_rect = rect_params['R2_rect']
    P1_rect = rect_params['P1_rect']
    P2_rect = rect_params['P2_rect']
    K1 = rect_params['K1']
    K2 = rect_params['K2']
    
    # Get rotation information
    rotation_angle = rect_params.get('rotation_angle', 0)
    original_image_size = rect_params.get('image_size', (0, 0))  # (width, height)
    
    # Get model dimensions
    model_w, model_h = model_size
    
    def rect_to_original(matches_model, R_rect, P_rect, K_orig):
        """Convert model coordinates to original coordinates through full inverse pipeline."""
        orig_w, orig_h = original_image_size
        crop_params = rect_params.get('crop_params')
        
        # Step 1: Model coordinates → Resized coordinates (crop inverse)
        if crop_params:
            crop_offset_x, crop_offset_y = crop_params['crop_offset']
            matches_uncropped = matches_model + np.array([crop_offset_x, crop_offset_y])
            resized_w, resized_h = crop_params['resized_size']
        else:
            matches_uncropped = matches_model
            resized_w, resized_h = model_w, model_h
        
        # Step 2: Resized coordinates → Rectified + Rotated coordinates (resize inverse)
        # Scale from resized dimensions back to full rectified+rotated resolution
        scale_x = orig_w / resized_w  
        scale_y = orig_h / resized_h
        matches_scaled = matches_uncropped * np.array([scale_x, scale_y])
        
        # Step 3: Standard rectification inverse transformation
        # Note: Rotation is now handled within rectification since camera poses were rotated before rectification
        matches_rect_coords = matches_scaled
        # Add homogeneous coordinate
        matches_homo = np.column_stack([matches_rect_coords, np.ones(len(matches_rect_coords))])
        
        # Convert from rectified image coordinates to rectified camera coordinates
        P_rect_inv = np.linalg.pinv(P_rect[:, :3])
        matches_rect_cam = (P_rect_inv @ matches_homo.T).T
        
        # Transform from rectified camera coordinates to original camera coordinates
        matches_orig_cam = (R_rect.T @ matches_rect_cam.T).T
        
        # Project to original image coordinates
        matches_orig_homo = (K_orig @ matches_orig_cam.T).T
        matches_orig = matches_orig_homo[:, :2] / matches_orig_homo[:, [2]]
        
        return matches_orig
    
    matches_orig1 = rect_to_original(matches_rect1, R1_rect, P1_rect, K1)
    matches_orig2 = rect_to_original(matches_rect2, R2_rect, P2_rect, K2)
    
    return matches_orig1, matches_orig2


def main():
    """
    Main function for rectifying stereo pairs from command line.
    """
    parser = argparse.ArgumentParser(description='Rectify stereo image pairs using COLMAP reconstruction')
    parser.add_argument('-s', '--scene_folder', type=str, required=True, 
                       help='Path to scene folder containing sparse/ and images/ directories')
    parser.add_argument('-o', '--output_dir', type=str, default='rectified_output',
                       help='Output directory name (default: rectified_output)')
    parser.add_argument('-1', '--img1_id', type=int, required=True, help='First image ID from COLMAP')
    parser.add_argument('-2', '--img2_id', type=int, required=True, help='Second image ID from COLMAP')
    parser.add_argument('--alpha', type=float, default=0.5, 
                       help='Rectification cropping parameter: 0=crop to valid region, 1=keep all pixels (default: 0.5)')
    parser.add_argument('--disable_rotation', action='store_true',
                       help='Disable automatic rotation for horizontal epipolar lines')
    
    args = parser.parse_args()
    
    # Setup paths
    scene_folder = Path(args.scene_folder)
    sparse_dir = scene_folder / 'sparse'
    images_dir = scene_folder / 'images'
    output_dir = scene_folder / args.output_dir
    
    # Validate input paths
    if not sparse_dir.exists():
        print(f"Error: Sparse directory not found at {sparse_dir}")
        return 1
    
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load COLMAP reconstruction
        print(f"Loading COLMAP reconstruction from {sparse_dir}")
        reconstruction = ColmapReconstruction(str(sparse_dir))
        
        print(f"Rectifying pair: Image ID {args.img1_id} <-> Image ID {args.img2_id}")
        print("Computing stereo rectification...")
        print(f"  - Alpha (cropping): {args.alpha} (0=crop more, 1=keep more content)")
        print(f"  - Rotation: {'enabled' if not args.disable_rotation else 'disabled'}")
        
        # Use the new rectification function
        result = rectify_image_pair_from_colmap(
            reconstruction, args.img1_id, args.img2_id, images_dir,
            alpha=args.alpha, enable_rotation=not args.disable_rotation
        )
        
        img1_rect = result['img1_rect']
        img2_rect = result['img2_rect']
        metadata = result['metadata']
        img1_name = result['img1_name']
        img2_name = result['img2_name']
        
        print(f"Rectified pair: {img1_name} (ID: {args.img1_id}) <-> {img2_name} (ID: {args.img2_id})")
        print(f"Rectification computed successfully:")
        print(f"  - Baseline: {metadata['rectification_params']['baseline']:.3f}")
        print(f"  - Rotation angle: {metadata['rectification_params']['rotation_angle']:.1f}°")
        print(f"  - Image size: {metadata['rectification_params']['image_size']}")
        
        # Save rectified images
        img1_output = output_dir / f"{args.img1_id:06d}_rectified.jpg"
        img2_output = output_dir / f"{args.img2_id:06d}_rectified.jpg"
        
        img1_rect_pil = Image.fromarray((img1_rect * 255).astype(np.uint8))
        img2_rect_pil = Image.fromarray((img2_rect * 255).astype(np.uint8))
        
        img1_rect_pil.save(img1_output)
        img2_rect_pil.save(img2_output)
        
        print(f"Saved rectified images:")
        print(f"  - {img1_output}")
        print(f"  - {img2_output}")
        
        # Save metadata
        metadata_path = output_dir / f"rectification_metadata_{args.img1_id:06d}_{args.img2_id:06d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Saved metadata: {metadata_path}")
        print("Rectification completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during rectification: {e}")
        return 1


def rectify_image_pair_from_colmap(reconstruction, img1_id: int, img2_id: int, images_dir, 
                                   alpha: float = 0.5, enable_rotation: bool = True):
    """
    Rectify a stereo image pair using COLMAP reconstruction data.
    
    Args:
        reconstruction: ColmapReconstruction object
        img1_id: First image ID
        img2_id: Second image ID  
        images_dir: Path to directory containing images
        alpha: Rectification cropping parameter (0=crop more, 1=keep more)
        enable_rotation: Whether to enable automatic rotation for horizontal epipolar lines
    
    Returns:
        dict containing:
            - 'img1_rect': Rectified first image as numpy array
            - 'img2_rect': Rectified second image as numpy array  
            - 'metadata': Complete metadata dictionary for coordinate transformations
            - 'img1_name': First image filename
            - 'img2_name': Second image filename
    """
    # Validate image IDs
    if not reconstruction.has_image(img1_id):
        raise ValueError(f"Image ID {img1_id} not found in reconstruction")
        
    if not reconstruction.has_image(img2_id):
        raise ValueError(f"Image ID {img2_id} not found in reconstruction")
    
    # Get image names
    img1_name = reconstruction.get_image_name(img1_id)
    img2_name = reconstruction.get_image_name(img2_id)
    
    # Compute rectification parameters
    rect_params = rectify_stereo_pair(
        reconstruction, img1_id, img2_id, 
        alpha=alpha, enable_rotation=enable_rotation
    )
    
    if rect_params is None:
        raise RuntimeError("Failed to compute rectification parameters")
    
    # Load images
    img1_path = Path(images_dir) / img1_name
    img2_path = Path(images_dir) / img2_name
    
    if not img1_path.exists():
        raise FileNotFoundError(f"Image not found: {img1_path}")
    if not img2_path.exists():
        raise FileNotFoundError(f"Image not found: {img2_path}")
    
    img1_pil = Image.open(img1_path)
    img2_pil = Image.open(img2_path)
    
    # Convert to numpy arrays
    img1_array = np.array(img1_pil, dtype=np.float32) / 255.0
    img2_array = np.array(img2_pil, dtype=np.float32) / 255.0
    
    # Apply rectification (includes rotation if enabled)
    rotation_angle = rect_params.get('rotation_angle', 0)
    
    img1_rect = apply_rectification_to_image(img1_array, rect_params['map1x'], rect_params['map1y'], 
                                           rotation_angle, rotate_first=True)
    img2_rect = apply_rectification_to_image(img2_array, rect_params['map2x'], rect_params['map2y'], 
                                           rotation_angle, rotate_first=True)
    
    if img1_rect is None or img2_rect is None:
        raise RuntimeError("Rectification failed")
    
    # Build comprehensive metadata for coordinate transformations
    metadata = {
        'images': {
            'img1_id': img1_id,
            'img2_id': img2_id,
            'img1_name': img1_name,
            'img2_name': img2_name,
            'img1_path': str(img1_path),
            'img2_path': str(img2_path)
        },
        'rectification_params': {
            'baseline': float(rect_params['baseline']),
            'rotation_angle': float(rect_params['rotation_angle']),
            'alpha': alpha,
            'rotation_enabled': enable_rotation,
            'image_size': rect_params['image_size'],  # (W, H) - rectified image size
            'original_image_size': rect_params['original_image_size']  # (W, H) - original COLMAP size
        },
        'transformation_pipeline': {
            # Camera matrices (updated for rotation)
            'K1': rect_params['K1'].tolist(),
            'K2': rect_params['K2'].tolist(),
            
            # Rectification matrices
            'R1_rect': rect_params['R1_rect'].tolist(),
            'R2_rect': rect_params['R2_rect'].tolist(),
            'P1_rect': rect_params['P1_rect'].tolist(),
            'P2_rect': rect_params['P2_rect'].tolist(),
            'Q': rect_params['Q'].tolist(),
            
            # Original camera poses
            'R1_orig': rect_params['R1_orig'].tolist(),
            'R2_orig': rect_params['R2_orig'].tolist(),
            't1_orig': rect_params['t1_orig'].tolist(),
            't2_orig': rect_params['t2_orig'].tolist(),
            
            # Image dimensions at each transformation stage
            'loaded_image_dims': {
                'img1': list(img1_array.shape[:2]),  # [H, W] as loaded from disk
                'img2': list(img2_array.shape[:2])   # [H, W] as loaded from disk
            },
            'rotation_angle': float(rect_params['rotation_angle']),
            
            # Transformation sequence: Original → Rotation → Rectification
            'transform_sequence': [
                {
                    'step': 'rotation',
                    'angle': float(rect_params['rotation_angle']),
                    'input_dims': list(img1_array.shape[:2]),  # [H, W]
                    'output_dims': [img1_array.shape[1], img1_array.shape[0]] if rect_params['rotation_angle'] in [90, 270] else list(img1_array.shape[:2])
                },
                {
                    'step': 'rectification',
                    'input_dims': rect_params['image_size'],  # (W, H)
                    'output_dims': rect_params['image_size']   # (W, H)
                }
            ]
        }
    }
    
    return {
        'img1_rect': img1_rect,
        'img2_rect': img2_rect,
        'metadata': metadata,
        'img1_name': img1_name,
        'img2_name': img2_name
    }


def rectified_to_original_coords(rect_points: np.ndarray, metadata: dict, image_idx: int = 1) -> np.ndarray:
    """
    Transform coordinates from rectified image space back to original image space.
    
    Correct Pipeline:
    1. Cropped rectified coords → Uncropped rectified coords (if there was cropping)
    2. Uncropped rectified coords → Rotated coords (inverse rectification with updated K)
    3. Rotated coords → Original coords (inverse rotation)
    
    Args:
        rect_points: Array of points in rectified space, shape (N, 2) [x, y]
        metadata: Rectification metadata dictionary
        image_idx: Which image (1 or 2)
    
    Returns:
        Array of points in original image space, shape (N, 2) [x, y]
    """
    transform_info = metadata['transformation_pipeline']
    rect_params = metadata['rectification_params']
    rotation_angle = transform_info['rotation_angle']
    alpha = rect_params['alpha']
    
    # Get the appropriate camera matrices and rectification parameters
    if image_idx == 1:
        K_updated = np.array(transform_info['K1'])  # K after rotation update
        R_rect = np.array(transform_info['R1_rect'])
        P_rect = np.array(transform_info['P1_rect'])
        loaded_dims = transform_info['loaded_image_dims']['img1']  # [H, W]
    else:
        K_updated = np.array(transform_info['K2'])  # K after rotation update
        R_rect = np.array(transform_info['R2_rect'])
        P_rect = np.array(transform_info['P2_rect'])
        loaded_dims = transform_info['loaded_image_dims']['img2']  # [H, W]
    
    original_size = rect_params['original_image_size']  # [W, H]
    rectified_size = rect_params['image_size']  # [W, H] - size after rotation
    
    # Step 1: Handle uncropping (if alpha < 1, there might be cropping)
    # For now, assume no cropping since we're working with full rectified images
    # TODO: Implement cropping inverse if needed based on valid ROI
    uncropped_points = rect_points.copy()
    
    # Step 2: Inverse rectification (rectified coords → rotated coords)
    # This uses the updated K matrices (after rotation) and gives us coordinates in rotated space
    rect_points_homo = np.column_stack([uncropped_points, np.ones(len(uncropped_points))])
    
    # Use pseudoinverse of P_rect 
    P_rect_inv = np.linalg.pinv(P_rect[:, :3])
    rect_cam_coords = (P_rect_inv @ rect_points_homo.T).T
    
    # Apply inverse rectification rotation (rectified camera → rotated camera)
    rotated_cam_coords = (R_rect.T @ rect_cam_coords.T).T
    
    # Convert to rotated image coordinates using updated K matrix
    rotated_img_coords_homo = (K_updated @ rotated_cam_coords.T).T
    rotated_img_coords = rotated_img_coords_homo[:, :2] / rotated_img_coords_homo[:, 2:3]
    
    # Step 3: Inverse rotation (rotated coords → original coords)
    if rotation_angle != 0:
        original_coords = apply_inverse_rotation(rotated_img_coords, rotation_angle, loaded_dims)
        return original_coords
    else:
        return rotated_img_coords


def apply_inverse_rotation(points: np.ndarray, rotation_angle: float, image_dims: list) -> np.ndarray:
    """
    Apply inverse rotation to points.
    
    Args:
        points: Points in rotated image space, shape (N, 2) [x, y]
        rotation_angle: Original rotation angle in degrees
        image_dims: Image dimensions [H, W] of the image BEFORE rotation
    
    Returns:
        Points in original (pre-rotation) image space
    """
    if rotation_angle == 0:
        return points
    
    h, w = image_dims  # Original image dimensions
    
    # Normalize angle
    angle = rotation_angle % 360
    
    if angle == 90:
        # Forward: (x, y) → (y, w-1-x)  [90° clockwise, using transpose + horizontal flip]
        # Inverse: (x', y') → (y', h-1-x')  [correct inverse found through testing]
        original_points = np.column_stack([
            points[:, 1],          # x_orig = y_rot
            h - 1 - points[:, 0]   # y_orig = h-1-x_rot
        ])
        
    elif angle == 180:
        # Forward: (x, y) → (w-1-x, h-1-y)
        # Inverse: (x', y') → (w-1-x', h-1-y')  [same transformation]
        original_points = np.column_stack([
            w - 1 - points[:, 0],  # x_orig = w-1-x_rot
            h - 1 - points[:, 1]   # y_orig = h-1-y_rot
        ])
        
    elif angle == 270:
        # Forward: (x, y) → (h-1-y, x)  [270° clockwise = transpose + vertical flip]
        # Inverse: (x', y') → (y', h-1-x')  [90° counter-clockwise]
        original_points = np.column_stack([
            points[:, 1],          # x_orig = y_rot
            h - 1 - points[:, 0]   # y_orig = h-1-x_rot
        ])
        
    else:
        # For non-90° increments, this is more complex and would require
        # the full rotation matrix approach
        print(f"Warning: Inverse rotation for {rotation_angle}° not implemented with exact precision")
        original_points = points  # Fallback
    
    return original_points


if __name__ == "__main__":
    exit(main()) 