# Image Rectification for MASt3R

This document explains how to use the new standalone rectification functionality for stereo image pairs.

## Overview

The rectification code has been separated from the main densification pipeline into a dedicated `rectify_images.py` script. This allows for standalone rectification and easier debugging of rectification parameters. The script automatically creates side-by-side comparison images for visual inspection.

**✨ Recent Improvements:** Now properly handles camera intrinsics updates during rotation, ensuring correct geometry and better content preservation when using `--alpha 1.0` with rotation enabled.

## Files

- **`rectify_images.py`** - Core rectification functions and command-line interface
- **`mark_points.py`** - Point marking utility for transforming rectified coordinates back to original images
- **`densify_mast3r.py`** - Main densification script (now cleaned of rectification code)

## Usage

### Command Line Interface

Rectify a stereo pair using COLMAP reconstruction data:

```bash
python rectify_images.py -s /path/to/scene -1 59 -2 58 -o rectified_output --alpha 0.8
```

**Arguments:**
- `-s, --scene_folder` - Path to scene folder containing `sparse/` and `images/` directories
- `-1, --img1_id` - First image ID from COLMAP reconstruction
- `-2, --img2_id` - Second image ID from COLMAP reconstruction  
- `-o, --output_dir` - Output directory name (default: rectified_output)
- `--alpha` - Rectification cropping parameter: 0=crop to valid region, 1=keep all pixels (default: 0.5)
- `--disable_rotation` - Disable automatic rotation for horizontal epipolar lines

**Examples:**

Basic usage:
```bash
python rectify_images.py -s ~/data/mini-7ee5/glomap/dense/ -1 59 -2 58 -o rectified_pairs
```

Preserve more content (less cropping):
```bash
python rectify_images.py -s ~/data/mini-7ee5/glomap/dense/ -1 59 -2 58 -o rectified_pairs --alpha 0.8
```

Keep all content (no cropping) - now works well with rotation:
```bash
python rectify_images.py -s ~/data/mini-7ee5/glomap/dense/ -1 59 -2 58 -o rectified_pairs --alpha 1.0
```

Disable rotation (keep original epipolar line orientation):
```bash
python rectify_images.py -s ~/data/mini-7ee5/glomap/dense/ -1 59 -2 58 -o rectified_pairs --disable_rotation
```

### Output

The script will create:
- `{img1_id:06d}_rectified.jpg` - Rectified first image
- `{img2_id:06d}_rectified.jpg` - Rectified second image  
- `original_pair_{img1_id:06d}_{img2_id:06d}.jpg` - Side-by-side original image pair
- `rectified_pair_{img1_id:06d}_{img2_id:06d}.jpg` - Side-by-side rectified image pair  
- `rectification_metadata_{img1_id:06d}_{img2_id:06d}.json` - Rectification metadata (always saved)

The comparison images make it easy to:
- **Original pair**: Verify that the correct images were selected
- **Rectified pair**: Check that epipolar lines are horizontal (corresponding points should be at the same y-coordinate)

## Content Preservation Options

### Alpha Parameter (Cropping Control)

The `--alpha` parameter controls how much image content is preserved during rectification:

- **`--alpha 0.0`** (most cropping) - Only keeps the overlapping valid region between both cameras
- **`--alpha 0.5`** (default) - Balanced compromise between content and black borders
- **`--alpha 1.0`** (least cropping) - Keeps all original image content, may have black borders

**If you're seeing too much cropping**, try:
```bash
python rectify_images.py -s ~/data/scene/ -1 59 -2 58 --alpha 0.8
```

**New in this version**: Rotation is now applied to camera poses before rectification computation, which means `--alpha 1.0` with rotation enabled should preserve much more content than before.

### Rotation Control

- **Default**: Automatically detects epipolar line orientation and rotates images to make them horizontal
- **`--disable_rotation`**: Keeps original orientation, useful if images are already well-aligned

## Point Marking

After rectification, you can mark points on the rectified images and automatically transform them back to the original image coordinates using the saved metadata.

### Mark Points on Original Images

```bash
python mark_points.py rectification_metadata_000059_000058.json --points 512,200 100,300 --color red --labels "Point1" "Point2"
```

**Arguments:**
- `metadata` - Path to the rectification metadata JSON file
- `--points` - Points in rectified image coordinates (format: x1,y1 x2,y2 ...)
- `--image` - Which image to mark: '1', '2', or 'both' (default: both)
- `-o, --output` - Output directory for marked images (default: marked_images)
- `--radius` - Radius of marking circles (default: 10)
- `--color` - Color of markings: red, green, blue, yellow, etc. (default: red)
- `--labels` - Optional text labels for each point

**Examples:**

Mark center point on both images:
```bash
python mark_points.py metadata.json --points 256,192 --color green --labels "Center"
```

Mark multiple points on image 1 only:
```bash
python mark_points.py metadata.json --points 100,50 400,300 200,200 --image 1 --color blue --radius 15
```

The script will:
1. Load the transformation parameters from the metadata JSON
2. Apply the inverse transformation: rectified → original coordinates
3. Mark the transformed points on the original images
4. Save marked images to the output directory

## Core Functions

### `rectify_stereo_pair(reconstruction, img1_id, img2_id, alpha=0.5, enable_rotation=True)`

Computes stereo rectification parameters for a pair of images.

**Args:**
- `alpha` - Rectification cropping parameter (0=crop to valid region, 1=keep all pixels)
- `enable_rotation` - Whether to apply rotation for horizontal epipolar lines

**Returns:** Dictionary containing:
- Rectification matrices (`R1_rect`, `R2_rect`, `P1_rect`, `P2_rect`)
- Rectification maps (`map1x`, `map1y`, `map2x`, `map2y`)
- **Updated camera matrices** (`K1`, `K2`) - intrinsics adjusted for rotation
- Rotation angle for horizontal epipolar lines
- Baseline distance
- **Updated image dimensions** (`image_size`) - swapped for 90°/270° rotations
- Original image size (`original_image_size`) for reference

### `apply_rectification_to_image(image_array, mapx, mapy, rotation_angle, rotate_first=True)`

Applies rotation and rectification to an image array.

**Args:**
- `image_array` - Input image as numpy array (H, W, C) in [0, 1] range
- `mapx, mapy` - Rectification maps from `cv2.initUndistortRectifyMap`
- `rotation_angle` - Rotation angle in degrees
- `rotate_first` - If True, apply rotation before rectification (recommended for better content preservation)

**Returns:** Processed image array or None if failed

### `transfer_matches_to_original(matches_rect1, matches_rect2, rect_params, model_size)`

Transfers matches from model coordinates back to original image coordinates.

**Pipeline:** Model coords → Crop inverse → Resize inverse → Rectification inverse → Original coords

**Note:** Rotation is now handled within rectification since camera poses are rotated before rectification computation.

### `update_intrinsics_for_rotation(K, orig_width, orig_height, rotation_angle)`

Updates camera intrinsics matrix for image rotation.

**Args:**
- `K` - Original intrinsics matrix (3x3)
- `orig_width, orig_height` - Original image dimensions
- `rotation_angle` - Rotation angle in degrees (0, 90, 180, 270)

**Returns:** Updated intrinsics matrix with:
- **Principal point coordinates** transformed for the rotation
- **Focal lengths swapped** for 90°/270° rotations (fx↔fy) 
- Correct pixel mapping for the rotated coordinate system

### `detect_epipolar_orientation(reconstruction, img1_id, img2_id)`

Detects epipolar line orientation and determines rotation needed for horizontal epipolar lines.

**Returns:** Rotation angle in degrees (0, 90, 180, 270)

### `create_comparison_image(img1_path, img2_path, output_path, title)`

Creates a side-by-side comparison image from two input images.

**Args:**
- `img1_path` - Path to first image
- `img2_path` - Path to second image  
- `output_path` - Path to save the merged image
- `title` - Title to add to the image

## Integration with Densification

To integrate rectification with the densification pipeline:

1. Use `rectify_images.py` functions to compute rectification parameters
2. Apply rectification to images before MASt3R inference
3. Use `transfer_matches_to_original()` to convert matches back to original coordinates
4. Perform triangulation using original camera parameters

## Example Python Usage

```python
from rectify_images import rectify_stereo_pair, apply_rectification_to_image
from colmap_utils import ColmapReconstruction
import numpy as np
from PIL import Image

# Load COLMAP reconstruction
reconstruction = ColmapReconstruction('/path/to/sparse')

# Compute rectification for image pair (with content preservation)
rect_params = rectify_stereo_pair(reconstruction, img1_id=59, img2_id=58, alpha=0.8, enable_rotation=True)

if rect_params is not None:
    # Load images
    img1 = np.array(Image.open('image1.jpg'), dtype=np.float32) / 255.0
    img2 = np.array(Image.open('image2.jpg'), dtype=np.float32) / 255.0
    
    # Apply rectification
    img1_rect = apply_rectification_to_image(
        img1, rect_params['map1x'], rect_params['map1y'], 
        rect_params['rotation_angle']
    )
    img2_rect = apply_rectification_to_image(
        img2, rect_params['map2x'], rect_params['map2y'], 
        rect_params['rotation_angle']
    )
    
    # Save rectified images
    Image.fromarray((img1_rect * 255).astype(np.uint8)).save('img1_rectified.jpg')
    Image.fromarray((img2_rect * 255).astype(np.uint8)).save('img2_rectified.jpg')
    
    # Optionally create comparison images
    from rectify_images import create_comparison_image
    create_comparison_image('image1.jpg', 'image2.jpg', 'original_pair.jpg', 'Original Pair')
    create_comparison_image('img1_rectified.jpg', 'img2_rectified.jpg', 'rectified_pair.jpg', 'Rectified Pair')
```

## Notes

- **Improved rectification approach**: Rotation is applied to camera poses AND intrinsics BEFORE computing rectification parameters, ensuring better content preservation and correct geometry
- The rectification process automatically detects epipolar line orientation and applies rotation to ensure horizontal epipolar lines (can be disabled with `--disable_rotation`)
- Rectification uses OpenCV's `stereoRectify` with `CALIB_ZERO_DISPARITY` flag
- Distortion coefficients are assumed to be zero for simplicity
- The coordinate transformation pipeline: Model coords → Crop inverse → Resize inverse → Rectification inverse → Original coords
- **Alpha parameter tradeoffs**:
  - Lower alpha (0.0-0.3): Less black borders, more content cropped
  - Higher alpha (0.7-1.0): More content preserved, may have black borders from rectification
  - Default (0.5): Good balance for most cases
- **Content preservation**: With rotation applied to camera poses and intrinsics first, `--alpha 1.0` should now preserve much more content even with rotation enabled
- **Intrinsics handling**: Camera intrinsics (focal lengths, principal point) are automatically updated when rotation is applied to ensure correct pixel-to-ray mapping
- **Efficient 90° rotations**: Uses transpose operations for 90°/270° rotations to avoid black padding and preserve all image content (H×W → W×H) 