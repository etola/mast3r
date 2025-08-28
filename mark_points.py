#!/usr/bin/env python3
"""
Point Marking for Rectified Images
==================================

This script takes coordinates on rectified images and transforms them back to
original image coordinates, then marks those points on the original images.

Usage:
    python mark_points.py metadata.json --points x1,y1 x2,y2 ... --output marked_images/
    python mark_points.py metadata.json --points 512,200 100,150 --radius 10 --color red

Author: AI Assistant 2025
"""

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import cv2
from rectify_images import rectified_to_original_coords


def load_metadata(metadata_path: str) -> dict:
    """Load rectification metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)





def mark_points_on_image(image_path: str, points: np.ndarray, output_path: str, 
                        radius: int = 10, color: str = 'red', point_labels: list = None):
    """
    Mark points on an image and save the result.
    
    Args:
        image_path: Path to input image
        points: Array of points to mark, shape (N, 2) [x, y]
        output_path: Path to save marked image
        radius: Radius of marking circles
        color: Color of marking circles
        point_labels: Optional labels for each point
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Color mapping
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'magenta': (255, 0, 255),
        'cyan': (0, 255, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }
    
    color_rgb = color_map.get(color.lower(), (255, 0, 0))  # Default to red
    
    # Mark each point
    for i, (x, y) in enumerate(points):
        # Draw circle
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                    outline=color_rgb, width=3)
        
        # Draw cross in center
        draw.line([x - radius//2, y, x + radius//2, y], fill=color_rgb, width=2)
        draw.line([x, y - radius//2, x, y + radius//2], fill=color_rgb, width=2)
        
        # Add label if provided
        if point_labels and i < len(point_labels):
            draw.text((x + radius + 5, y - radius), str(point_labels[i]), 
                     fill=color_rgb)
    
    # Save marked image
    img.save(output_path, quality=95)
    print(f"Saved marked image: {output_path}")


def parse_points(point_strings: list) -> np.ndarray:
    """Parse point coordinates from command line arguments."""
    points = []
    for point_str in point_strings:
        try:
            x, y = map(float, point_str.split(','))
            points.append([x, y])
        except ValueError:
            print(f"Error: Invalid point format '{point_str}'. Use format 'x,y'")
            sys.exit(1)
    
    return np.array(points)


def main():
    parser = argparse.ArgumentParser(description="Mark points on original images using rectified coordinates")
    parser.add_argument('metadata', help='Path to rectification metadata JSON file')
    parser.add_argument('--points', nargs='+', required=True, 
                       help='Points in rectified image coordinates (format: x1,y1 x2,y2 ...)')
    parser.add_argument('--image', choices=['1', '2', 'both'], default='both',
                       help='Which image to mark points on (default: both)')
    parser.add_argument('-o', '--output', default='marked_images',
                       help='Output directory for marked images (default: marked_images)')
    parser.add_argument('--radius', type=int, default=10,
                       help='Radius of marking circles (default: 10)')
    parser.add_argument('--color', default='red',
                       help='Color of markings (default: red)')
    parser.add_argument('--labels', nargs='*',
                       help='Optional labels for each point')
    
    args = parser.parse_args()
    
    # Parse rectified coordinates
    rect_points = parse_points(args.points)
    print(f"Input rectified coordinates: {len(rect_points)} points")
    for i, (x, y) in enumerate(rect_points):
        label = f" ({args.labels[i]})" if args.labels and i < len(args.labels) else ""
        print(f"  Point {i+1}: ({x:.1f}, {y:.1f}){label}")
    
    # Load metadata
    metadata = load_metadata(args.metadata)
    print(f"Loaded metadata from {args.metadata}")
    print(f"  - Rotation angle: {metadata['transformation_pipeline']['rotation_angle']}°")
    print(f"  - Rectified image size: {metadata['rectification_params']['image_size']}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Process images
    images_to_process = []
    if args.image in ['1', 'both']:
        images_to_process.append(1)
    if args.image in ['2', 'both']:
        images_to_process.append(2)
    
    for img_idx in images_to_process:
        print(f"\nProcessing image {img_idx}:")
        
        # Transform coordinates
        orig_points = rectified_to_original_coords(rect_points, metadata, img_idx)
        
        print(f"  Transformed to original coordinates:")
        for i, (x, y) in enumerate(orig_points):
            label = f" ({args.labels[i]})" if args.labels and i < len(args.labels) else ""
            print(f"    Point {i+1}: ({x:.1f}, {y:.1f}){label}")
        
        # Get original image path
        img_key = f'img{img_idx}_path'
        if img_key not in metadata['images']:
            print(f"  Warning: {img_key} not found in metadata, using relative path")
            img_name = metadata['images'][f'img{img_idx}_name']
            # Try to find image relative to metadata file
            metadata_dir = Path(args.metadata).parent
            img_path = metadata_dir.parent / 'images' / img_name
        else:
            img_path = Path(metadata['images'][img_key])
        
        if not img_path.exists():
            print(f"  Error: Image not found at {img_path}")
            continue
        
        # Mark points and save
        img_id = metadata['images'][f'img{img_idx}_id']
        output_path = output_dir / f"marked_{img_id:06d}_{img_path.stem}.jpg"
        
        mark_points_on_image(
            str(img_path), orig_points, str(output_path),
            radius=args.radius, color=args.color, point_labels=args.labels
        )
    
    print(f"\nMarked images saved to: {output_dir}")


if __name__ == '__main__':
    main() 