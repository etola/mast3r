"""
Configuration Management for MASt3R Densification
================================================

This module provides a structured way to manage all configuration parameters
for the densification process, making the code more maintainable and reducing
parameter passing complexity.

Author: Blake Troutman 2025
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch
from typing import Optional


@dataclass
class DensificationConfig:
    """Configuration class for MASt3R densification parameters."""
    
    # Core paths
    scene_dir: str
    output_dir: str = "densified_output"
    reconstruction_path: Optional[str] = None
    img_dir: Optional[str] = None
    output_path: Optional[str] = None
    pairs_path: Optional[str] = None
    
    # Processing parameters
    batch_size: int = 1
    sampling_factor: int = 8
    min_feature_coverage: float = 0.6
    force_cpu: bool = False
    verbose: bool = False
    use_existing_pairs: bool = False
    cache_memory_gb: float = 16.0
    
    # Multi-pairing consistency parameters
    enable_consistency_check: bool = False
    max_pairs_per_image: int = 7
    min_consistent_pairs: int = 3
    depth_consistency_threshold: float = 0.05
    
    # Bounding box filtering parameters
    disable_bbox_filter: bool = False
    min_point_visibility: int = 3
    bbox_padding_factor: float = 1.0
    
    # Point cloud outlier removal parameters
    enable_outlier_removal: bool = False
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    
    # Matching parameters
    block_size_power: int = 14
    
    # Model parameters (fixed for MASt3R)
    model_w: int = 512
    model_h: int = 384
    size: int = 512
    model_path: str = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    
    def __post_init__(self):
        """Initialize derived paths after object creation."""
        if self.reconstruction_path is None:
            self.reconstruction_path = str(Path(self.scene_dir) / "sparse")
        if self.img_dir is None:
            self.img_dir = str(Path(self.scene_dir) / "images")
    
    def get_device(self) -> torch.device:
        """Get the appropriate torch device based on configuration."""
        if self.force_cpu:
            return torch.device("cpu")
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def setup_output_paths(self) -> None:
        """Set up output directory and file paths."""
        output_dir_path = Path(self.scene_dir) / self.output_dir
        output_dir_path.mkdir(exist_ok=True)
        
        self.output_path = str(output_dir_path / "dense.ply")
        self.pairs_path = str(output_dir_path / "pairs.json")
        
        if self.verbose:
            print(f"Created output directory: {output_dir_path}")
    
    def validate_paths(self) -> None:
        """Validate that required directories exist."""
        scene_path = Path(self.scene_dir)
        reconstruction_path = Path(self.reconstruction_path)
        img_path = Path(self.img_dir)
        
        if not scene_path.exists():
            raise ValueError(f"Scene directory does not exist: {self.scene_dir}")
        if not reconstruction_path.exists():
            raise ValueError(f"Sparse reconstruction directory does not exist: {self.reconstruction_path}")
        if not img_path.exists():
            raise ValueError(f"Images directory does not exist: {self.img_dir}")
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'DensificationConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def print_summary(self) -> None:
        """Print a summary of the configuration."""
        print("=" * 60)
        print("DENSIFICATION CONFIGURATION")
        print("=" * 60)
        print(f"Scene directory: {self.scene_dir}")
        print(f"Output directory: {Path(self.scene_dir) / self.output_dir}")
        print(f"Dense point cloud: {Path(self.output_path).name if self.output_path else 'dense.ply'}")
        print(f"Pairs file: {Path(self.pairs_path).name if self.pairs_path else 'pairs.json'}")
        print()
        print("Processing Parameters:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Sampling factor: {self.sampling_factor}")
        print(f"  Min feature coverage: {self.min_feature_coverage}")
        print(f"  Device: {'CPU' if self.force_cpu else 'GPU (if available)'}")
        print(f"  Image cache memory: {self.cache_memory_gb:.1f}GB")
        print(f"  Verbose: {self.verbose}")
        print()
        print("Advanced Options:")
        print(f"  Consistency checking: {self.enable_consistency_check}")
        if self.enable_consistency_check:
            print(f"    Max pairs per image: {self.max_pairs_per_image}")
            print(f"    Min consistent pairs: {self.min_consistent_pairs}")
            print(f"    Depth threshold: {self.depth_consistency_threshold}")
        print(f"  Bounding box filtering: {'Disabled' if self.disable_bbox_filter else 'Enabled (default)'}")
        if not self.disable_bbox_filter:
            print(f"    Min point visibility: {self.min_point_visibility}")
            print(f"    Padding factor: {self.bbox_padding_factor}")
        print(f"  Outlier removal: {self.enable_outlier_removal}")
        if self.enable_outlier_removal:
            print(f"    Neighbors: {self.outlier_nb_neighbors}")
            print(f"    Std ratio: {self.outlier_std_ratio}")
        print(f"  Block size for matching: 2^{self.block_size_power}")
        print("=" * 60)
    
    def get_block_size(self) -> int:
        """Get the computed block size for fast matching."""
        return 2 ** self.block_size_power


def create_config_from_args(args) -> DensificationConfig:
    """Create a DensificationConfig from command line arguments."""
    return DensificationConfig(
        scene_dir=args.scene_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sampling_factor=args.sampling_factor,
        min_feature_coverage=args.min_feature_coverage,
        force_cpu=args.force_cpu,
        verbose=args.verbose,
        use_existing_pairs=args.use_existing_pairs,
        cache_memory_gb=args.cache_memory_gb,
        enable_consistency_check=args.enable_consistency_check,
        max_pairs_per_image=args.max_pairs_per_image,
        min_consistent_pairs=args.min_consistent_pairs,
        depth_consistency_threshold=args.depth_consistency_threshold,
        disable_bbox_filter=args.disable_bbox_filter,
        min_point_visibility=args.min_point_visibility,
        bbox_padding_factor=args.bbox_padding_factor,
        enable_outlier_removal=args.enable_outlier_removal,
        outlier_nb_neighbors=args.outlier_nb_neighbors,
        outlier_std_ratio=args.outlier_std_ratio,
        block_size_power=args.block_size_power,
    ) 