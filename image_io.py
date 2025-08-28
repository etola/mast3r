"""
Image I/O and Caching System for MASt3R
=======================================

This module provides efficient image loading and caching with memory management.
It maintains the same interface as the original dust3r load_images function while
adding intelligent caching to avoid redundant loading and resizing operations.

Features:
- LRU cache with configurable memory limit
- Automatic memory tracking and eviction
- Thread-safe operations
- Same interface as dust3r.utils.image.load_images
"""

import os
import sys
import time
import threading
from collections import OrderedDict
from typing import List, Dict, Union, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image as PIL_Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf

# Add current directory to path to find dust3r modules
current_dir = os.path.dirname(os.path.abspath(__file__))
dust3r_path = os.path.join(current_dir, 'dust3r')
if dust3r_path not in sys.path:
    sys.path.insert(0, dust3r_path)

# Import the original ImgNorm transform
try:
    from dust3r.utils.image import _resize_pil_image
    from dust3r.datasets.utils.transforms import ImgNorm
except ImportError:
    # Fallback definition if imports fail
    ImgNorm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def _resize_pil_image(img, long_edge_size):
        """Fallback resize function"""
        S = max(img.size)
        if S > long_edge_size:
            interp = PIL_Image.LANCZOS
        elif S <= long_edge_size:
            interp = PIL_Image.BICUBIC
        new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
        return img.resize(new_size, interp)

# Check for HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False


class ImageCache:
    """
    LRU cache for images with memory management.
    
    Caches processed images to avoid redundant loading and resizing operations.
    Automatically evicts least recently used images when memory limit is exceeded.
    """
    
    def __init__(self, max_memory_gb: float = 32.0):
        """
        Initialize the image cache.
        
        Args:
            max_memory_gb: Maximum memory usage in GB (default: 32GB)
        """
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.current_memory_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.RLock()
        
        print(f"Initialized ImageCache with {max_memory_gb:.1f}GB memory limit")
    
    def _estimate_memory_usage(self, img_data: Dict[str, Any]) -> int:
        """
        Estimate memory usage of a cached image entry.
        
        Args:
            img_data: Dictionary containing image data
            
        Returns:
            Estimated memory usage in bytes
        """
        memory = 0
        
        # Estimate tensor memory (img field)
        if 'img' in img_data:
            tensor = img_data['img']
            if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                memory += tensor.numel() * tensor.element_size()
            else:
                # Fallback estimation
                memory += sys.getsizeof(tensor)
        
        # Add memory for other fields
        for key, value in img_data.items():
            if key != 'img':
                memory += sys.getsizeof(value)
        
        return memory
    
    def _evict_lru(self, required_memory: int) -> None:
        """
        Evict least recently used items until enough memory is available.
        
        Args:
            required_memory: Bytes of memory needed
        """
        while (self.current_memory_bytes + required_memory > self.max_memory_bytes 
               and self.cache):
            # Remove oldest item
            cache_key, img_data = self.cache.popitem(last=False)
            memory_freed = self._estimate_memory_usage(img_data)
            self.current_memory_bytes -= memory_freed
            print(f"Evicted cached image: {cache_key} (freed {memory_freed / 1024 / 1024:.1f}MB)")
    
    def _generate_cache_key(self, img_path: str, size: int, square_ok: bool = False, 
                           patch_size: int = 16) -> str:
        """
        Generate a unique cache key for an image with processing parameters.
        
        Args:
            img_path: Path to the image file
            size: Target size for resizing
            square_ok: Whether square images are acceptable
            patch_size: Patch size for alignment
            
        Returns:
            Unique cache key string
        """
        # Include file modification time to detect changes
        try:
            mtime = os.path.getmtime(img_path)
        except OSError:
            mtime = 0
        
        return f"{img_path}:{size}:{square_ok}:{patch_size}:{mtime}"
    
    def get_cached_image(self, img_path: str, size: int, square_ok: bool = False, 
                        patch_size: int = 16) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached image if available.
        
        Args:
            img_path: Path to the image file
            size: Target size for resizing
            square_ok: Whether square images are acceptable
            patch_size: Patch size for alignment
            
        Returns:
            Cached image data or None if not found
        """
        cache_key = self._generate_cache_key(img_path, size, square_ok, patch_size)
        
        with self.lock:
            if cache_key in self.cache:
                # Move to end (mark as recently used)
                img_data = self.cache.pop(cache_key)
                self.cache[cache_key] = img_data
                self.cache_hits += 1
                return img_data
            else:
                self.cache_misses += 1
                return None
    
    def cache_image(self, img_path: str, img_data: Dict[str, Any], size: int, 
                   square_ok: bool = False, patch_size: int = 16) -> None:
        """
        Cache a processed image.
        
        Args:
            img_path: Path to the image file
            img_data: Processed image data
            size: Target size for resizing
            square_ok: Whether square images are acceptable
            patch_size: Patch size for alignment
        """
        cache_key = self._generate_cache_key(img_path, size, square_ok, patch_size)
        memory_needed = self._estimate_memory_usage(img_data)
        
        with self.lock:
            # Evict old items if necessary
            self._evict_lru(memory_needed)
            
            # Add to cache
            self.cache[cache_key] = img_data
            self.current_memory_bytes += memory_needed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                'cached_images': len(self.cache),
                'memory_usage_mb': self.current_memory_bytes / 1024 / 1024,
                'memory_limit_mb': self.max_memory_bytes / 1024 / 1024,
                'memory_usage_percent': (self.current_memory_bytes / self.max_memory_bytes) * 100,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate
            }
    
    def clear(self) -> None:
        """Clear all cached images."""
        with self.lock:
            self.cache.clear()
            self.current_memory_bytes = 0
            print("Cleared image cache")


# Global cache instance
_global_cache: Optional[ImageCache] = None


def initialize_cache(max_memory_gb: float = 32.0) -> ImageCache:
    """
    Initialize the global image cache.
    
    Args:
        max_memory_gb: Maximum memory usage in GB
        
    Returns:
        The initialized cache instance
    """
    global _global_cache
    _global_cache = ImageCache(max_memory_gb)
    return _global_cache


def get_cache() -> ImageCache:
    """
    Get the global image cache, initializing it if necessary.
    
    Returns:
        The global cache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ImageCache()
    return _global_cache


def _process_single_image(img_path: str, size: int, square_ok: bool = False, 
                         patch_size: int = 16, verbose: bool = True) -> Dict[str, Any]:
    """
    Process a single image with resizing and normalization.
    
    Args:
        img_path: Path to the image file
        size: Target size for resizing
        square_ok: Whether square images are acceptable
        patch_size: Patch size for alignment
        verbose: Whether to print processing information
        
    Returns:
        Dictionary containing processed image data
    """
    cache = get_cache()
    
    # Check cache first
    cached_result = cache.get_cached_image(img_path, size, square_ok, patch_size)
    if cached_result is not None:
        if verbose:
            print(f' - loaded {img_path} from cache')
        return cached_result
    
    # Load and process image
    try:
        img = exif_transpose(PIL_Image.open(img_path)).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to load image {img_path}: {e}")
    
    W1, H1 = img.size
    
    # Resize logic (same as original load_images)
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    
    W, H = img.size
    cx, cy = W//2, H//2
    
    # Crop logic (same as original load_images)
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    
    W2, H2 = img.size
    
    # Create result dictionary (same format as original load_images)
    result = {
        'img': ImgNorm(img)[None],
        'true_shape': np.int32([img.size[::-1]]),
        'idx': 0,  # Will be set by caller
        'instance': '0'  # Will be set by caller
    }
    
    if verbose:
        print(f' - processed {img_path} with resolution {W1}x{H1} --> {W2}x{H2}')
    
    # Cache the result
    cache.cache_image(img_path, result, size, square_ok, patch_size)
    
    return result


def load_images(folder_or_list: Union[str, List[str]], size: int, square_ok: bool = False, 
               verbose: bool = True, patch_size: int = 16, max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load and process images with caching support and parallel processing.
    
    This function maintains the same interface as dust3r.utils.image.load_images
    but adds intelligent caching and parallel processing to avoid redundant 
    loading and speed up processing.
    
    Args:
        folder_or_list: Path to folder or list of image paths
        size: Target size for resizing (224 or 512)
        square_ok: Whether square images are acceptable
        verbose: Whether to print loading information
        patch_size: Patch size for alignment (default: 16)
        max_workers: Maximum number of worker threads (default: min(32, cpu_count + 4))
        
    Returns:
        List of dictionaries containing processed image data
    """
    start_time = time.time()
    
    # Handle folder vs list input (same as original)
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))
        image_paths = [os.path.join(root, path) for path in folder_content]
    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        image_paths = folder_or_list
    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')
    
    # Filter supported image extensions
    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)
    
    valid_paths = [path for path in image_paths 
                   if path.lower().endswith(supported_images_extensions)]
    
    if not valid_paths:
        raise RuntimeError(f'No supported images found in {folder_or_list}')
    
    # Process all images in parallel
    def process_image_with_index(args):
        """Helper function to process image with its index."""
        i, img_path = args
        try:
            img_data = _process_single_image(img_path, size, square_ok, patch_size, verbose=False)
            # Set correct index and instance
            img_data['idx'] = i
            img_data['instance'] = str(i)
            return i, img_data, None  # (index, data, error)
        except Exception as e:
            return i, None, str(e)  # (index, data, error)
    
    # Use ThreadPoolExecutor for parallel processing
    imgs = [None] * len(valid_paths)  # Pre-allocate to maintain order
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_image_with_index, (i, img_path)): i 
            for i, img_path in enumerate(valid_paths)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            try:
                i, img_data, error = future.result()
                if error is None:
                    imgs[i] = img_data
                    if verbose:
                        print(f' - processed {valid_paths[i]} [{i+1}/{len(valid_paths)}]')
                else:
                    failed_count += 1
                    if verbose:
                        print(f'Warning: Failed to process {valid_paths[i]}: {error}')
            except Exception as e:
                failed_count += 1
                if verbose:
                    print(f'Warning: Unexpected error processing image: {e}')
    
    # Filter out None values (failed images) and maintain order
    imgs = [img for img in imgs if img is not None]
    
    if not imgs:
        raise RuntimeError(f'No images could be processed from {folder_or_list}')
    
    # Print statistics
    if verbose:
        elapsed_time = time.time() - start_time
        cache_stats = get_cache().get_stats()
        print(f' (Processed {len(imgs)} images in {elapsed_time:.2f}s, '
              f'{failed_count} failed, cache hit rate: {cache_stats["hit_rate"]:.1%})')
    
    return imgs


def print_cache_stats() -> None:
    """Print detailed cache statistics."""
    stats = get_cache().get_stats()
    print(f"""
Image Cache Statistics:
  Cached images: {stats['cached_images']}
  Memory usage: {stats['memory_usage_mb']:.1f}MB / {stats['memory_limit_mb']:.1f}MB ({stats['memory_usage_percent']:.1f}%)
  Cache hits: {stats['cache_hits']}
  Cache misses: {stats['cache_misses']}
  Hit rate: {stats['hit_rate']:.1%}
""")


def clear_cache() -> None:
    """Clear the image cache."""
    get_cache().clear()


def load_images_parallel(folder_or_list: Union[str, List[str]], size: int = 512, 
                        square_ok: bool = False, verbose: bool = True, 
                        patch_size: int = 16, max_workers: Optional[int] = None,
                        benchmark: bool = False) -> List[Dict[str, Any]]:
    """
    Convenient wrapper for load_images with parallel processing and benchmarking.
    
    This function provides an easy way to load images with optimal parallel settings
    and optionally compare performance against sequential processing.
    
    Args:
        folder_or_list: Path to folder or list of image paths
        size: Target size for resizing (224 or 512)
        square_ok: Whether square images are acceptable
        verbose: Whether to print loading information
        patch_size: Patch size for alignment (default: 16)
        max_workers: Maximum number of worker threads (None for auto)
        benchmark: If True, also run sequential version for comparison
        
    Returns:
        List of dictionaries containing processed image data
    """
    import os
    
    # Auto-configure max_workers if not specified
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    if verbose:
        print(f"Using {max_workers} worker threads for parallel processing")
    
    # Load images with parallel processing
    start_time = time.time()
    imgs = load_images(folder_or_list, size, square_ok, verbose, patch_size, max_workers)
    parallel_time = time.time() - start_time
    
    if benchmark and len(imgs) > 1:
        # Clear cache and run sequential version for comparison
        if verbose:
            print("\n--- Benchmark: Running sequential version for comparison ---")
        clear_cache()
        
        start_time = time.time()
        imgs_seq = load_images(folder_or_list, size, square_ok, verbose, patch_size, max_workers=1)
        sequential_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        
        if verbose:
            print(f"\n--- Performance Comparison ---")
            print(f"Sequential time: {sequential_time:.2f}s")
            print(f"Parallel time:   {parallel_time:.2f}s") 
            print(f"Speedup:         {speedup:.2f}x")
            print(f"Workers used:    {max_workers}")
    
    return imgs 