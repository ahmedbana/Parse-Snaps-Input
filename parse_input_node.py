import json
import requests
import torch
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Tuple, Optional
import os
import asyncio
import aiohttp
from functools import lru_cache
import hashlib
import concurrent.futures

class ParseInputNode:
    """
    Custom ComfyUI node that parses JSON input and returns:
    - Segmented Scenes (IMAGES): Load all scene images from URLs
    - Segmented Masks (MASKS): Load all mask images from URLs and convert to masks
    - Total Scenes (INTEGER): Count of scenes
    - Face Image (SINGLE IMAGE): Load face image from URL
    - Generation ID (STRING): Generation identifier
    - Kid Name (STRING): Kid's name
    - Demo Text (STRING): Demo text
    """
    
    def __init__(self):
        self.output_dir = "output"
        self.type = "output"
        self.output_node = True
        self._session = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("segmented_scenes", "segmented_masks", "total_scenes", "face_image", "generation_id", "kid_name", "demo_text")
    FUNCTION = "parse_input"
    CATEGORY = "Snaps"
    
    @lru_cache(maxsize=100)
    def _load_image_cached(self, url_hash: str, url: str) -> Optional[np.ndarray]:
        """Cached version of image loading to avoid re-downloading same images"""
        try:
            response = requests.get(url, timeout=10)  # Reduced timeout
            response.raise_for_status()
            
            # Load image from bytes
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Normalize to 0-1 range
            if img_array.dtype == np.uint8:
                img_array = img_array.astype(np.float32) / 255.0
                
            return img_array
            
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            return None
    
    def load_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """Load image from URL and convert to numpy array with caching"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self._load_image_cached(url_hash, url)
    
    @lru_cache(maxsize=100)
    def _load_mask_cached(self, url_hash: str, url: str) -> Optional[np.ndarray]:
        """Cached version of mask loading to avoid re-downloading same masks"""
        try:
            response = requests.get(url, timeout=10)  # Reduced timeout
            response.raise_for_status()
            
            # Load image from bytes
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy array
            mask_array = np.array(image)
            
            # Normalize to 0-1 range
            if mask_array.dtype == np.uint8:
                mask_array = mask_array.astype(np.float32) / 255.0
                
            # Convert to binary mask (threshold at 0.5)
            mask_array = (mask_array > 0.5).astype(np.float32)
                
            return mask_array
            
        except Exception as e:
            print(f"Error loading mask from {url}: {e}")
            return None
    
    def load_mask_from_url(self, url: str) -> Optional[np.ndarray]:
        """Load mask from URL and convert to binary mask with caching"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self._load_mask_cached(url_hash, url)
    
    async def _download_image_async(self, session: aiohttp.ClientSession, url: str) -> Optional[np.ndarray]:
        """Download image asynchronously"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Load image from bytes
                    image = Image.open(io.BytesIO(content))
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Convert to numpy array
                    img_array = np.array(image)
                    
                    # Normalize to 0-1 range
                    if img_array.dtype == np.uint8:
                        img_array = img_array.astype(np.float32) / 255.0
                        
                    return img_array
                else:
                    print(f"Error loading image from {url}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            return None
    
    async def _download_mask_async(self, session: aiohttp.ClientSession, url: str) -> Optional[np.ndarray]:
        """Download mask asynchronously"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Load image from bytes
                    image = Image.open(io.BytesIO(content))
                    
                    # Convert to grayscale
                    if image.mode != 'L':
                        image = image.convert('L')
                    
                    # Convert to numpy array
                    mask_array = np.array(image)
                    
                    # Normalize to 0-1 range
                    if mask_array.dtype == np.uint8:
                        mask_array = mask_array.astype(np.float32) / 255.0
                        
                    # Convert to binary mask (threshold at 0.5)
                    mask_array = (mask_array > 0.5).astype(np.float32)
                        
                    return mask_array
                else:
                    print(f"Error loading mask from {url}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            print(f"Error loading mask from {url}: {e}")
            return None
    
    async def _download_all_async(self, data: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
        """Download all images and masks asynchronously while maintaining scene order"""
        scenes = data.get("scenes", [])
        scenes.sort(key=lambda x: int(x.get("scene_order", 0)))
        
        # Create tasks for parallel download with scene order tracking
        scene_tasks = []
        mask_tasks = []
        face_url = data.get("face_url")
        
        # Create scene image tasks with order tracking
        for i, scene in enumerate(scenes):
            scene_url = scene.get("scene_url")
            if scene_url:
                scene_tasks.append((i, self._download_image_async(self._session, scene_url)))
            else:
                scene_tasks.append((i, None))
        
        # Create mask image tasks with order tracking
        for i, scene in enumerate(scenes):
            mask_url = scene.get("mask_url")
            if mask_url:
                mask_tasks.append((i, self._download_mask_async(self._session, mask_url)))
            else:
                mask_tasks.append((i, None))
        
        # Create face image task
        face_task = self._download_image_async(self._session, face_url) if face_url else None
        
        # Execute all downloads in parallel
        all_tasks = []
        task_types = []  # Track what each task is
        
        # Add scene tasks
        for order, task in scene_tasks:
            if task is not None:
                all_tasks.append(task)
                task_types.append(('scene', order))
            else:
                task_types.append(('scene', order))
        
        # Add mask tasks
        for order, task in mask_tasks:
            if task is not None:
                all_tasks.append(task)
                task_types.append(('mask', order))
            else:
                task_types.append(('mask', order))
        
        # Add face task
        if face_task is not None:
            all_tasks.append(face_task)
            task_types.append(('face', -1))
        
        # Execute all downloads in parallel
        if all_tasks:
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
        else:
            results = []
        
        # Reconstruct results in proper order
        scene_images = [None] * len(scenes)
        mask_images = [None] * len(scenes)
        face_image = None
        
        result_idx = 0
        for task_type, order in task_types:
            if task_type == 'scene':
                if order < len(scenes):
                    if result_idx < len(results):
                        result = results[result_idx]
                        if result is not None and not isinstance(result, Exception):
                            scene_images[order] = result
                        result_idx += 1
            elif task_type == 'mask':
                if order < len(scenes):
                    if result_idx < len(results):
                        result = results[result_idx]
                        if result is not None and not isinstance(result, Exception):
                            mask_images[order] = result
                        result_idx += 1
            elif task_type == 'face':
                if result_idx < len(results):
                    result = results[result_idx]
                    if result is not None and not isinstance(result, Exception):
                        face_image = result
                    result_idx += 1
        
        # Filter out None values and maintain order
        scene_images = [img for img in scene_images if img is not None]
        mask_images = [mask for mask in mask_images if mask is not None]
        
        return scene_images, mask_images, face_image
    
    def parse_input(self, json_input: str) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, str, str, str]:
        """Parse JSON input and return the specified outputs with optimized parallel downloads"""
        
        try:
            # Parse JSON input
            data = json.loads(json_input)
            
            # Extract metadata first (fast)
            generation_id = data.get("generation_id", "")
            kid_name = data.get("kid_name", "")
            demo_text = data.get("demo_text", "")
            scenes = data.get("scenes", [])
            total_scenes = len(scenes)
            
            # Sort scenes by scene_order
            scenes.sort(key=lambda x: int(x.get("scene_order", 0)))
            
            # Try async download first, fallback to sync if aiohttp not available
            try:
                # Initialize session for async downloads
                if self._session is None:
                    self._session = aiohttp.ClientSession()
                
                # Download all images asynchronously
                scene_images, mask_images, face_image = asyncio.run(self._download_all_async(data))
                    
            except Exception as e:
                # Fallback to synchronous downloads if async fails
                scene_images = []
                for i, scene in enumerate(scenes):
                    scene_url = scene.get("scene_url")
                    if scene_url:
                        img_array = self.load_image_from_url(scene_url)
                        if img_array is not None:
                            scene_images.append(img_array)
                
                # Load mask images in order
                mask_images = []
                for i, scene in enumerate(scenes):
                    mask_url = scene.get("mask_url")
                    if mask_url:
                        mask_array = self.load_mask_from_url(mask_url)
                        if mask_array is not None:
                            mask_images.append(mask_array)
                
                # Load face image
                face_image = None
                face_url = data.get("face_url")
                if face_url:
                    face_image = self.load_image_from_url(face_url)
            
            # Handle missing face image
            if face_image is None:
                face_image = np.zeros((512, 512, 3), dtype=np.float32)
            
            # Convert lists to numpy arrays for ComfyUI
            if scene_images:
                segmented_scenes = np.stack(scene_images, axis=0)
            else:
                # Create default empty scene
                segmented_scenes = np.zeros((1, 512, 512, 3), dtype=np.float32)
            
            if mask_images:
                segmented_masks = np.stack(mask_images, axis=0)
            else:
                # Create default empty mask
                segmented_masks = np.zeros((1, 512, 512), dtype=np.float32)
            
            # Ensure face_image is 3D (add batch dimension if needed)
            if len(face_image.shape) == 3:
                face_image = np.expand_dims(face_image, axis=0)
            
            # Convert numpy arrays to PyTorch tensors for ComfyUI
            segmented_scenes_tensor = torch.from_numpy(segmented_scenes)
            segmented_masks_tensor = torch.from_numpy(segmented_masks)
            face_image_tensor = torch.from_numpy(face_image)
            
            return (
                segmented_scenes_tensor,  # IMAGE
                segmented_masks_tensor,   # MASK
                total_scenes,             # INT
                face_image_tensor,        # IMAGE
                generation_id,            # STRING
                kid_name,                 # STRING
                demo_text                 # STRING
            )
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            # Return default values on error
            default_scenes = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            default_masks = torch.zeros((1, 512, 512), dtype=torch.float32)
            default_face = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            
            return (
                default_scenes,           # Empty scene
                default_masks,            # Empty mask
                0,                        # No scenes
                default_face,             # Empty face
                "",                       # Empty generation_id
                "",                       # Empty kid_name
                ""                        # Empty demo_text
            )
        except Exception as e:
            print(f"Unexpected error: {e}")
            # Return default values on error
            default_scenes = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            default_masks = torch.zeros((1, 512, 512), dtype=torch.float32)
            default_face = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            
            return (
                default_scenes,           # Empty scene
                default_masks,            # Empty mask
                0,                        # No scenes
                default_face,             # Empty face
                "",                       # Empty generation_id
                "",                       # Empty kid_name
                ""                        # Empty demo_text
            ) 