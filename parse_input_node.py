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
import time

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
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Story_Name": ("STRING", {"default": ""}),
                "Face_Image": ("STRING", {"default": ""}),
                "Limit": ("INT", {"default": 10, "min": 1, "max": 100}),
                "Kid_Name": ("STRING", {"default": ""}),
                "Demo_Text": ("STRING", {"default": ""}),
                "Generation_ID": ("STRING", {"default": ""}),
                "Scene_Orders": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("segmented_scenes", "segmented_masks", "total_scenes", "face_image", "generation_id", "kid_name", "demo_text", "scene_orders")
    FUNCTION = "parse_input"
    CATEGORY = "Snaps"
    
    def _download_image_sync(self, url: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """Download image synchronously with retry logic"""
        for attempt in range(max_retries):
            try:
                # Use a session for better connection handling
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (compatible; ComfyUI-ParseInput/1.0)'
                })
                
                response = session.get(url, timeout=30, stream=True)
                if response.status_code == 200:
                    content = response.content
                    
                    # Load image from bytes with proper error handling
                    try:
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
                    except Exception as e:
                        print(f"Error processing image from {url}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                else:
                    print(f"Error loading image from {url}: HTTP {response.status_code} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"Timeout loading image from {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
            except requests.exceptions.ConnectionError:
                print(f"Connection error loading image from {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
            except Exception as e:
                print(f"Error loading image from {url}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        print(f"Failed to load image from {url} after {max_retries} attempts")
        return None
    
    def _download_mask_sync(self, url: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """Download mask synchronously with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.content
                    
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
                    print(f"Error loading mask from {url}: HTTP {response.status_code} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
            except Exception as e:
                print(f"Error loading mask from {url}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        print(f"Failed to load mask from {url} after {max_retries} attempts")
        return None
    
    def _download_all_sync(self, data: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray], bool]:
        """Download all images and masks synchronously with retry logic while maintaining scene order"""
        scenes = data.get("scenes", [])
        scenes.sort(key=lambda x: int(x.get("scene_order", 0)))
        
        # Initialize result arrays
        scene_images = [None] * len(scenes)
        mask_images = [None] * len(scenes)
        face_image = None
        has_loading_error = False
        
        # Download face image first (if exists)
        face_url = data.get("face_url")
        if face_url:
            face_image = self._download_image_sync(face_url)
            if face_image is None:
                has_loading_error = True
                print("Failed to load face image after retries")
        
        # Download all scene images and masks with retry logic
        max_retry_rounds = 3
        for retry_round in range(max_retry_rounds):
            print(f"Download round {retry_round + 1}/{max_retry_rounds}")
            
            # Download failed scene images
            for i, scene in enumerate(scenes):
                if scene_images[i] is None:  # Only retry failed downloads
                    scene_url = scene.get("scene_url")
                    if scene_url:
                        scene_images[i] = self._download_image_sync(scene_url)
            
            # Download failed mask images
            for i, scene in enumerate(scenes):
                if mask_images[i] is None:  # Only retry failed downloads
                    mask_url = scene.get("mask_url")
                    if mask_url:
                        mask_images[i] = self._download_mask_sync(mask_url)
            
            # Check if all downloads are complete
            all_complete = True
            for i in range(len(scenes)):
                if scene_images[i] is None or mask_images[i] is None:
                    all_complete = False
                    break
            
            if all_complete:
                break
            
            # If this is not the last round, wait before next retry
            if retry_round < max_retry_rounds - 1:
                time.sleep(1)  # Brief pause between retry rounds
        
        # Check for any remaining failures
        for i in range(len(scenes)):
            scene_img = scene_images[i]
            mask_img = mask_images[i]
            
            # Check if we have incomplete data
            if (scene_img is not None and mask_img is None) or (scene_img is None and mask_img is not None):
                has_loading_error = True
                print(f"Scene {i} has incomplete data after all retries - missing image or mask")
            
            # Check if both are missing (this is also an error)
            if scene_img is None and mask_img is None:
                scene_url = scenes[i].get("scene_url")
                mask_url = scenes[i].get("mask_url")
                if scene_url or mask_url:  # Only error if URLs were provided
                    has_loading_error = True
                    print(f"Scene {i} failed to load both image and mask after all retries")
        
        # Ensure both arrays have the same length and are properly aligned
        # Only include scenes where both image and mask are available
        aligned_scene_images = []
        aligned_mask_images = []
        
        for i in range(len(scenes)):
            scene_img = scene_images[i]
            mask_img = mask_images[i]
            
            # Only include if both scene and mask are available
            if scene_img is not None and mask_img is not None:
                aligned_scene_images.append(scene_img)
                aligned_mask_images.append(mask_img)
        
        return aligned_scene_images, aligned_mask_images, face_image, has_loading_error
    
    def parse_input(self, Story_Name: str, Face_Image: str, Limit: int, Kid_Name: str, Demo_Text: str, Generation_ID: str, Scene_Orders: str) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, str, str, str, str]:
        """Parse individual inputs and return the specified outputs"""
        
        try:
            # Construct the base path from Story_Name in ComfyUI input folder
            base_path = os.path.join("input", Story_Name)
            scenes_path = os.path.join(base_path, "Scenes")
            masks_path = os.path.join(base_path, "Masks")
            
            # Check if the directories exist with proper error handling
            try:
                if not os.path.exists(scenes_path):
                    print(f"Scenes directory not found: {scenes_path}")
                    return self._return_defaults()
                
                if not os.path.exists(masks_path):
                    print(f"Masks directory not found: {masks_path}")
                    return self._return_defaults()
            except (OSError, PermissionError) as e:
                print(f"Error checking directories: {e}")
                return self._return_defaults()
            
            # Load scene images and masks
            scene_images, mask_images = self._load_images_from_folders(scenes_path, masks_path, Limit)
            
            # Load face image from URL
            face_image = None
            if Face_Image:
                face_image = self._download_image_sync(Face_Image)
                if face_image is not None:
                    # Ensure face_image is 3D (add batch dimension if needed)
                    if len(face_image.shape) == 3:
                        face_image = np.expand_dims(face_image, axis=0)
                    face_image_tensor = torch.from_numpy(face_image)
                else:
                    print(f"Failed to load face image from URL: {Face_Image}")
                    face_image_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            else:
                face_image_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            
            # Use the input parameters directly
            total_scenes = len(scene_images) if scene_images else 0
            generation_id = Generation_ID
            kid_name = Kid_Name
            demo_text = Demo_Text
            scene_orders_string = Scene_Orders
            
            # Convert to tensors
            if scene_images:
                segmented_scenes = torch.from_numpy(np.stack(scene_images, axis=0))
            else:
                segmented_scenes = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            
            if mask_images:
                segmented_masks = torch.from_numpy(np.stack(mask_images, axis=0))
            else:
                segmented_masks = torch.zeros((1, 512, 512), dtype=torch.float32)
            
            return (
                segmented_scenes,         # IMAGE - loaded from Scenes folder
                segmented_masks,          # MASK - loaded from Masks folder
                total_scenes,             # INT - count of loaded scenes
                face_image_tensor,        # IMAGE - will be populated based on Face_Image
                generation_id,            # STRING
                kid_name,                 # STRING
                demo_text,                # STRING
                scene_orders_string       # STRING
            )
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            return self._return_defaults()
    
    def _load_images_from_folders(self, scenes_path: str, masks_path: str, limit: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load images from Scenes and Masks folders"""
        scene_images = []
        mask_images = []
        
        try:
            # Get list of files in both directories with natural sorting
            import re
            
            def natural_sort_key(text):
                """Convert a string into a list of string and number chunks.
                "z23a" -> ["z", 23, "a"]
                """
                return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]
            
            # Use case-insensitive file extension matching for cross-platform compatibility
            def is_image_file(filename):
                """Check if file is an image with case-insensitive extension"""
                lower_filename = filename.lower()
                return any(lower_filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'])
            
            # Get files with proper error handling
            try:
                scene_files = [f for f in os.listdir(scenes_path) if is_image_file(f)]
            except (OSError, PermissionError) as e:
                print(f"Error reading scenes directory {scenes_path}: {e}")
                scene_files = []
            
            try:
                mask_files = [f for f in os.listdir(masks_path) if is_image_file(f)]
            except (OSError, PermissionError) as e:
                print(f"Error reading masks directory {masks_path}: {e}")
                mask_files = []
            
            # Sort using natural sorting
            scene_files.sort(key=natural_sort_key)
            mask_files.sort(key=natural_sort_key)
            
            # Limit the number of files to process
            scene_files = scene_files[:limit]
            mask_files = mask_files[:limit]
            
            print(f"Found {len(scene_files)} scene files and {len(mask_files)} mask files")
            print(f"Scene files order: {scene_files}")
            print(f"Mask files order: {mask_files}")
            
            # Load scene images
            for scene_file in scene_files:
                scene_path = os.path.join(scenes_path, scene_file)
                try:
                    # Use with statement for proper file handling
                    with Image.open(scene_path) as image:
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        img_array = np.array(image)
                        if img_array.dtype == np.uint8:
                            img_array = img_array.astype(np.float32) / 255.0
                        
                        scene_images.append(img_array)
                        print(f"Loaded scene: {scene_file}")
                except Exception as e:
                    print(f"Error loading scene {scene_file}: {e}")
            
            # Load mask images
            for mask_file in mask_files:
                mask_path = os.path.join(masks_path, mask_file)
                try:
                    # Use with statement for proper file handling
                    with Image.open(mask_path) as image:
                        if image.mode != 'L':
                            image = image.convert('L')
                        
                        mask_array = np.array(image)
                        if mask_array.dtype == np.uint8:
                            mask_array = mask_array.astype(np.float32) / 255.0
                        
                        # Convert to binary mask (threshold at 0.5)
                        mask_array = (mask_array > 0.5).astype(np.float32)
                        
                        mask_images.append(mask_array)
                        print(f"Loaded mask: {mask_file}")
                except Exception as e:
                    print(f"Error loading mask {mask_file}: {e}")
            
        except Exception as e:
            print(f"Error loading images from folders: {e}")
        
        return scene_images, mask_images
    
    def _return_defaults(self) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, str, str, str, str]:
        """Return default values when loading fails"""
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
            "",                       # Empty demo_text
            ""                        # Empty scene_orders
        ) 