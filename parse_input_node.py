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
                "json_input": ("STRING", {"default": "", "multiline": True}),
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
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.content
                    
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
                    print(f"Error loading image from {url}: HTTP {response.status_code} (attempt {attempt + 1}/{max_retries})")
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
    
    def parse_input(self, json_input: str) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, str, str, str, str]:
        """Parse JSON input and return the specified outputs with async downloads"""
        
        try:
            # Parse JSON input
            data = json.loads(json_input)
            
            # Extract metadata first (fast)
            generation_id = data.get("generation_id", "")
            kid_name = data.get("kid_name", "")
            demo_text = data.get("demo_text", "")
            scenes = data.get("scenes", [])
            total_scenes = len(scenes)
            
            # Extract scene orders (excluding 0)
            scene_orders = []
            for scene in scenes:
                scene_order = scene.get("scene_order", 0)
                # Convert to int and exclude 0
                try:
                    order_int = int(scene_order)
                    if order_int != 0:
                        scene_orders.append(str(order_int))
                except (ValueError, TypeError):
                    # If scene_order is not a valid number, skip it
                    continue
            
            # Convert to comma-separated string
            scene_orders_string = ",".join(scene_orders)
            
            # Sort scenes by scene_order
            scenes.sort(key=lambda x: int(x.get("scene_order", 0)))
            
            # Download all images synchronously
            scene_images, mask_images, face_image, has_loading_error = self._download_all_sync(data)
            
            # If there was a loading error, return error state
            if has_loading_error:
                print("Loading error detected - returning error state")
                # Return default values on loading error
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
                demo_text,                # STRING
                scene_orders_string      # STRING
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
                "",                       # Empty demo_text
                ""                        # Empty scene_orders
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
                "",                       # Empty demo_text
                ""                        # Empty scene_orders
            ) 