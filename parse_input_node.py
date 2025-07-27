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
    
    async def _download_image_async(self, session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """Download image asynchronously with retry logic"""
        for attempt in range(max_retries):
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
                        print(f"Error loading image from {url}: HTTP {response.status} (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                        
            except Exception as e:
                print(f"Error loading image from {url}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        print(f"Failed to load image from {url} after {max_retries} attempts")
        return None
    
    async def _download_mask_async(self, session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """Download mask asynchronously with retry logic"""
        for attempt in range(max_retries):
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
                        print(f"Error loading mask from {url}: HTTP {response.status} (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                        
            except Exception as e:
                print(f"Error loading mask from {url}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        print(f"Failed to load mask from {url} after {max_retries} attempts")
        return None
    
    async def _download_all_async(self, data: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray], bool]:
        """Download all images and masks asynchronously with retry logic while maintaining scene order"""
        scenes = data.get("scenes", [])
        scenes.sort(key=lambda x: int(x.get("scene_order", 0)))
        
        # Create a new session for this operation
        async with aiohttp.ClientSession() as session:
            # Initialize result arrays
            scene_images = [None] * len(scenes)
            mask_images = [None] * len(scenes)
            face_image = None
            has_loading_error = False
            
            # Download face image first (if exists)
            face_url = data.get("face_url")
            if face_url:
                face_image = await self._download_image_async(session, face_url)
                if face_image is None:
                    has_loading_error = True
                    print("Failed to load face image after retries")
            
            # Download all scene images and masks with retry logic
            max_retry_rounds = 3
            for retry_round in range(max_retry_rounds):
                print(f"Download round {retry_round + 1}/{max_retry_rounds}")
                
                # Create tasks for failed downloads only
                scene_tasks = []
                mask_tasks = []
                
                # Add scene image tasks for failed downloads
                for i, scene in enumerate(scenes):
                    if scene_images[i] is None:  # Only retry failed downloads
                        scene_url = scene.get("scene_url")
                        if scene_url:
                            scene_tasks.append((i, self._download_image_async(session, scene_url)))
                
                # Add mask image tasks for failed downloads
                for i, scene in enumerate(scenes):
                    if mask_images[i] is None:  # Only retry failed downloads
                        mask_url = scene.get("mask_url")
                        if mask_url:
                            mask_tasks.append((i, self._download_mask_async(session, mask_url)))
                
                # If no failed downloads to retry, break
                if not scene_tasks and not mask_tasks:
                    break
                
                # Execute retry downloads in parallel
                all_tasks = []
                task_types = []
                
                # Add scene tasks
                for order, task in scene_tasks:
                    all_tasks.append(task)
                    task_types.append(('scene', order))
                
                # Add mask tasks
                for order, task in mask_tasks:
                    all_tasks.append(task)
                    task_types.append(('mask', order))
                
                # Execute all retry downloads in parallel
                if all_tasks:
                    results = await asyncio.gather(*all_tasks, return_exceptions=True)
                    
                    # Process results
                    result_idx = 0
                    for task_type, order in task_types:
                        if result_idx < len(results):
                            result = results[result_idx]
                            if result is not None and not isinstance(result, Exception):
                                if task_type == 'scene':
                                    scene_images[order] = result
                                elif task_type == 'mask':
                                    mask_images[order] = result
                            result_idx += 1
                
                # If this is not the last round, wait before next retry
                if retry_round < max_retry_rounds - 1:
                    await asyncio.sleep(1)  # Brief pause between retry rounds
            
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
    
    async def parse_input(self, json_input: str) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, str, str, str, str]:
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
            
            # Extract scene orders
            scene_orders = []
            for scene in scenes:
                scene_order = scene.get("scene_order", 0)
                scene_orders.append({"scene_order": scene_order})
            
            # Convert scene_orders to JSON string
            scene_orders_json = json.dumps(scene_orders)
            
            # Sort scenes by scene_order
            scenes.sort(key=lambda x: int(x.get("scene_order", 0)))
            
            # Download all images asynchronously
            scene_images, mask_images, face_image, has_loading_error = await self._download_all_async(data)
            
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
                    "[]"                     # Empty scene_orders
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
                scene_orders_json        # STRING
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
                "[]"                     # Empty scene_orders
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
                "[]"                     # Empty scene_orders
            ) 