import json
import requests
import torch
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Tuple, Optional
import os

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
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("segmented_scenes", "segmented_masks", "total_scenes", "face_image", "generation_id", "kid_name", "demo_text")
    FUNCTION = "parse_input"
    CATEGORY = "Snaps"
    
    def load_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """Load image from URL and convert to numpy array"""
        try:
            response = requests.get(url, timeout=30)
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
    
    def load_mask_from_url(self, url: str) -> Optional[np.ndarray]:
        """Load mask from URL and convert to binary mask"""
        try:
            response = requests.get(url, timeout=30)
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
    
    def parse_input(self, json_input: str) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, str, str, str]:
        """Parse JSON input and return the specified outputs"""
        
        try:
            # Parse JSON input
            data = json.loads(json_input)
            
            # Extract scenes and sort by scene_order
            scenes = data.get("scenes", [])
            scenes.sort(key=lambda x: int(x.get("scene_order", 0)))
            
            # Load scene images
            scene_images = []
            for scene in scenes:
                scene_url = scene.get("scene_url")
                if scene_url:
                    img_array = self.load_image_from_url(scene_url)
                    if img_array is not None:
                        scene_images.append(img_array)
            
            # Load mask images
            mask_images = []
            for scene in scenes:
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
                if face_image is None:
                    # Create a default face image (black image)
                    face_image = np.zeros((512, 512, 3), dtype=np.float32)
            else:
                # Create a default face image (black image)
                face_image = np.zeros((512, 512, 3), dtype=np.float32)
            
            # Extract metadata
            generation_id = data.get("generation_id", "")
            kid_name = data.get("kid_name", "")
            demo_text = data.get("demo_text", "")
            total_scenes = len(scenes)
            
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