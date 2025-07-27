# Snaps Parse Input Node

A custom ComfyUI node that parses JSON input and returns segmented scenes, masks, and metadata.

## Features

This node takes a JSON string as input and returns:

- **Segmented Scenes (IMAGES)**: Loads all scene images from URLs in the same order
- **Segmented Masks (MASKS)**: Loads all mask images from URLs and converts them to binary masks
- **Total Scenes (INTEGER)**: Count of scenes in the JSON
- **Face Image (SINGLE IMAGE)**: Loads the face image from URL
- **Generation ID (STRING)**: Generation identifier
- **Kid Name (STRING)**: Kid's name
- **Demo Text (STRING)**: Demo text

## Installation

1. Copy this folder to your ComfyUI `custom_nodes` directory
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI

## Usage

### Input JSON Format

The node expects a JSON string with the following structure:

```json
{
    "scenes": [
        {
            "scene_order": "0",
            "type": "cover",
            "scene_url": "https://example.com/scene1.png",
            "mask_url": "https://example.com/mask1.png"
        },
        {
            "scene_order": "1",
            "type": "scene",
            "scene_url": "https://example.com/scene2.png",
            "mask_url": "https://example.com/mask2.png"
        }
    ],
    "face_url": "https://example.com/face.jpg",
    "generation_id": "PRVSHPTST01",
    "kid_name": "OMAR",
    "demo_text": "MOHAMED"
}
```

### Example JSON Input

```json
{
    "scenes": [
        {
            "scene_order": "0",
            "type": "cover",
            "scene_url": "https://snapsai.blob.core.windows.net/stories/ToyStory/FinalOutput/Story/00.png",
            "mask_url": "https://snapsai.blob.core.windows.net/stories/ToyStory/FinalOutput/Masks/00.png"
        },
        {
            "scene_order": "1",
            "type": "scene",
            "scene_url": "https://snapsai.blob.core.windows.net/stories/ToyStory/FinalOutput/Story/01.png",
            "mask_url": "https://snapsai.blob.core.windows.net/stories/ToyStory/FinalOutput/Masks/01.png"
        },
        {
            "scene_order": "4",
            "type": "cover",
            "scene_url": "https://snapsai.blob.core.windows.net/stories/ToyStory/FinalOutput/Story/04.png",
            "mask_url": "https://snapsai.blob.core.windows.net/stories/ToyStory/FinalOutput/Masks/04.png"
        }
    ],
    "face_url": "https://snapsai.blob.core.windows.net/input/Faces/OMar123.JPG",
    "generation_id": "PRVSHPTST01",
    "kid_name": "OMAR",
    "demo_text": "MOHAMED"
}
```

### Node Outputs

1. **segmented_scenes** (IMAGE): Batch of scene images loaded from URLs
2. **segmented_masks** (MASK): Batch of binary masks converted from mask images
3. **total_scenes** (INT): Number of scenes in the JSON
4. **face_image** (IMAGE): Single face image loaded from URL
5. **generation_id** (STRING): Generation identifier from JSON
6. **kid_name** (STRING): Kid's name from JSON
7. **demo_text** (STRING): Demo text from JSON

## Error Handling

The node includes robust error handling:

- Invalid JSON: Returns default empty values
- Network errors: Returns default empty values for failed image loads
- Missing URLs: Creates default black images/masks
- Invalid image formats: Attempts to convert to appropriate format

## Dependencies

- requests: For downloading images from URLs
- Pillow: For image processing
- numpy: For array operations
- torch: For ComfyUI compatibility

## Category

The node appears in the "Snaps" category in ComfyUI. 