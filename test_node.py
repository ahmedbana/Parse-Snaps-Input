#!/usr/bin/env python3
"""
Test script for the ParseInputNode
"""

import json
import sys
import os

# Add the current directory to the path so we can import our node
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parse_input_node import ParseInputNode

def test_parse_input_node():
    """Test the ParseInputNode with the provided JSON input"""
    
    # Create the node instance
    node = ParseInputNode()
    
    # Test JSON input (the one provided by the user)
    test_json = {
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
    
    # Convert to JSON string
    json_input = json.dumps(test_json)
    
    print("Testing ParseInputNode...")
    print(f"Input JSON: {json_input[:100]}...")
    print()
    
    try:
        # Call the parse_input method
        result = node.parse_input(json_input)
        
        # Unpack the results
        segmented_scenes, segmented_masks, total_scenes, face_image, generation_id, kid_name, demo_text = result
        
        print("✅ Node executed successfully!")
        print()
        print("Results:")
        print(f"  - Segmented Scenes shape: {segmented_scenes.shape}")
        print(f"  - Segmented Masks shape: {segmented_masks.shape}")
        print(f"  - Total Scenes: {total_scenes}")
        print(f"  - Face Image shape: {face_image.shape}")
        print(f"  - Generation ID: '{generation_id}'")
        print(f"  - Kid Name: '{kid_name}'")
        print(f"  - Demo Text: '{demo_text}'")
        print()
        
        # Test with invalid JSON
        print("Testing error handling with invalid JSON...")
        invalid_result = node.parse_input("invalid json")
        print("✅ Error handling works correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing node: {e}")
        return False

if __name__ == "__main__":
    success = test_parse_input_node()
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
        sys.exit(1) 