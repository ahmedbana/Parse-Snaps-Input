#!/usr/bin/env python3
"""
Debug test script for the ParseInputNode
"""

import json
import sys
import os

# Add the current directory to the path so we can import our node
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parse_input_node import ParseInputNode

def debug_test():
    """Debug the ParseInputNode"""
    
    # Create the node instance
    node = ParseInputNode()
    
    # Test JSON input
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
    
    print("Debug Test:")
    print(f"JSON input: {json_input}")
    print(f"JSON length: {len(json_input)}")
    print()
    
    try:
        # Parse JSON manually first
        data = json.loads(json_input)
        print(f"JSON parsed successfully")
        print(f"Scenes count: {len(data.get('scenes', []))}")
        print(f"Face URL: {data.get('face_url')}")
        print()
        
        # Call the parse_input method
        print("Calling parse_input...")
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
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_test()
    if success:
        print("🎉 Debug test completed!")
    else:
        print("💥 Debug test failed!")
        sys.exit(1) 