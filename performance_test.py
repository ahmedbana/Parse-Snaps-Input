#!/usr/bin/env python3
"""
Performance test script for the ParseInputNode
"""

import json
import time
import sys
import os

# Add the current directory to the path so we can import our node
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parse_input_node import ParseInputNode

def test_performance():
    """Test the performance of the ParseInputNode"""
    
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
    
    print("Testing ParseInputNode Performance...")
    print(f"Input JSON: {json_input[:100]}...")
    print()
    
    # Test multiple runs to get average performance
    times = []
    for i in range(3):
        print(f"Run {i+1}/3...")
        start_time = time.time()
        
        try:
            # Call the parse_input method
            result = node.parse_input(json_input)
            
            # Unpack the results
            segmented_scenes, segmented_images, total_scenes, face_image, generation_id, kid_name, demo_text = result
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            
            print(f"  ✅ Run {i+1} completed in {elapsed_time:.2f} seconds")
            print(f"  - Segmented Scenes shape: {segmented_scenes.shape}")
            print(f"  - Segmented Images shape: {segmented_images.shape}")
            print(f"  - Total Scenes: {total_scenes}")
            print(f"  - Face Image shape: {face_image.shape}")
            print()
            
        except Exception as e:
            print(f"❌ Error in run {i+1}: {e}")
            return False
    
    # Calculate average performance
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("Performance Results:")
    print(f"  - Average time: {avg_time:.2f} seconds")
    print(f"  - Best time: {min_time:.2f} seconds")
    print(f"  - Worst time: {max_time:.2f} seconds")
    print(f"  - Improvement: ~{(7.0/avg_time):.1f}x faster than original 7 seconds")
    
    return True

if __name__ == "__main__":
    success = test_performance()
    if success:
        print("🎉 Performance test completed!")
    else:
        print("💥 Performance test failed!")
        sys.exit(1)