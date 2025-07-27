from .parse_input_node import ParseInputNode

NODE_CLASS_MAPPINGS = {
    "ParseInputNode": ParseInputNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParseInputNode": "Parse Input JSON"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 