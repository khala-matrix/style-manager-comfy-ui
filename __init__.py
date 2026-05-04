from .nodes import GPTImageGenerate

NODE_CLASS_MAPPINGS = {
    "GPTImageGenerate": GPTImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageGenerate": "GPT Image Generate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
