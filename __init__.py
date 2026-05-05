try:
    from .nodes import GPTImageGenerate, StyleManagerQuery
except ImportError:
    from nodes import GPTImageGenerate, StyleManagerQuery

NODE_CLASS_MAPPINGS = {
    "GPTImageGenerate": GPTImageGenerate,
    "StyleManagerQuery": StyleManagerQuery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageGenerate": "GPT Image Generate",
    "StyleManagerQuery": "Style Manager Query",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
