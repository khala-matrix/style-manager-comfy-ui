try:
    from .nodes import GPTImageGenerate, StyleManagerBatchGenerate, StyleManagerQuery
except ImportError:
    from nodes import GPTImageGenerate, StyleManagerBatchGenerate, StyleManagerQuery

NODE_CLASS_MAPPINGS = {
    "GPTImageGenerate": GPTImageGenerate,
    "StyleManagerBatchGenerate": StyleManagerBatchGenerate,
    "StyleManagerQuery": StyleManagerQuery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageGenerate": "GPT Image Generate",
    "StyleManagerBatchGenerate": "Style Manager Batch Generate",
    "StyleManagerQuery": "Style Manager Query",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
