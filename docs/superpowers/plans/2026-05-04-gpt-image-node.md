# GPT-Image-2 Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ComfyUI custom node that generates images via OpenAI's GPT-Image-2 API from text prompts.

**Architecture:** Single-file node implementation (`nodes.py`) with a registration module (`__init__.py`). The node accepts text input (with optional external text concatenation), calls the OpenAI images API, decodes base64 response into a ComfyUI-compatible IMAGE tensor.

**Tech Stack:** Python, OpenAI SDK, Pillow, PyTorch, NumPy

---

## File Structure

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Declare pip dependencies (openai, Pillow) |
| `nodes.py` | `GPTImageGenerate` class — input spec, text concat, API call, image decode, tensor output |
| `__init__.py` | Register node with ComfyUI via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` |
| `tests/test_nodes.py` | Unit tests for text concatenation logic and image decoding |

---

### Task 1: Project Setup — requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```txt
openai>=1.0.0
Pillow>=9.0.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "feat: add requirements.txt with openai and Pillow dependencies"
```

---

### Task 2: Node Implementation — Text Concatenation and Input Definition

**Files:**
- Create: `nodes.py`
- Create: `tests/test_nodes.py`

- [ ] **Step 1: Write failing tests for text concatenation**

```python
# tests/test_nodes.py
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import GPTImageGenerate


class TestBuildPrompt:
    def setup_method(self):
        self.node = GPTImageGenerate()

    def test_both_inputs(self):
        result = self.node.build_prompt("external design req", "style: minimalist")
        assert result == "external design req\nstyle: minimalist"

    def test_external_only(self):
        result = self.node.build_prompt("external design req", "")
        assert result == "external design req"

    def test_prompt_only(self):
        result = self.node.build_prompt("", "local prompt text")
        assert result == "local prompt text"

    def test_both_empty_raises(self):
        with pytest.raises(ValueError, match="at least one text input"):
            self.node.build_prompt("", "")

    def test_none_external(self):
        result = self.node.build_prompt(None, "local prompt")
        assert result == "local prompt"

    def test_whitespace_only_treated_as_empty(self):
        with pytest.raises(ValueError, match="at least one text input"):
            self.node.build_prompt("   ", "   ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mars/Projects/style-manager-comfy-ui && python -m pytest tests/test_nodes.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError` (nodes.py doesn't exist yet)

- [ ] **Step 3: Write the node class with input spec and build_prompt**

```python
# nodes.py
import base64
import io

import numpy as np
import torch
from PIL import Image
from openai import OpenAI


class GPTImageGenerate:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Style Manager/GPT Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "size": (["1024x1024", "1536x1024", "1024x1536", "auto"],),
                "quality": (["auto", "low", "medium", "high"],),
                "background": (["auto", "transparent", "opaque"],),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["png", "webp"],),
            },
            "optional": {
                "external_text": ("STRING", {"forceInput": True}),
            },
        }

    def build_prompt(self, external_text, prompt):
        ext = (external_text or "").strip()
        loc = (prompt or "").strip()

        if ext and loc:
            return ext + "\n" + loc
        elif ext:
            return ext
        elif loc:
            return loc
        else:
            raise ValueError("Prompt is empty: at least one text input must be provided")

    def generate(self, api_key, prompt, size, quality, background, n, output_format, external_text=None):
        final_prompt = self.build_prompt(external_text, prompt)

        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model="gpt-image-1",
            prompt=final_prompt,
            size=size,
            quality=quality,
            background=background,
            n=n,
            output_format=output_format,
        )

        images = []
        for item in response.data:
            image_bytes = base64.b64decode(item.b64_json)
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = pil_image.convert("RGBA") if background == "transparent" else pil_image.convert("RGB")
            np_array = np.array(pil_image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np_array)
            images.append(tensor)

        batch = torch.stack(images, dim=0)
        return (batch,)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mars/Projects/style-manager-comfy-ui && python -m pytest tests/test_nodes.py::TestBuildPrompt -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add nodes.py tests/test_nodes.py
git commit -m "feat: implement GPTImageGenerate node with text concatenation and API call"
```

---

### Task 3: Image Decoding Tests

**Files:**
- Modify: `tests/test_nodes.py`

- [ ] **Step 1: Write tests for image decoding logic**

Add to `tests/test_nodes.py`:

```python
from unittest.mock import patch, MagicMock


class TestImageDecode:
    def setup_method(self):
        self.node = GPTImageGenerate()

    def _make_fake_response(self, width=64, height=64, n=1, mode="RGB"):
        """Create a fake OpenAI API response with base64-encoded images."""
        images_data = []
        for i in range(n):
            img = Image.new(mode, (width, height), color=(255, 0, 0) if i == 0 else (0, 255, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            mock_item = MagicMock()
            mock_item.b64_json = b64
            images_data.append(mock_item)

        mock_response = MagicMock()
        mock_response.data = images_data
        return mock_response

    @patch("nodes.OpenAI")
    def test_single_image_output_shape(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.images.generate.return_value = self._make_fake_response(64, 64, n=1)

        result = self.node.generate(
            api_key="test-key", prompt="test", size="1024x1024",
            quality="auto", background="opaque", n=1, output_format="png"
        )

        assert result[0].shape == (1, 64, 64, 3)
        assert result[0].dtype == torch.float32
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    @patch("nodes.OpenAI")
    def test_batch_output_shape(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.images.generate.return_value = self._make_fake_response(64, 64, n=3)

        result = self.node.generate(
            api_key="test-key", prompt="test", size="1024x1024",
            quality="auto", background="opaque", n=3, output_format="png"
        )

        assert result[0].shape == (3, 64, 64, 3)

    @patch("nodes.OpenAI")
    def test_transparent_background_rgba(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.images.generate.return_value = self._make_fake_response(64, 64, n=1, mode="RGBA")

        result = self.node.generate(
            api_key="test-key", prompt="test", size="1024x1024",
            quality="auto", background="transparent", n=1, output_format="png"
        )

        assert result[0].shape == (1, 64, 64, 4)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/mars/Projects/style-manager-comfy-ui && python -m pytest tests/test_nodes.py -v`
Expected: All 9 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_nodes.py
git commit -m "test: add image decoding and batch output tests"
```

---

### Task 4: Node Registration — __init__.py

**Files:**
- Create: `__init__.py`

- [ ] **Step 1: Create __init__.py**

```python
# __init__.py
from .nodes import GPTImageGenerate

NODE_CLASS_MAPPINGS = {
    "GPTImageGenerate": GPTImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageGenerate": "GPT Image Generate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/mars/Projects/style-manager-comfy-ui && python -c "from __init__ import NODE_CLASS_MAPPINGS; print(NODE_CLASS_MAPPINGS)"`
Expected: `{'GPTImageGenerate': <class 'nodes.GPTImageGenerate'>}`

- [ ] **Step 3: Commit**

```bash
git add __init__.py
git commit -m "feat: add __init__.py with ComfyUI node registration"
```

---

### Task 5: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/mars/Projects/style-manager-comfy-ui && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify file structure is correct**

Run: `ls -la /Users/mars/Projects/style-manager-comfy-ui/*.py /Users/mars/Projects/style-manager-comfy-ui/requirements.txt /Users/mars/Projects/style-manager-comfy-ui/tests/`
Expected: `__init__.py`, `nodes.py`, `requirements.txt`, `tests/test_nodes.py` all present
