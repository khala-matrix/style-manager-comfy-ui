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
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
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

    def generate(self, api_key, base_url, prompt, size, quality, background, n, output_format, external_text=None):
        final_prompt = self.build_prompt(external_text, prompt)

        client = OpenAI(api_key=api_key, base_url=base_url)
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
            if background == "transparent":
                pil_image = pil_image.convert("RGBA")
            else:
                pil_image = pil_image.convert("RGB")
            np_array = np.array(pil_image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np_array)
            images.append(tensor)

        batch = torch.stack(images, dim=0)
        return (batch,)
