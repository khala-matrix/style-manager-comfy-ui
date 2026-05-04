import base64
import io
import time

import numpy as np
import requests
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
                "mode": (["openai_sdk", "gptsapi_async"],),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "size": (["1024x1024", "1536x1024", "1024x1536", "auto"],),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"],),
                "quality": (["auto", "low", "medium", "high"],),
                "background": (["auto", "transparent", "opaque"],),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["png", "webp"],),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 30}),
                "timeout": ("INT", {"default": 120, "min": 30, "max": 600}),
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

    def generate(self, api_key, base_url, mode, prompt, size, aspect_ratio, quality, background, n, output_format, poll_interval, timeout, external_text=None):
        final_prompt = self.build_prompt(external_text, prompt)

        if mode == "openai_sdk":
            return self._generate_openai(api_key, base_url, final_prompt, size, quality, background, n, output_format)
        else:
            return self._generate_gptsapi(api_key, base_url, final_prompt, aspect_ratio, quality, background, n, output_format, poll_interval, timeout)

    def _generate_openai(self, api_key, base_url, final_prompt, size, quality, background, n, output_format):
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

    def _generate_gptsapi(self, api_key, base_url, final_prompt, aspect_ratio, quality, background, n, output_format, poll_interval, timeout):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": final_prompt,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "background": background,
            "n": n,
            "output_format": output_format,
        }

        submit_url = f"{base_url.rstrip('/')}/api/v3/openai/gpt-image-2-plus/text-to-image"
        resp = requests.post(submit_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 200:
            raise RuntimeError(f"API error: {result.get('message', 'Unknown error')}")

        poll_url = result["data"]["urls"]["get"]

        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Image generation timed out after {timeout}s")

            time.sleep(poll_interval)

            poll_resp = requests.get(poll_url, headers=headers, timeout=30)
            poll_resp.raise_for_status()
            poll_result = poll_resp.json()

            status = poll_result["data"]["status"]
            if status == "completed":
                output_urls = poll_result["data"]["outputs"]
                break
            elif status in ("failed", "error"):
                error_msg = poll_result["data"].get("error") or "Generation failed"
                raise RuntimeError(f"Image generation failed: {error_msg}")

        images = []
        for url in output_urls:
            img_resp = requests.get(url, timeout=60)
            img_resp.raise_for_status()
            pil_image = Image.open(io.BytesIO(img_resp.content))
            if background == "transparent":
                pil_image = pil_image.convert("RGBA")
            else:
                pil_image = pil_image.convert("RGB")
            np_array = np.array(pil_image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np_array)
            images.append(tensor)

        batch = torch.stack(images, dim=0)
        return (batch,)
