import base64
import io
import json
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
        for output in output_urls:
            if not output:
                continue
            if output.startswith(("http://", "https://")):
                img_resp = requests.get(output, timeout=60)
                img_resp.raise_for_status()
                image_bytes = img_resp.content
            else:
                image_bytes = base64.b64decode(output)
            pil_image = Image.open(io.BytesIO(image_bytes))
            if background == "transparent":
                pil_image = pil_image.convert("RGBA")
            else:
                pil_image = pil_image.convert("RGB")
            np_array = np.array(pil_image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np_array)
            images.append(tensor)

        if not images:
            raise RuntimeError("No valid image URLs returned by the API")

        batch = torch.stack(images, dim=0)
        return (batch,)


class StyleManagerQuery:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "query"
    CATEGORY = "Style Manager/Query"

    STYLE_MANAGER_BASE = "http://sandbox.cyber-psychosis.net/sandbox/style-manager-service/api"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "openai_api_key": ("STRING", {"default": ""}),
                "openai_base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "external_text": ("STRING", {"forceInput": True}),
            },
        }

    def query(self, api_key, openai_api_key, openai_base_url, prompt, external_text=None):
        ext = (external_text or "").strip()
        loc = (prompt or "").strip()
        if ext and loc:
            user_input = ext + "\n" + loc
        elif ext:
            user_input = ext
        elif loc:
            user_input = loc
        else:
            raise ValueError("At least one text input must be provided")

        filters = self._get_filters(api_key)
        selected = self._select_filters(openai_api_key, openai_base_url, user_input, filters)
        result = self._query_prompts(api_key, selected)
        return (json.dumps(result, ensure_ascii=False),)

    def _get_filters(self, api_key):
        resp = requests.get(
            f"{self.STYLE_MANAGER_BASE}/v1/prompts/filters",
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _select_filters(self, openai_api_key, openai_base_url, user_input, filters):
        system_prompt = (
            "You are a filter selector. Given the user's text and the available filters, "
            "select the most relevant combination. You MUST only use values from the provided lists.\n\n"
            "Available filters:\n"
            f"- scenes: {json.dumps(filters.get('scenes', []), ensure_ascii=False)}\n"
            f"- tags: {json.dumps(filters.get('tags', []), ensure_ascii=False)}\n"
            f"- titles: {json.dumps(filters.get('titles', []), ensure_ascii=False)}\n\n"
            "Respond with ONLY a JSON object (no markdown, no explanation) in this format:\n"
            '{"scene": "<one scene or null>", "tags": "<comma-separated tags or null>", "q": "<search keyword or null>"}\n'
            "Rules:\n"
            "- scene must be one value from scenes list, or null\n"
            "- tags must be from the tags list, or null\n"
            "- q must be a substring from the titles list, or null\n"
            "- Select values most relevant to the user's input\n"
            "- Use null for fields that are not relevant"
        )

        client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)

    def _query_prompts(self, api_key, selected):
        params = {"page": 1, "page_size": 20}
        if selected.get("scene"):
            params["scene"] = selected["scene"]
        if selected.get("tags"):
            params["tags"] = selected["tags"]
        if selected.get("q"):
            params["q"] = selected["q"]

        resp = requests.get(
            f"{self.STYLE_MANAGER_BASE}/v1/prompts",
            headers={"X-API-Key": api_key},
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
