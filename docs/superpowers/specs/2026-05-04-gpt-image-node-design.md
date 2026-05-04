# GPT-Image-2 ComfyUI Node Design

## Overview

A ComfyUI custom node that calls OpenAI's GPT-Image-2 model to generate images from long text prompts (e.g., design requirements).

## Node Info

- **Display Name**: GPT Image Generate
- **Category**: `Style Manager/GPT Image`
- **Class Name**: `GPTImageGenerate`

## Inputs

### Required

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `api_key` | STRING | (required) | — | OpenAI API Key |
| `prompt` | STRING (multiline) | `""` | — | Text prompt on the node itself |
| `size` | COMBO | `1024x1024` | `1024x1024`, `1536x1024`, `1024x1536`, `auto` | Image dimensions |
| `quality` | COMBO | `auto` | `low`, `medium`, `high`, `auto` | Generation quality |
| `background` | COMBO | `auto` | `transparent`, `opaque`, `auto` | Background type |
| `n` | INT | `1` | 1–4 | Number of images to generate |
| `output_format` | COMBO | `png` | `png`, `webp` | Output image format |

### Optional

| Parameter | Type | Description |
|-----------|------|-------------|
| `external_text` | STRING | Text from an external node, concatenated before `prompt` |

## Output

| Name | Type | Description |
|------|------|-------------|
| `IMAGE` | IMAGE | Standard ComfyUI IMAGE tensor (B, H, W, C) float32 [0,1]. When n>1, batch dimension holds multiple images. |

## Text Concatenation Logic

```
if external_text and prompt:
    final_prompt = external_text + "\n" + prompt
elif external_text:
    final_prompt = external_text
else:
    final_prompt = prompt
```

Both empty → raise error.

## Core Flow

1. Concatenate text inputs into `final_prompt`
2. Call OpenAI API: `client.images.generate(model="gpt-image-1", prompt=final_prompt, size=..., quality=..., background=..., n=..., output_format=...)`
3. Receive base64-encoded image data from response
4. Decode base64 → PIL Image → numpy array (H, W, C) uint8 → float32 / 255.0 → torch tensor
5. Stack multiple images along batch dimension if n > 1
6. Return tensor as IMAGE output

## Error Handling

Raise exceptions on failure (API errors, invalid key, content policy violations, network issues). ComfyUI will display the error and halt the workflow.

## File Structure

```
style-manager-comfy-ui/
├── __init__.py          # Node registration (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
├── nodes.py             # GPTImageGenerate node implementation
└── requirements.txt     # openai, Pillow
```

## Dependencies

- `openai` — OpenAI Python SDK
- `Pillow` — Image decoding
- `torch`, `numpy` — Provided by ComfyUI runtime

## API Key Handling

Provided as a node input (STRING). This is necessary for PaaS platforms like RunningHub where environment variables and config files are not user-controllable. The key will be present in workflow JSON.

## ComfyUI Integration Notes

- Node registered via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py`
- `RETURN_TYPES = ("IMAGE",)`
- `FUNCTION = "generate"`
- `CATEGORY = "Style Manager/GPT Image"`
- `external_text` uses ComfyUI's optional input mechanism (`"optional": {...}`)
