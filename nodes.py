"""
Archibal Callback Node for ComfyUI
-----------------------------------
Sends full workflow provenance to the Archibal platform:
  - Final output (image batch or video file), base64-encoded
  - All reference media found in the workflow, base64-encoded
  - All prompt/text values extracted from workflow nodes
  - All model/checkpoint names used
  - Full workflow JSON for backend parsing

Server-side only. Works in standard UI, API mode, and Comfy Cloud.
"""

import base64
import io
import logging
import os
from typing import Optional

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_REFERENCE_ITEMS = 10
MAX_BATCH_ITEMS = 20

LOADER_NODES_IMAGE = {"LoadImage", "LoadImageMask"}
LOADER_NODES_VIDEO = {"VHS_LoadVideo", "LoadVideo"}
LOADER_NODES = LOADER_NODES_IMAGE | LOADER_NODES_VIDEO

MODEL_NODES = {
    "CheckpointLoaderSimple", "CheckpointLoader",
    "LoraLoader", "LoraLoaderModelOnly",
    "ControlNetLoader", "IPAdapterModelLoader",
    "UNETLoader", "VAELoader", "CLIPLoader",
    "DiffusersLoader",
}

MODEL_FIELDS = frozenset({
    "ckpt_name", "lora_name", "control_net_name",
    "model_name", "vae_name", "clip_name", "unet_name",
})

PROMPT_FIELDS = frozenset({
    "text", "text_positive", "text_negative",
    "prompt", "negative_prompt", "clip_l", "t5xxl",
})

ALLOWED_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
    ".mp4", ".mov", ".webm", ".avi", ".mkv",
})


def _get_input_directory() -> Optional[str]:
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except ImportError:
        base = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            base = os.path.dirname(base)
            candidate = os.path.join(base, "input")
            if os.path.isdir(candidate):
                return candidate
    return None


def _tensor_to_b64_png(tensor) -> Optional[str]:
    try:
        arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Archibal: image encode failed: {e}")
        return None


def _load_file_as_b64(filepath: str) -> Optional[str]:
    try:
        size = os.path.getsize(filepath)
        if size > MAX_REFERENCE_BYTES:
            logger.warning(
                f"Archibal: skipping {filepath} ({size / 1024 / 1024:.1f} MB exceeds limit)"
            )
            return None
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Archibal: could not read {filepath}: {e}")
        return None


def _extract_provenance(prompt: dict) -> dict:
    prompts = []
    models = []
    reference_files = []

    for node_id, node in prompt.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        for field, value in inputs.items():
            if field in PROMPT_FIELDS and isinstance(value, str) and value.strip():
                prompts.append({
                    "node_id": node_id,
                    "node_type": class_type,
                    "field": field,
                    "text": value.strip(),
                })

        if class_type in MODEL_NODES:
            for field in MODEL_FIELDS:
                val = inputs.get(field)
                if isinstance(val, str) and val:
                    models.append({
                        "node_id": node_id,
                        "node_type": class_type,
                        "field": field,
                        "name": val,
                    })

        if class_type in LOADER_NODES:
            filename = inputs.get("image") or inputs.get("video") or inputs.get("file")
            if isinstance(filename, str) and filename:
                media_type = "video" if class_type in LOADER_NODES_VIDEO else "image"
                reference_files.append({
                    "node_id": node_id,
                    "node_type": class_type,
                    "filename": filename,
                    "media_type": media_type,
                })

    return {
        "prompts": prompts,
        "models": models,
        "reference_files": reference_files,
    }


def _encode_references(reference_files: list, input_dir: Optional[str]) -> list:
    if not input_dir:
        return []

    encoded = []
    for ref in reference_files:
        if len(encoded) >= MAX_REFERENCE_ITEMS:
            logger.warning("Archibal: reference media cap reached, skipping remaining")
            break

        fname = ref["filename"]
        if os.path.isabs(fname):
            filepath = fname
        else:
            filepath = os.path.join(input_dir, fname)
            if not os.path.isfile(filepath):
                subfolder_path = os.path.join(input_dir, os.path.dirname(fname))
                if os.path.isdir(subfolder_path):
                    filepath = os.path.join(subfolder_path, os.path.basename(fname))

        if not os.path.isfile(filepath):
            logger.warning(f"Archibal: reference not found: {fname}")
            continue

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"Archibal: unsupported reference type {ext}, skipping {fname}")
            continue

        b64 = _load_file_as_b64(filepath)
        if not b64:
            continue

        encoded.append({
            "type": ref["media_type"],
            "format": ext.lstrip(".") or "bin",
            "data": b64,
            "filename": fname,
            "node_id": ref["node_id"],
            "node_type": ref["node_type"],
            "role": "reference",
        })

    return encoded


class ArchibalCallback:
    CATEGORY = "Archibal"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "project_id": ("INT", {"default": 0}),
                "webhook_url": (
                    "STRING",
                    {
                        "default": "https://api.archibal.ai/api/comfy/callback",
                        "multiline": False,
                    },
                ),
                "include_references": ("BOOLEAN", {"default": True}),
                "shot_label": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def run(
        self,
        image,
        api_key,
        project_id=0,
        webhook_url="https://api.archibal.ai/api/comfy/callback",
        include_references=True,
        shot_label="",
        prompt=None,
        extra_pnginfo=None,
    ):
        if not api_key:
            logger.warning("Archibal: no API key, skipping")
            return (image,)

        if not webhook_url:
            logger.warning("Archibal: no webhook URL, skipping")
            return (image,)

        if not hasattr(image, "shape") or len(image.shape) < 3:
            logger.warning("Archibal: unexpected image tensor shape %r", getattr(image, "shape", None))
            return (image,)

        final_media = []
        batch_size = min(image.shape[0], MAX_BATCH_ITEMS)
        for i in range(batch_size):
            b64 = _tensor_to_b64_png(image[i])
            if b64:
                final_media.append({
                    "type": "image",
                    "format": "png",
                    "data": b64,
                    "index": i,
                    "role": "final_output",
                })

        provenance = _extract_provenance(prompt) if prompt else {
            "prompts": [], "models": [], "reference_files": [],
        }

        reference_media = []
        if include_references and provenance["reference_files"]:
            input_dir = _get_input_directory()
            reference_media = _encode_references(provenance["reference_files"], input_dir)

        payload = {
            "workflow_json": prompt or {},
            "final_media": final_media,
            "reference_media": reference_media,
            "prompts": provenance["prompts"],
            "models": provenance["models"],
        }

        if project_id and project_id > 0:
            payload["project_id"] = project_id

        if shot_label:
            payload["shot_label"] = shot_label

        if final_media:
            payload["image_b64"] = final_media[0]["data"]

        if extra_pnginfo:
            payload["extra_pnginfo"] = extra_pnginfo

        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    webhook_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(
                        f"Archibal: project={data.get('project_id')} | "
                        f"models={data.get('models_found')} | "
                        f"risk={data.get('risk_level')} | "
                        f"refs={len(reference_media)} | "
                        f"prompts={len(provenance['prompts'])}"
                    )
                else:
                    logger.warning(f"Archibal: HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Archibal: callback failed: {e}")

        return (image,)


NODE_CLASS_MAPPINGS = {
    "ArchibalCallback": ArchibalCallback,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchibalCallback": "Archibal Callback",
}
