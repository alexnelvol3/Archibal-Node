"""
Archibal Callback Node for ComfyUI
-----------------------------------
Sends full workflow provenance to the Archibal platform:
  - Final output (image batch, native VIDEO, and/or VHS_FILENAMES video), base64-encoded
  - All reference media found in the workflow, base64-encoded
  - All prompt/text values extracted from workflow nodes
  - All model/checkpoint names used
  - Full workflow JSON for backend parsing
  - Optional shot_label to support multi-shot projects

Authentication, two modes:
  - Paste the webhook URL from your Archibal Capture page into webhook_url and
    leave api_key empty (the token in the URL authenticates the request).
  - Or keep the legacy DEFAULT_WEBHOOK_URL and set api_key to a bearer key with
    the comfy:write scope.

Server-side only. Works in standard UI, API mode, and Comfy Cloud.
"""

import base64
import io
import logging
import os
import tempfile
from typing import Optional

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_REFERENCE_ITEMS = 10
MAX_BATCH_ITEMS = 20
# 70 MB raw -> ~93 MB base64, keeps the JSON body under the 100 MB edge cap
MAX_VIDEO_BYTES = 70 * 1024 * 1024
# Edge allows 300 s for uploads
HTTP_TIMEOUT = 300

# Legacy route; requires api_key. Capture-page webhook URLs need no key.
DEFAULT_WEBHOOK_URL = "https://archibal.ai/api/comfy/callback"

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".avi", ".mkv"})

# Per-process cache: node_id -> {"data": <base64 png>, "shot_label": str}
# Lets a downstream ArchibalCallback pull the output image of an upstream one
# even though it's a tensor flowing through the graph, not a file on disk.
_CALLBACK_IMAGE_CACHE: dict = {}

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
    "model",
})

PROMPT_FIELDS = frozenset({
    "text", "text_positive", "text_negative",
    "prompt", "negative_prompt", "clip_l", "t5xxl",
})

ALLOWED_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".exr",
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


def _video_file_to_b64(filepath: str) -> Optional[dict]:
    """Encode a video file on disk as a final_media payload entry."""
    try:
        if not os.path.isfile(filepath):
            logger.warning(f"Archibal: video file not found: {filepath}")
            return None
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            logger.warning(f"Archibal: unsupported video type {ext}, skipping {filepath}")
            return None
        size = os.path.getsize(filepath)
        if size > MAX_VIDEO_BYTES:
            logger.warning(
                f"Archibal: skipping video {filepath} "
                f"({size / 1024 / 1024:.1f} MB exceeds {MAX_VIDEO_BYTES / 1024 / 1024:.0f} MB limit)"
            )
            return None
        with open(filepath, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return {
            "type": "video",
            "format": ext.lstrip("."),
            "data": data,
            "filename": os.path.basename(filepath),
            "role": "final_output",
        }
    except Exception as e:
        logger.warning(f"Archibal: could not read video {filepath}: {e}")
        return None


def _get_temp_directory() -> str:
    try:
        import folder_paths
        return folder_paths.get_temp_directory()
    except Exception:
        return tempfile.gettempdir()


def _encode_native_video(video) -> Optional[dict]:
    """Encode a ComfyUI native VIDEO input (VideoInput API). Tries the source
    file path first; falls back to save_to() into a temp file."""
    try:
        # VideoFromFile keeps the original path in __file (name-mangled);
        # check the common public-ish spots without depending on any one API.
        for attr in ("_VideoFromFile__file", "file", "path"):
            src = getattr(video, attr, None)
            if isinstance(src, str) and os.path.isfile(src):
                return _video_file_to_b64(src)

        tmp_dir = _get_temp_directory()
        os.makedirs(tmp_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=tmp_dir)
        os.close(fd)
        try:
            video.save_to(tmp_path)
            return _video_file_to_b64(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    except Exception as e:
        logger.warning(f"Archibal: could not encode VIDEO input: {e}")
        return None


def _encode_vhs_video(vhs_filenames) -> Optional[dict]:
    """Encode the final video from a VHS_FILENAMES value: (save_output, [paths...]).
    The last entry with a video extension is the muxed output."""
    try:
        paths = None
        if isinstance(vhs_filenames, (tuple, list)) and len(vhs_filenames) == 2:
            paths = vhs_filenames[1]
        elif isinstance(vhs_filenames, (tuple, list)):
            paths = vhs_filenames
        if not paths:
            logger.warning(f"Archibal: unexpected VHS_FILENAMES value: {vhs_filenames!r}")
            return None
        for candidate in reversed(list(paths)):
            if not isinstance(candidate, str):
                continue
            if os.path.splitext(candidate)[1].lower() in VIDEO_EXTENSIONS:
                return _video_file_to_b64(candidate)
        logger.warning("Archibal: no video file found in VHS_FILENAMES")
        return None
    except Exception as e:
        logger.warning(f"Archibal: could not encode VHS video: {e}")
        return None


def _collect_ancestors(prompt: dict, start_node_id, boundary_class: str = "ArchibalCallback") -> set:
    """Return the set of node IDs reachable by walking input connections
    backwards from start_node_id (inclusive). Traversal stops at any upstream
    node whose class_type matches boundary_class, so each ArchibalCallback only
    captures the slice of the workflow since the previous one."""
    ancestors: set = set()
    if start_node_id is None:
        return ancestors
    start = str(start_node_id)
    if start not in prompt:
        return ancestors
    stack = [start]
    while stack:
        nid = stack.pop()
        if nid in ancestors:
            continue
        ancestors.add(nid)
        node = prompt.get(nid)
        if not node:
            continue
        if nid != start and node.get("class_type") == boundary_class:
            continue
        for value in node.get("inputs", {}).values():
            # ComfyUI represents connections as [node_id, output_index]
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
                upstream = str(value[0])
                if upstream in prompt and upstream not in ancestors:
                    stack.append(upstream)
    return ancestors


def _extract_provenance(prompt: dict, node_ids: Optional[set] = None) -> dict:
    prompts = []
    models = []
    reference_files = []

    for node_id, node in prompt.items():
        if node_ids is not None and node_id not in node_ids:
            continue
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

        seen_model_key = set()
        for field in MODEL_FIELDS:
            val = inputs.get(field)
            if not isinstance(val, str) or not val.strip():
                continue
            dedup_key = f"{class_type}:{val}"
            if dedup_key in seen_model_key:
                continue
            seen_model_key.add(dedup_key)
            models.append({
                "node_id": node_id,
                "node_type": class_type,
                "field": field,
                "name": val.strip(),
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
    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("image", "video")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Only needed for the legacy /api/comfy/callback route. "
                            "Leave empty when using a webhook URL from the Capture page."
                        ),
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "vhs_filenames": ("VHS_FILENAMES",),
                "project_id": ("INT", {"default": 0}),
                "webhook_url": (
                    "STRING",
                    {
                        "default": DEFAULT_WEBHOOK_URL,
                        "multiline": False,
                        "tooltip": (
                            "Paste the webhook URL from your Archibal Capture page. "
                            "Defaults to the legacy route, which requires api_key."
                        ),
                    },
                ),
                "include_references": ("BOOLEAN", {"default": True}),
                "shot_label": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional label for this shot (e.g. 'Shot 01', 'Scene 3 - Wide'). "
                            "Multiple ArchibalCallback nodes in the same project should each "
                            "have a unique shot_label."
                        ),
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    def run(
        self,
        api_key,
        image=None,
        video=None,
        vhs_filenames=None,
        project_id=0,
        shot_label="",
        webhook_url=DEFAULT_WEBHOOK_URL,
        include_references=True,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        if not webhook_url:
            logger.warning("Archibal: no webhook URL, skipping")
            return (image, video)

        if image is None and video is None and vhs_filenames is None:
            logger.warning("Archibal: no image or video connected, skipping")
            return (image, video)

        final_media = []
        first_image_b64 = None

        if image is not None:
            if not hasattr(image, "shape") or len(image.shape) < 3:
                logger.warning(
                    "Archibal: unexpected image tensor shape %r", getattr(image, "shape", None)
                )
            else:
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
                if final_media:
                    first_image_b64 = final_media[0]["data"]

        video_entry = None
        if video is not None:
            video_entry = _encode_native_video(video)
        if video_entry is None and vhs_filenames is not None:
            video_entry = _encode_vhs_video(vhs_filenames)
        if video_entry:
            final_media.append(video_entry)

        if not final_media:
            logger.warning("Archibal: nothing could be encoded, skipping")
            return (image, video)

        if first_image_b64 and unique_id is not None:
            _CALLBACK_IMAGE_CACHE[str(unique_id)] = {
                "data": first_image_b64,
                "shot_label": shot_label.strip() if shot_label else "",
            }

        ancestors: set = set()
        if prompt:
            ancestors = _collect_ancestors(prompt, unique_id)
            provenance = _extract_provenance(prompt, ancestors)
        else:
            provenance = {"prompts": [], "models": [], "reference_files": []}

        reference_media = []
        if include_references and provenance["reference_files"]:
            input_dir = _get_input_directory()
            reference_media = _encode_references(provenance["reference_files"], input_dir)

        prior_archibal = []
        if include_references and prompt and unique_id is not None:
            self_id = str(unique_id)
            for nid in ancestors:
                if nid == self_id:
                    continue
                node = prompt.get(nid) or {}
                if node.get("class_type") != "ArchibalCallback":
                    continue
                cached = _CALLBACK_IMAGE_CACHE.get(nid)
                entry = {
                    "node_id": nid,
                    "shot_label": cached.get("shot_label", "") if cached else "",
                }
                prior_archibal.append(entry)
                if cached and len(reference_media) < MAX_REFERENCE_ITEMS:
                    reference_media.append({
                        "type": "image",
                        "format": "png",
                        "data": cached["data"],
                        "node_id": nid,
                        "node_type": "ArchibalCallback",
                        "role": "prior_archibal_output",
                        "shot_label": cached.get("shot_label", ""),
                    })

        payload = {
            "workflow_json": prompt or {},
            "final_media": final_media,
            "reference_media": reference_media,
            "prompts": provenance["prompts"],
            "models": provenance["models"],
        }

        if project_id and project_id > 0:
            payload["project_id"] = project_id

        if shot_label and shot_label.strip():
            payload["shot_label"] = shot_label.strip()

        if prior_archibal:
            payload["prior_archibal"] = prior_archibal

        if first_image_b64:
            payload["image_b64"] = first_image_b64

        if extra_pnginfo:
            payload["extra_pnginfo"] = extra_pnginfo

        api_key = (api_key or "").strip()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.post(webhook_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if "ingested" in data:  # capture-page webhook route
                        logger.info(
                            f"Archibal: project={data.get('project_id')} | "
                            f"ingested={data.get('ingested')}/{data.get('received')} | "
                            f"duplicates={data.get('duplicates')} | "
                            f"assets={data.get('asset_ids')}"
                        )
                    else:  # legacy /api/comfy/callback
                        logger.info(
                            f"Archibal: project={data.get('project_id')} | "
                            f"shot={data.get('shot_label')!r} | "
                            f"models={data.get('models_found')} | "
                            f"risk={data.get('risk_level')} | "
                            f"refs={data.get('references_stored', 0)} | "
                            f"replaced={data.get('replaced_asset')}"
                        )
                else:
                    logger.warning(f"Archibal: HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Archibal: callback failed: {e}")

        return (image, video)


NODE_CLASS_MAPPINGS = {
    "ArchibalCallback": ArchibalCallback,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchibalCallback": "Archibal Callback",
}
