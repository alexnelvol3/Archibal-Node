"""
Archibal Callback Node for ComfyUI
-----------------------------------
Sends the generated image and full workflow JSON to the Archibal platform
for compliance cataloguing, model extraction, and risk assessment.

Server-side only — no JavaScript or custom UI required.
Works in standard UI, API mode, and Comfy Cloud.
"""

import base64
import io
import logging

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ArchibalCallback:
    """
    Receives the final image, serialises it with the full workflow,
    and POSTs both to the Archibal webhook. The image passes through unchanged.
    """

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
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    def run(self, image, api_key, project_id=0, webhook_url="", prompt=None):
        if not api_key:
            logger.warning("Archibal: No API key provided, skipping callback")
            return (image,)

        if not webhook_url:
            logger.warning("Archibal: No webhook URL provided, skipping callback")
            return (image,)

        # Convert image tensor to PNG bytes
        # image is a torch tensor of shape (batch, height, width, channels) with values 0-1
        try:
            img_array = image[0].cpu().numpy()
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)

            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Archibal: Failed to encode image: {e}")
            return (image,)

        # Build payload
        payload = {
            "workflow_json": prompt or {},
            "image_b64": image_b64,
        }
        if project_id and project_id > 0:
            payload["project_id"] = project_id

        # Send to Archibal
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.post(
                    webhook_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(
                        f"Archibal: Run saved to project {data.get('project_id')}, "
                        f"models found: {data.get('models_found')}, "
                        f"risk: {data.get('risk_level')}"
                    )
                else:
                    logger.warning(
                        f"Archibal: Callback returned {resp.status_code}: "
                        f"{resp.text[:200]}"
                    )
        except Exception as e:
            logger.error(f"Archibal: Failed to send callback: {e}")

        # Pass image through unchanged
        return (image,)


NODE_CLASS_MAPPINGS = {
    "ArchibalCallback": ArchibalCallback,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArchibalCallback": "Archibal Callback",
}
