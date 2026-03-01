from __future__ import annotations

import os
import time
import random
from typing import Any, Dict, List

import numpy as np
from PIL import Image

try:
    import folder_paths  # ComfyUI
except Exception:
    folder_paths = None


def _get_temp_dir() -> str:
    if folder_paths is not None and hasattr(folder_paths, "get_temp_directory"):
        return folder_paths.get_temp_directory()
    return os.path.join(os.getcwd(), "temp")


def _unique_basename(prefix: str = "preview") -> str:
    return f"{prefix}_{int(time.time()*1000)}_{random.randint(0, 999999):06d}"


def _tensor_to_pil(img: Any) -> Image.Image:
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()

    arr = np.asarray(img)

    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported image shape for preview: {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] == 4:
        pil = Image.fromarray(arr, mode="RGBA").convert("RGB")
    else:
        pil = Image.fromarray(arr, mode="RGB")

    return pil


class PreviewImageNoMetadata:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def preview(self, image):

        temp_dir = _get_temp_dir()
        os.makedirs(temp_dir, exist_ok=True)

        images_ui: List[Dict[str, str]] = []

        batch = image

        if hasattr(batch, "detach"):
            b = int(batch.shape[0])
            get_item = lambda i: batch[i]
        else:
            arr = np.asarray(batch)
            b = int(arr.shape[0])
            get_item = lambda i: arr[i]

        base = _unique_basename("preview")

        for i in range(b):
            pil = _tensor_to_pil(get_item(i))

            filename = f"{base}_{i:05d}.png"
            out_path = os.path.join(temp_dir, filename)

            # Всегда сохраняем БЕЗ metadata
            pil.save(out_path, compress_level=4)

            images_ui.append({"filename": filename, "subfolder": "", "type": "temp"})

        return {"ui": {"images": images_ui}, "result": (image,)}
