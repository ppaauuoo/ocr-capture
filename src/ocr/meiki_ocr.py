"""OCR via MeikiOCR (ONNX-based, lightweight, Japanese games)."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Keep CUDA/cuDNN DLL directories alive for ONNX Runtime on Windows.
_DLL_DIRS = []
if os.name == "nt":
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    dll_paths = []
    for name in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "torch/lib"):
        path = site_packages / name
        if path.exists():
            dll_paths.append(str(path))
            _DLL_DIRS.append(os.add_dll_directory(str(path)))
    os.environ["PATH"] = os.pathsep.join(dll_paths + [os.environ.get("PATH", "")])

import onnxruntime as ort
from meikiocr import MeikiOCR
from typing import List


class MeikiOCREngine:
    def __init__(self):
        # ONNX Runtime needs cuDNN loaded before creating CUDA sessions on Windows.
        ort.preload_dlls()
        # ponytail: downloads model on first run
        self._ocr = MeikiOCR(provider="CUDAExecutionProvider")
        if self._ocr.active_provider != "CUDAExecutionProvider":
            raise RuntimeError(
                f"MeikiOCR is using {self._ocr.active_provider}; "
                f"available providers: {ort.get_available_providers()}"
            )

    def run(self, image_path: str = "temp/image.png") -> List[dict]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        result = self._ocr.run_ocr(image)
        return result

    def extract(self, raw: List[dict]) -> List[str]:
        lines = [
            item["text"].strip()
            for item in raw
            if item.get("text", "").strip() and len(item["text"].strip()) > 1
        ]
        return lines
