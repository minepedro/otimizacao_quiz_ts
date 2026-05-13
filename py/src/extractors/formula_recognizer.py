"""
Reconhecedor de formulas matematicas — converte imagem pra LaTeX.

Usa pix2tex (LaTeX-OCR) — modelo pre-treinado especifico pra
formulas matematicas digitadas.

Modelo: pix2tex weights v0.0.1 (~115MB)
Tempo: ~1-2s por formula em CPU.

LIMITACAO: foi treinado em formulas DIGITADAS (tipografadas com LaTeX),
nao em manuscrito. PDFs scaneados de manuscrito tipo Aula 04 resolucao
provavelmente vao errar.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pix2tex.cli import LatexOCR

_OCR: "LatexOCR | None" = None
_LOCK = threading.Lock()


def _get_ocr():
    global _OCR
    if _OCR is None:
        with _LOCK:
            if _OCR is None:
                from pix2tex.cli import LatexOCR
                _OCR = LatexOCR()
    return _OCR


def reconhecer_formula(img_bgr: np.ndarray) -> str:
    """Recebe imagem da formula em BGR (numpy), devolve string LaTeX.

    Em caso de erro, devolve string vazia (caller pode fallback pra OCR cru).
    """
    if img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
        return ""
    try:
        from PIL import Image
        import cv2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
    except Exception:
        return ""
    try:
        ocr = _get_ocr()
        latex = ocr(img_pil)
        return (latex or "").strip()
    except Exception:
        return ""
