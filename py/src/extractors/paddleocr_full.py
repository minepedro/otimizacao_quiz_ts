"""
Extrator OCR puro com PaddleOCR (motor original do paddlepaddle).

Equivalente do `rapidocr_full` mas usando paddlepaddle direto. Em
teoria, PaddleOCR + paddlepaddle deveria dar resultado equivalente
ao RapidOCR (que usa os mesmos modelos via ONNX), mas:

  1. **No Windows + AMD Ryzen Zen5 com AVX-512**: o detector
     PP-OCRv5 server CRASHA silenciosamente dentro do C++ quando
     oneDNN/MKL-DNN esta habilitado (bug conhecido do oneDNN com
     AVX-512 da AMD Zen5). Workaround obrigatorio: `enable_mkldnn=False`.
  2. Modelo SERVER (mais pesado e preciso) tambem crasha em algumas
     configs. Por garantia, usamos o modelo MOBILE (menor mas
     equivalente em qualidade pra texto comum).

Pra GPU (RTX 5070 / Blackwell), precisa instalar paddlepaddle-gpu
separado. CPU funciona com a config abaixo.

Filtro de confidence em nivel de linha (CONFIDENCE_MIN).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import fitz
import numpy as np

from ..schemas import BlocoExtraido, ResultadoExtracao

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

NOME_EXTRATOR = "paddleocr-full"

DPI_RENDER = 200
CONFIDENCE_MIN = 0.5


_OCR: "PaddleOCR | None" = None


def _get_ocr() -> "PaddleOCR":
    """Singleton do PaddleOCR pra evitar re-init entre paginas/PDFs."""
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(
            lang="pt",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # Modelo mobile (menor mas estavel; server crasha em Zen5)
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
            # CRITICAL: oneDNN crasha com AVX-512 da AMD Zen5
            enable_mkldnn=False,
            cpu_threads=4,
        )
    return _OCR


def _renderizar_pagina_bgr(page: fitz.Page, dpi: int = DPI_RENDER) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"PixMap n={pix.n}")


def extrair(caminho_pdf: str | Path) -> ResultadoExtracao:
    inicio = time.perf_counter()
    caminho = Path(caminho_pdf)

    try:
        doc = fitz.open(caminho)
    except Exception as e:
        return ResultadoExtracao(
            extrator=NOME_EXTRATOR,
            texto="",
            blocks=[],
            markdown="",
            num_paginas=0,
            num_caracteres=0,
            tempo_segundos=round(time.perf_counter() - inicio, 3),
            erro=f"{type(e).__name__}: {e}",
        )

    ocr = _get_ocr()
    paginas_texto: list[str] = []
    blocks: list[BlocoExtraido] = []

    try:
        for page_idx, page in enumerate(doc):
            try:
                img = _renderizar_pagina_bgr(page)
                result = ocr.predict(img)
            except Exception:
                paginas_texto.append("")
                continue

            if not result:
                paginas_texto.append("")
                continue

            r = result[0]
            texts = r.get("rec_texts", []) or []
            scores = r.get("rec_scores", []) or []
            polys = r.get("rec_polys", []) or r.get("dt_polys", []) or []

            partes: list[str] = []
            for i, (txt, score) in enumerate(zip(texts, scores)):
                txt = (txt or "").strip()
                score = float(score)
                if not txt or score < CONFIDENCE_MIN:
                    continue
                partes.append(txt)
                bbox: tuple[float, float, float, float] | None = None
                if i < len(polys):
                    poly = polys[i]
                    try:
                        xs = [float(p[0]) for p in poly]
                        ys = [float(p[1]) for p in poly]
                        bbox = (min(xs), min(ys), max(xs), max(ys))
                    except Exception:
                        bbox = None
                blocks.append(
                    BlocoExtraido(
                        type="paragraph",
                        text=txt,
                        page_idx=page_idx,
                        bbox=bbox,
                        block_id=f"p{page_idx}-l{i}",
                        ocr_confidence=round(score * 100, 2),
                    )
                )
            paginas_texto.append("\n".join(partes))

        texto = "\n\n".join(paginas_texto)
        num_paginas = doc.page_count
    finally:
        doc.close()

    return ResultadoExtracao(
        extrator=NOME_EXTRATOR,
        texto=texto,
        blocks=blocks,
        markdown="",
        num_paginas=num_paginas,
        num_caracteres=len(texto),
        tempo_segundos=round(time.perf_counter() - inicio, 3),
        erro=None,
    )
