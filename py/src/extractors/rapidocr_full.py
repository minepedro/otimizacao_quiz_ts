"""
Extrator OCR puro com RapidOCR (modelos PaddleOCR via ONNX Runtime).

Equivalente Python do `tesseract.ts` TS: renderiza cada pagina em
PNG (PyMuPDF, DPI configuravel) e roda OCR completo. Texto vetorial
do PDF eh ignorado — soh OCR. Serve pra comparar com extratores
hibridos (texto + OCR seletivo) e medir o teto do OCR puro.

RapidOCR usa os MESMOS modelos do PaddleOCR (PP-OCRv4 detection +
recognition), mas via ONNX Runtime — sem dep de paddlepaddle, mais
estavel no Windows. Qualidade equivalente ao paddleocr puro.

Tem 1 instancia singleton do RapidOCR (init eh ~0.2s, mas reusar
evita overhead em batch).
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
    from rapidocr_onnxruntime import RapidOCR

NOME_EXTRATOR = "rapidocr-full"

# DPI da renderizacao. 200 eh bom balanco qualidade/tempo.
# DPI 300 melhora confidence mas dobra o tempo.
DPI_RENDER = 200

# Filtro de confidence. Linhas abaixo disso sao descartadas.
# 0.5 eh permissivo; 0.6-0.7 mais conservador (igual ao confidenceMin
# do hibrido-clean-conf TS).
CONFIDENCE_MIN = 0.5


_OCR: "RapidOCR | None" = None


def _get_ocr() -> "RapidOCR":
    """Singleton do RapidOCR pra evitar re-inicializar entre PDFs."""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def _renderizar_pagina_bgr(page: fitz.Page, dpi: int = DPI_RENDER) -> np.ndarray:
    """Renderiza pagina pra np.ndarray BGR (formato esperado pelo OCR)."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"PixMap com n={pix.n} (esperado 1/3/4)")


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
                resultado, _elapsed = ocr(img)
            except Exception as e:
                # Falha em 1 pagina nao quebra todo o PDF
                paginas_texto.append("")
                continue

            if not resultado:
                paginas_texto.append("")
                continue

            partes_pagina: list[str] = []
            for i, item in enumerate(resultado):
                # item: [bbox, text, score]
                bbox_pts = item[0]
                texto_linha = item[1] or ""
                score = float(item[2]) if len(item) > 2 else 1.0
                if score < CONFIDENCE_MIN:
                    continue
                texto_linha = texto_linha.strip()
                if not texto_linha:
                    continue
                partes_pagina.append(texto_linha)

                # bbox em pontos da imagem (4 pontos, polygon).
                # Converte pra (x1, y1, x2, y2) calculando min/max.
                xs = [p[0] for p in bbox_pts]
                ys = [p[1] for p in bbox_pts]
                bbox = (
                    float(min(xs)),
                    float(min(ys)),
                    float(max(xs)),
                    float(max(ys)),
                )

                blocks.append(
                    BlocoExtraido(
                        type="paragraph",
                        text=texto_linha,
                        page_idx=page_idx,
                        bbox=bbox,
                        block_id=f"p{page_idx}-l{i}",
                        ocr_confidence=round(score * 100, 2),
                    )
                )
            paginas_texto.append("\n".join(partes_pagina))

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
