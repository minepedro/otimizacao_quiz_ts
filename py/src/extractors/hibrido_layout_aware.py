"""
Extrator hibrido layout-aware: PyMuPDF + DocLayout-YOLO + RapidOCR seletivo.

Diferenca pro `hibrido_pymupdf_rapidocr`:
- Usa DocLayout-YOLO pra DETECTAR regioes na pagina (title, text, figure,
  table, formula, header/footer)
- OCR APENAS em regioes detectadas como `figure` (mais preciso que filtrar
  imagens por hash/tamanho como o hibrido anterior)
- Cada bloco vira um BlocoExtraido com TIPO SEMANTICO correto
  (title/paragraph/table/etc), nao mais tudo "paragraph"
- Reading order top-down, left-right baseado no bbox do layout (resolve
  PDFs multi-coluna)
- Headers/rodapes ("abandon") vao pro JSON mas NAO pro texto puro
  (matem qualidade da metrica)

Pipeline:
  1. PyMuPDF: texto vetorial por pagina (com bbox dos blocos)
  2. Renderizar pagina em imagem (DPI 150 — suficiente pra YOLO)
  3. DocLayout-YOLO: detectar regioes da pagina
  4. Pra cada regiao em reading order:
     a. Encontrar blocos vetoriais sobrepostos com a regiao
     b. Se NAO tem texto vetorial (regiao = imagem pura): OCR + filtra confidence
     c. Se TEM texto vetorial: usa texto vetorial (com tipo da regiao)
     d. Mapeia classe layout -> TipoBloco
  5. Output: BlocoExtraido[] tipados + texto puro (sem header/footer) +
     markdown estruturado
"""

from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import fitz
import numpy as np

from ..schemas import BlocoExtraido, ResultadoExtracao
from . import layout_detector
from .layout_detector import RegiaoLayout, classe_para_tipo_bloco

if TYPE_CHECKING:
    from rapidocr_onnxruntime import RapidOCR

NOME_EXTRATOR = "hibrido-layout-aware"

DPI_RENDER = 150  # menor que outros extratores (YOLO funciona bem em 1024px)
CONFIDENCE_OCR_MIN = 0.5
LIMIAR_OCR_MIN_CHARS = 5

# Confidence minima do layout pra confiar no tipo (regioes de baixa
# confidence sao mantidas mas com tipo "paragraph" generico)
LAYOUT_CONF_FORTE = 0.30

# Classes que NAO entram no texto puro (so nos blocks)
CLASSES_FORA_DO_TEXTO = {"abandon"}


_OCR: "RapidOCR | None" = None


def _get_ocr() -> "RapidOCR":
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
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


def _bbox_intersecta(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Area da intersecao / area de a (0..1). Util pra detectar bloco vetorial dentro de regiao."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1.0)
    return inter / area_a


def _extrair_texto_vetorial(page: fitz.Page) -> list[dict]:
    """Devolve lista de blocos vetoriais com bbox em pontos (PDF coords)."""
    blocos = []
    for bx in page.get_text("blocks", sort=True):
        if len(bx) < 7:
            continue
        x0, y0, x1, y1, texto, block_no, block_type = bx[:7]
        if block_type != 0:
            continue
        t = (texto or "").strip()
        if not t:
            continue
        blocos.append(
            {
                "bbox_pdf": (float(x0), float(y0), float(x1), float(y1)),
                "text": t,
                "block_no": int(block_no),
            }
        )
    return blocos


def _ocr_recorte(img_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> str:
    """OCR de um recorte da imagem (bbox em pixels). Retorna texto."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = img_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return ""
    recorte = img_bgr[y1:y2, x1:x2].copy()
    ocr = _get_ocr()
    try:
        resultado, _ = ocr(recorte)
    except Exception:
        return ""
    if not resultado:
        return ""
    linhas: list[str] = []
    for item in resultado:
        score = float(item[2]) if len(item) > 2 else 1.0
        if score < CONFIDENCE_OCR_MIN:
            continue
        t = (item[1] or "").strip()
        if t:
            linhas.append(t)
    return "\n".join(linhas)


def _converter_bbox_pdf_para_img(
    bbox_pdf: tuple[float, float, float, float],
    page_width_pt: float,
    page_height_pt: float,
    img_width_px: int,
    img_height_px: int,
) -> tuple[float, float, float, float]:
    """Converte bbox de coords PDF (pontos) pra coords da imagem (pixels)."""
    sx = img_width_px / page_width_pt
    sy = img_height_px / page_height_pt
    x1, y1, x2, y2 = bbox_pdf
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


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

    blocks_finais: list[BlocoExtraido] = []
    paginas_texto: list[str] = []
    md_partes: list[str] = []

    try:
        num_paginas = doc.page_count
        for page_idx, page in enumerate(doc):
            page_width_pt = page.rect.width
            page_height_pt = page.rect.height

            # 1. Render + 2. Layout
            try:
                img_bgr = _renderizar_pagina_bgr(page)
                regioes = layout_detector.detectar(img_bgr)
            except Exception:
                regioes = []
                img_bgr = None

            # 3. Texto vetorial (com bbox em coords PDF)
            blocos_vetorial = _extrair_texto_vetorial(page)

            # Converte bbox vetorial pra coords da imagem pra cruzar com layout
            img_h, img_w = (img_bgr.shape[:2] if img_bgr is not None else (0, 0))
            for b in blocos_vetorial:
                if img_w > 0:
                    b["bbox_img"] = _converter_bbox_pdf_para_img(
                        b["bbox_pdf"], page_width_pt, page_height_pt, img_w, img_h
                    )
                else:
                    b["bbox_img"] = b["bbox_pdf"]
                b["usado"] = False

            partes_texto_pagina: list[str] = []
            partes_md_pagina: list[str] = []

            # 4. Pra cada regiao em reading order
            for ri, regiao in enumerate(regioes):
                tipo = classe_para_tipo_bloco(regiao.classe)

                # Acha blocos vetoriais sobrepostos com a regiao
                vetoriais_dentro = [
                    b for b in blocos_vetorial
                    if not b["usado"] and _bbox_intersecta(b["bbox_img"], regiao.bbox) >= 0.5
                ]

                if vetoriais_dentro:
                    # Tem texto vetorial — usa ele com tipo da regiao
                    texto_regiao = "\n".join(b["text"] for b in vetoriais_dentro)
                    for b in vetoriais_dentro:
                        b["usado"] = True
                else:
                    # Sem texto vetorial — OCR (so se for figure/table)
                    if regiao.classe in ("figure", "table", "isolate_formula") and img_bgr is not None:
                        texto_regiao = _ocr_recorte(img_bgr, regiao.bbox)
                    else:
                        continue  # regiao sem texto vetorial e sem OCR — pula

                texto_regiao = texto_regiao.strip()
                if len(texto_regiao) < LIMIAR_OCR_MIN_CHARS and regiao.classe != "title":
                    continue

                bloco = BlocoExtraido(
                    type=tipo,
                    text=texto_regiao,
                    page_idx=page_idx,
                    bbox=regiao.bbox,
                    block_id=f"p{page_idx}-r{ri}",
                    level=1 if tipo == "title" else None,
                )
                blocks_finais.append(bloco)

                # Pra texto puro: pula header/footer
                if regiao.classe not in CLASSES_FORA_DO_TEXTO:
                    partes_texto_pagina.append(texto_regiao)

                    # Markdown
                    if tipo == "title":
                        partes_md_pagina.append(f"# {texto_regiao}")
                    elif tipo == "table":
                        partes_md_pagina.append(f"```\n{texto_regiao}\n```")
                    elif tipo == "formula":
                        partes_md_pagina.append(f"$$\n{texto_regiao}\n$$")
                    elif tipo == "caption":
                        partes_md_pagina.append(f"_{texto_regiao}_")
                    else:
                        partes_md_pagina.append(texto_regiao)

            # Blocos vetoriais que NAO casaram com nenhuma regiao detectada:
            # adiciona como paragraph generico (fallback pra cobrir lacunas
            # do detector — algumas regioes ele perde, principalmente em
            # paginas estranhas)
            for b in blocos_vetorial:
                if b["usado"]:
                    continue
                blocks_finais.append(
                    BlocoExtraido(
                        type="paragraph",
                        text=b["text"],
                        page_idx=page_idx,
                        bbox=b["bbox_pdf"],
                        block_id=f"p{page_idx}-vet-{b['block_no']}",
                    )
                )
                partes_texto_pagina.append(b["text"])
                partes_md_pagina.append(b["text"])

            paginas_texto.append("\n".join(partes_texto_pagina))
            if partes_md_pagina:
                md_partes.append("\n\n".join(partes_md_pagina))

        texto = "\n\n".join(paginas_texto)
        markdown = "\n\n---\n\n".join(md_partes)
    finally:
        doc.close()

    return ResultadoExtracao(
        extrator=NOME_EXTRATOR,
        texto=texto,
        blocks=blocks_finais,
        markdown=markdown,
        num_paginas=num_paginas,
        num_caracteres=len(texto),
        tempo_segundos=round(time.perf_counter() - inicio, 3),
        erro=None,
    )
