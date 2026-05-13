"""
PDF backend MIT/Apache (pypdfium2 + pdfplumber).

Substitui PyMuPDF (AGPL) por:
  - pypdfium2 (Apache 2.0, Google PDFium): renderizacao
  - pdfplumber (MIT): extracao de texto vetorial com bbox + font

Mantem mesma interface da versao PyMuPDF anterior pra plug-and-play
no hibrido_layout_aware.py.

API publica:
  - PdfBackend(caminho): abre PDF, fecha automatico
  - backend.page_count
  - backend.get_page(i): retorna PageInfo
  - PageInfo.width / .height (em pontos)
  - PageInfo.render_bgr(dpi=150): retorna numpy array BGR
  - PageInfo.extract_blocks(): retorna lista de dict equivalente ao
      modo 'blocks' do PyMuPDF, com:
        bbox_pdf: (x0, y0, x1, y1) em pontos PDF (top-left)
        text: str
        block_no: int
        font_size: float
        is_bold: bool
        is_italic: bool
"""

from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Iterator

import cv2
import numpy as np
import pdfplumber
import pypdfium2 as pdfium

# ============================================================
# PageInfo — wrapper da pagina
# ============================================================


class PageInfo:
    """Wrapper de uma pagina PDF combinando pypdfium2 (render) + pdfplumber (texto)."""

    def __init__(self, pypdfium_page, pdfplumber_page):
        self._py = pypdfium_page
        self._pl = pdfplumber_page
        self.width = float(pypdfium_page.get_width())
        self.height = float(pypdfium_page.get_height())

    def render_bgr(self, dpi: int = 150) -> np.ndarray:
        """Renderiza pagina em numpy array BGR pra OpenCV/YOLO."""
        scale = dpi / 72
        bitmap = self._py.render(scale=scale, rotation=0)
        img_pil = bitmap.to_pil()  # RGB
        # Converte PIL RGB → numpy BGR
        arr = np.array(img_pil)
        if arr.ndim == 2:  # grayscale
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:  # RGBA
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        bitmap.close()
        return arr

    def extract_blocks(self) -> list[dict]:
        """Extrai blocos vetoriais agrupando palavras proximas.

        Retorna lista igual ao modo 'blocks' do PyMuPDF:
          { bbox_pdf: (x0,y0,x1,y1), text, block_no, font_size, is_bold, is_italic }

        Algoritmo:
          1. extract_words com font info do pdfplumber
          2. Agrupa palavras em LINHAS (mesmo top, ±2px)
          3. Agrupa linhas em PARAGRAFOS (gap vertical < 1.5x line-height)
          4. Pra cada paragrafo: bbox uniao, font_size mediano, bold/italic majoritario
        """
        # 1. Palavras com font info
        try:
            words = self._pl.extract_words(
                extra_attrs=["fontname", "size"],
                use_text_flow=True,  # respeita ordem natural do stream PDF
                keep_blank_chars=False,
            )
        except Exception:
            return []

        if not words:
            return []

        # 2. Agrupa em linhas (palavras com top similar)
        # Ordena por (top, x0) — top-down, left-right
        words_sorted = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
        linhas: list[list[dict]] = []
        for w in words_sorted:
            if not linhas:
                linhas.append([w])
                continue
            ultima_linha = linhas[-1]
            # Mesma "linha" se top similar (±3px) e altura similar
            top_ult = ultima_linha[0]["top"]
            if abs(w["top"] - top_ult) < 3:
                ultima_linha.append(w)
            else:
                linhas.append([w])

        if not linhas:
            return []

        # 3. Agrupa linhas em paragrafos (gap vertical < 1.5x altura linha)
        paragrafos: list[list[list[dict]]] = []
        for linha in linhas:
            if not paragrafos:
                paragrafos.append([linha])
                continue
            ult_par = paragrafos[-1]
            ult_linha_par = ult_par[-1]
            ult_bottom = max(w["bottom"] for w in ult_linha_par)
            cur_top = min(w["top"] for w in linha)
            gap = cur_top - ult_bottom
            altura_linha = max(w["bottom"] - w["top"] for w in linha)
            # Mesma coluna (X overlap razoavel)?
            ult_x_min = min(w["x0"] for w in ult_linha_par)
            ult_x_max = max(w["x1"] for w in ult_linha_par)
            cur_x_min = min(w["x0"] for w in linha)
            cur_x_max = max(w["x1"] for w in linha)
            x_overlap = min(ult_x_max, cur_x_max) - max(ult_x_min, cur_x_min)
            mesma_coluna = x_overlap > 0
            # Threshold conservador: gap < 0.5x altura_linha = continuacao do paragrafo
            # gap >= 0.5x = paragrafo novo. Aproxima do comportamento do PyMuPDF blocks.
            if mesma_coluna and gap < 0.5 * altura_linha:
                ult_par.append(linha)
            else:
                paragrafos.append([linha])

        # 4. Pra cada paragrafo: bbox uniao, texto, font_size, bold/italic
        blocos = []
        for block_no, par in enumerate(paragrafos):
            todas_palavras = [w for linha in par for w in linha]
            x0 = min(w["x0"] for w in todas_palavras)
            x1 = max(w["x1"] for w in todas_palavras)
            y0 = min(w["top"] for w in todas_palavras)
            y1 = max(w["bottom"] for w in todas_palavras)
            # Texto: linha por linha, palavras com espaco entre
            linhas_texto = []
            for linha in par:
                texto_linha = " ".join(w["text"] for w in linha)
                linhas_texto.append(texto_linha)
            texto = "\n".join(linhas_texto)
            # Font_size mediano
            sizes = [float(w.get("size", 12.0)) for w in todas_palavras if w.get("size")]
            font_size = median(sizes) if sizes else 12.0
            # Bold/Italic via fontname (heuristica)
            n_bold = sum(1 for w in todas_palavras if "bold" in (w.get("fontname", "") or "").lower())
            n_italic = sum(1 for w in todas_palavras if "italic" in (w.get("fontname", "") or "").lower() or "oblique" in (w.get("fontname", "") or "").lower())
            n = max(1, len(todas_palavras))
            is_bold = n_bold >= n * 0.80
            is_italic = n_italic >= n * 0.80

            blocos.append(
                {
                    "bbox_pdf": (float(x0), float(y0), float(x1), float(y1)),
                    "text": texto,
                    "block_no": block_no,
                    "font_size": float(font_size),
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                }
            )

        return blocos


# ============================================================
# PdfBackend — wrapper do documento
# ============================================================


class PdfBackend:
    """Wrapper de PDF combinando pypdfium2 + pdfplumber.

    Uso:
        with PdfBackend(caminho) as pdf:
            for i in range(pdf.page_count):
                page = pdf.get_page(i)
                img = page.render_bgr(dpi=150)
                blocos = page.extract_blocks()
    """

    def __init__(self, caminho: str | Path):
        self.caminho = Path(caminho)
        self._py_doc = pdfium.PdfDocument(str(self.caminho))
        self._pl_doc = pdfplumber.open(self.caminho)
        self.page_count = len(self._py_doc)

    def get_page(self, i: int) -> PageInfo:
        return PageInfo(self._py_doc[i], self._pl_doc.pages[i])

    def __iter__(self) -> Iterator[PageInfo]:
        for i in range(self.page_count):
            yield self.get_page(i)

    def close(self) -> None:
        try:
            self._pl_doc.close()
        except Exception:
            pass
        try:
            self._py_doc.close()
        except Exception:
            pass

    def __enter__(self) -> "PdfBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
