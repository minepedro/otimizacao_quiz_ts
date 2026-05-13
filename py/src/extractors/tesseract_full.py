"""
Extrator OCR puro com Tesseract (via pytesseract).

Equivalente Python do `tesseract.ts` TS, mas usando o binario
nativo do Tesseract em vez do tesseract.js (WASM). Mais rapido
e com modelos LSTM completos.

Setup necessario:
  1. Binario do Tesseract instalado:
     winget install --id UB-Mannheim.TesseractOCR
  2. Modelos `por.traineddata` e `eng.traineddata` em `py/tessdata/`:
     curl -L -o py/tessdata/por.traineddata \\
       https://github.com/tesseract-ocr/tessdata/raw/main/por.traineddata
     curl -L -o py/tessdata/eng.traineddata \\
       https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata

Filtro de confidence em nivel de palavra (TESS_CONFIDENCE_MIN).
Tesseract 5 LSTM devolve confidence por palavra via image_to_data.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract

from ..schemas import BlocoExtraido, ResultadoExtracao

NOME_EXTRATOR = "tesseract-full"

# Path do binario tesseract.exe (instalado via winget UB-Mannheim)
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Path do diretorio com por.traineddata + eng.traineddata
TESSDATA_DIR = Path(__file__).resolve().parents[3] / "py" / "tessdata"

# Idiomas a usar (PT + EN)
LANG = "por+eng"

# DPI da renderizacao
DPI_RENDER = 200

# Filtro de confidence (0..100). Tesseract 5 LSTM devolve por palavra.
TESS_CONFIDENCE_MIN = 50


# Configuracao global do pytesseract — feita 1x por processo
_CONFIGURADO = False


def _configurar() -> None:
    global _CONFIGURADO
    if _CONFIGURADO:
        return
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)
    _CONFIGURADO = True


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


def _ocr_pagina(img: np.ndarray) -> tuple[str, list[tuple[float, float, float, float, str, float]]]:
    """OCR uma pagina. Retorna (texto, lista de palavras com bbox + conf).

    Usa image_to_data pra ter confidence por palavra (filtragem fina).
    """
    data = pytesseract.image_to_data(
        img, lang=LANG, output_type=pytesseract.Output.DICT
    )
    n = len(data["text"])
    palavras: list[tuple[float, float, float, float, str, float]] = []
    for i in range(n):
        try:
            conf = float(data["conf"][i])
        except (ValueError, KeyError):
            continue
        texto = (data["text"][i] or "").strip()
        if not texto:
            continue
        if conf < TESS_CONFIDENCE_MIN:
            continue
        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])
        palavras.append((x, y, x + w, y + h, texto, conf))

    # Reagrupa por linha (mesmo block_num + par_num + line_num)
    # Pra simplicidade aqui, junta tudo pelo Y/H — palavras na mesma
    # linha tem `top` parecido. Vou usar block_num+par_num+line_num.
    linhas: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n):
        if not (data["text"][i] or "").strip():
            continue
        try:
            if float(data["conf"][i]) < TESS_CONFIDENCE_MIN:
                continue
        except (ValueError, KeyError):
            continue
        chave = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        linhas.setdefault(chave, []).append(i)

    linhas_texto: list[str] = []
    for chave in sorted(linhas):
        idxs = linhas[chave]
        partes = [(data["text"][j] or "").strip() for j in idxs]
        partes = [p for p in partes if p]
        if partes:
            linhas_texto.append(" ".join(partes))
    texto = "\n".join(linhas_texto)
    return texto, palavras


def extrair(caminho_pdf: str | Path) -> ResultadoExtracao:
    inicio = time.perf_counter()
    caminho = Path(caminho_pdf)
    _configurar()

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

    paginas_texto: list[str] = []
    blocks: list[BlocoExtraido] = []

    try:
        for page_idx, page in enumerate(doc):
            try:
                img = _renderizar_pagina_bgr(page)
                texto_pagina, palavras = _ocr_pagina(img)
            except Exception:
                paginas_texto.append("")
                continue

            paginas_texto.append(texto_pagina)
            for i, (x1, y1, x2, y2, txt, conf) in enumerate(palavras):
                blocks.append(
                    BlocoExtraido(
                        type="paragraph",
                        text=txt,
                        page_idx=page_idx,
                        bbox=(x1, y1, x2, y2),
                        block_id=f"p{page_idx}-w{i}",
                        ocr_confidence=round(conf, 2),
                    )
                )

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
