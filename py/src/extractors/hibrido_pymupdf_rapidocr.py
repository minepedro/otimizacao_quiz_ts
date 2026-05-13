"""
Extrator hibrido: PyMuPDF (texto vetorial) + RapidOCR seletivo nas imagens.

Equivalente Python do `hibrido-clean-conf-best` TS (vencedor MIT-only
do lab TS, cob 0.884 vs gabarito v1). Porta as 4 heuristicas-chave:

  1. Background detection por hash (SHA-256 dos bytes da imagem):
     se uma imagem aparece em >= 50% das paginas, eh background
     decorativo — pula OCR.
  2. Cache OCR por hash: nao OCR a mesma imagem 2x.
  3. Filtro de confidence: descarta linhas OCR com score < 0.5.
  4. Dedup OCR vs vetorial: pra cada frase OCR, se >= 60% das
     palavras ja estao no texto vetorial daquela pagina, descarta
     (provavel duplicacao).

Comparado ao TS (que usa unpdf + tesseract.js), aqui usa:
  - PyMuPDF (em vez de unpdf): mais rapido, pega bbox das imagens
  - RapidOCR (em vez de tesseract.js): modelos PaddleOCR via ONNX,
    melhor qualidade que tesseract em geral
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import fitz
import numpy as np

from ..schemas import BlocoExtraido, ResultadoExtracao

if TYPE_CHECKING:
    from rapidocr_onnxruntime import RapidOCR

NOME_EXTRATOR = "hibrido-pymupdf-rapidocr"

# Constantes (espelham _hibrido-img-base.ts)
LIMIAR_PIXELS = 100_000  # so OCR em imagens grandes
FRACAO_PARA_BACKGROUND = 0.5  # >= 50% das paginas = background
LIMIAR_OCR_MIN_CHARS = 5
LIMIAR_SOBREPOSICAO = 0.6  # frase com >= 60% sobreposta = duplicada
CONFIDENCE_MIN = 0.5  # 0..1 (TS usa 0..100, equivalente 50)


_OCR: "RapidOCR | None" = None


def _get_ocr() -> "RapidOCR":
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


# ============================================================
# Helpers de texto (port de normalizar/tokenizar/quebrarEmFrases)
# ============================================================


def _remover_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _normalizar(s: str) -> str:
    return _remover_acentos(s).lower()


_RE_TOKEN_SPLIT = re.compile(r"[^\w]+", flags=re.UNICODE)
_RE_FRASE_SPLIT = re.compile(r"[.!?\n\r]+")


def _tokenizar(s: str) -> list[str]:
    return [t for t in _RE_TOKEN_SPLIT.split(_normalizar(s)) if len(t) >= 2]


def _quebrar_em_frases(s: str) -> list[str]:
    return [f.strip() for f in _RE_FRASE_SPLIT.split(s) if f.strip()]


def _limpar_ocr_sobreposto(texto_ocr: str, texto_vetorial: str) -> str:
    """Pra cada frase OCR, se >= 60% das palavras ja estao no vetorial,
    descarta (provavel duplicacao). Retorna OCR limpo concatenado."""
    tokens_vetorial = set(_tokenizar(texto_vetorial))
    if not tokens_vetorial:
        return texto_ocr
    mantidas: list[str] = []
    for frase in _quebrar_em_frases(texto_ocr):
        tokens = _tokenizar(frase)
        if not tokens:
            continue
        sobrepostas = sum(1 for t in tokens if t in tokens_vetorial)
        if sobrepostas / len(tokens) < LIMIAR_SOBREPOSICAO:
            mantidas.append(frase)
    return " ".join(mantidas)


# ============================================================
# Listagem de imagens com bytes + hash
# ============================================================


def _listar_imagens_pagina(doc: fitz.Document, page: fitz.Page) -> list[dict]:
    """Lista imagens da pagina com bytes brutos + hash + dimensoes."""
    out: list[dict] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            im = doc.extract_image(xref)
        except Exception:
            continue
        bytes_img = im.get("image")
        if not bytes_img:
            continue
        h = hashlib.sha256(bytes_img).hexdigest()
        out.append(
            {
                "xref": xref,
                "ext": im.get("ext", "png"),
                "bytes": bytes_img,
                "width": im.get("width", 0),
                "height": im.get("height", 0),
                "pixels": im.get("width", 0) * im.get("height", 0),
                "hash": h,
            }
        )
    return out


def _detectar_hashes_background(imgs_por_pagina: list[list[dict]]) -> set[str]:
    """Hashes que aparecem em >= 50% das paginas viram background."""
    total = len(imgs_por_pagina)
    if total == 0:
        return set()
    counts: dict[str, int] = {}
    for imgs in imgs_por_pagina:
        vistos: set[str] = set()
        for im in imgs:
            h = im["hash"]
            if h in vistos:
                continue
            vistos.add(h)
            counts[h] = counts.get(h, 0) + 1
    limiar = total * FRACAO_PARA_BACKGROUND
    return {h for h, c in counts.items() if c >= limiar}


# ============================================================
# OCR de uma imagem
# ============================================================


def _bytes_imagem_para_bgr(bytes_img: bytes) -> np.ndarray | None:
    """Decodifica bytes da imagem (PNG/JPEG/etc) pra np.ndarray BGR."""
    arr = np.frombuffer(bytes_img, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _ocr_imagem(im: dict) -> str:
    """OCR de uma imagem. Filtra confidence e devolve texto concatenado."""
    bgr = _bytes_imagem_para_bgr(im["bytes"])
    if bgr is None:
        return ""
    ocr = _get_ocr()
    try:
        result, _elapsed = ocr(bgr)
    except Exception:
        return ""
    if not result:
        return ""
    linhas: list[str] = []
    for item in result:
        texto = (item[1] or "").strip()
        score = float(item[2]) if len(item) > 2 else 1.0
        if score < CONFIDENCE_MIN:
            continue
        if not texto:
            continue
        linhas.append(texto)
    return "\n".join(linhas)


# ============================================================
# Extrator principal
# ============================================================


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

    try:
        num_paginas = doc.page_count

        # 1. Texto vetorial por pagina (PyMuPDF)
        textos_vetorial: list[str] = []
        blocks_vetorial: list[BlocoExtraido] = []
        for page_idx, page in enumerate(doc):
            page_blocks = page.get_text("blocks", sort=True)
            partes: list[str] = []
            for bx in page_blocks:
                if len(bx) < 7:
                    continue
                x0, y0, x1, y1, texto_bloco, block_no, block_type = bx[:7]
                if block_type != 0:
                    continue
                t = (texto_bloco or "").strip()
                if not t:
                    continue
                partes.append(t)
                blocks_vetorial.append(
                    BlocoExtraido(
                        type="paragraph",
                        text=t,
                        page_idx=page_idx,
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                        block_id=f"vec-p{page_idx}-b{block_no}",
                    )
                )
            textos_vetorial.append("\n".join(partes))

        # 2. Lista imagens por pagina (com hash)
        imgs_por_pagina: list[list[dict]] = [
            _listar_imagens_pagina(doc, page) for page in doc
        ]
        hashes_background = _detectar_hashes_background(imgs_por_pagina)

        # 3. Junta imagens-alvo deduplicadas por hash, anotando em quais paginas aparecem
        alvos_por_hash: dict[str, dict] = {}
        for i, imgs in enumerate(imgs_por_pagina):
            for im in imgs:
                if im["pixels"] < LIMIAR_PIXELS:
                    continue
                if im["hash"] in hashes_background:
                    continue
                existente = alvos_por_hash.get(im["hash"])
                if existente:
                    existente["paginas"].append(i)
                else:
                    alvos_por_hash[im["hash"]] = {
                        "imagem": im,
                        "paginas": [i],
                    }

        # 4. OCR (com cache implicito por dedup de hash) e dedup vs vetorial
        textos_finais = list(textos_vetorial)
        blocks_ocr: list[BlocoExtraido] = []
        for hash_, alvo in alvos_por_hash.items():
            texto_bruto = _ocr_imagem(alvo["imagem"])
            if len(texto_bruto) < LIMIAR_OCR_MIN_CHARS:
                continue
            for page_idx in alvo["paginas"]:
                limpo = _limpar_ocr_sobreposto(texto_bruto, textos_vetorial[page_idx])
                if len(limpo) < LIMIAR_OCR_MIN_CHARS:
                    continue
                textos_finais[page_idx] = (
                    f"{textos_finais[page_idx]}\n{limpo}"
                    if textos_finais[page_idx]
                    else limpo
                )
                blocks_ocr.append(
                    BlocoExtraido(
                        type="paragraph",
                        text=limpo,
                        page_idx=page_idx,
                        block_id=f"ocr-p{page_idx}-{hash_[:8]}",
                    )
                )

        texto = "\n\n".join(textos_finais)
    finally:
        doc.close()

    return ResultadoExtracao(
        extrator=NOME_EXTRATOR,
        texto=texto,
        blocks=blocks_vetorial + blocks_ocr,
        markdown="",
        num_paginas=num_paginas,
        num_caracteres=len(texto),
        tempo_segundos=round(time.perf_counter() - inicio, 3),
        erro=None,
    )
