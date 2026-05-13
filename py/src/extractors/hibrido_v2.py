"""
Hibrido v2 — pipeline plugavel pra testar diferentes matchers.

Sprint 7a: experimento pra comparar 4 estrategias de matching entre
texto vetorial (PyMuPDF) e regioes do layout (DocLayout-YOLO):

  1. best_0       — best match SEM threshold (= hibrido_layout_aware atual)
  2. best_02      — best match com threshold IoS 0.2 (estilo Docling)
  3. greedy_05    — first-claim greedy com threshold 0.5 (estilo MinerU)
  4. confidence   — threshold dinamico baseado em YOLO confidence

Cada matcher exporta `extrair_*` que chama _extrair_com_matcher().
Pra comparar: adicione todos os 4 ao benchmark.py e rode normalmente.

Reusa muito do hibrido_layout_aware atual (extracao vetorial, OCR de
regioes figure, hierarquia de listas, refinos de tipo). Soh muda a
funcao de cruzamento bbox vetorial × regiao YOLO.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Callable

import cv2
import fitz
import numpy as np

from ..schemas import BlocoExtraido, ResultadoExtracao
from . import layout_detector
from .layout_detector import RegiaoLayout, classe_para_tipo_bloco

if TYPE_CHECKING:
    from rapidocr_onnxruntime import RapidOCR


# ============================================================
# Constantes (reusadas do hibrido_layout_aware)
# ============================================================

DPI_RENDER = 150
CONFIDENCE_OCR_MIN = 0.5
LIMIAR_OCR_MIN_CHARS = 5
MIN_FIGURE_WIDTH = 200
MIN_FIGURE_HEIGHT = 150
TIPOS_FORA_DO_TEXTO = {"header", "footer", "page_number"}

# Thresholds pra niveis de titulo
TITULO_LEVEL_1 = 1.5
TITULO_LEVEL_2 = 1.25
TITULO_LEVEL_3 = 1.1

HEADER_Y_MAX = 0.15
FOOTER_Y_MIN = 0.85

# Regex pra list_item (mesma do hibrido_layout_aware)
RE_LIST_ITEM = re.compile(
    r"^[\s ]*("
    r"[-–—•·●○◦▪▫►▸→❯>]"
    r"|\d+[.)]"
    r"|[a-zA-Z][.)]"
    r"|\([a-zA-Z0-9]{1,3}\)"
    r")\s+"
)

# Cache OCR + singleton
_OCR_CACHE: dict[str, str] = {}
_OCR: "RapidOCR | None" = None


# ============================================================
# Helpers (port do hibrido_layout_aware)
# ============================================================


def _get_ocr() -> "RapidOCR":
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def _limpar_texto(s: str) -> str:
    return s.replace(" ", " ").replace(" ", " ")


def _deduplica_linhas_consecutivas(texto: str) -> str:
    if not texto:
        return texto
    linhas = texto.split("\n")
    out: list[str] = []
    ultima = None
    for ln in linhas:
        ln_strip = ln.strip()
        if ln_strip and ln_strip == ultima:
            continue
        out.append(ln)
        ultima = ln_strip
    return "\n".join(out)


def _eh_list_item(texto: str) -> bool:
    return bool(RE_LIST_ITEM.match(texto))


def _refinar_tipo_abandon(
    bbox_img: tuple[float, float, float, float],
    img_height_px: int,
) -> str:
    if img_height_px <= 0:
        return "header"
    y_centro = (bbox_img[1] + bbox_img[3]) / 2
    if y_centro < img_height_px * HEADER_Y_MAX:
        return "header"
    if y_centro > img_height_px * FOOTER_Y_MIN:
        return "footer"
    return "page_number"


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


def _intersecao_relativa(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoS: area_intersect(a,b) / area(a). 0..1"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1.0)
    return inter / area_a


def _converter_bbox_pdf_para_img(
    bbox_pdf: tuple[float, float, float, float],
    page_width_pt: float,
    page_height_pt: float,
    img_width_px: int,
    img_height_px: int,
) -> tuple[float, float, float, float]:
    sx = img_width_px / page_width_pt
    sy = img_height_px / page_height_pt
    x1, y1, x2, y2 = bbox_pdf
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def _extrair_blocos_vetorial(page: fitz.Page) -> list[dict]:
    """Devolve blocos vetoriais com bbox + texto + font_size mediano."""
    d = page.get_text("dict")
    spans_info: list[dict] = []
    for bx in d.get("blocks", []):
        if bx.get("type") != 0:
            continue
        for line in bx.get("lines", []):
            for span in line.get("spans", []):
                sz = span.get("size")
                bbox_span = span.get("bbox")
                if sz is None or not bbox_span or len(bbox_span) < 4:
                    continue
                spans_info.append({"bbox": tuple(bbox_span[:4]), "size": float(sz)})

    blocos = []
    for bx in page.get_text("blocks", sort=True):
        if len(bx) < 7:
            continue
        x0, y0, x1, y1, texto, block_no, block_type = bx[:7]
        if block_type != 0:
            continue
        texto = _limpar_texto((texto or "").strip())
        if not texto:
            continue
        sizes_no_bloco = [
            s["size"] for s in spans_info
            if x0 - 1 <= s["bbox"][0] and s["bbox"][2] <= x1 + 1
            and y0 - 1 <= s["bbox"][1] and s["bbox"][3] <= y1 + 1
        ]
        font_size = median(sizes_no_bloco) if sizes_no_bloco else 12.0
        blocos.append(
            {
                "bbox_pdf": (float(x0), float(y0), float(x1), float(y1)),
                "text": texto,
                "block_no": int(block_no),
                "font_size": float(font_size),
            }
        )
    return blocos


def _calcular_nivel_titulo(font_size: float, font_size_mediano: float) -> int:
    if font_size_mediano <= 0:
        return 2
    razao = font_size / font_size_mediano
    if razao >= TITULO_LEVEL_1:
        return 1
    if razao >= TITULO_LEVEL_2:
        return 2
    if razao >= TITULO_LEVEL_3:
        return 3
    return 4


def _hash_recorte(recorte: np.ndarray) -> str:
    return hashlib.sha256(recorte.tobytes()).hexdigest()


def _ocr_recorte_cached(recorte: np.ndarray, hash_: str) -> str:
    if hash_ in _OCR_CACHE:
        return _OCR_CACHE[hash_]
    ocr = _get_ocr()
    try:
        resultado, _ = ocr(recorte)
    except Exception:
        _OCR_CACHE[hash_] = ""
        return ""
    if not resultado:
        _OCR_CACHE[hash_] = ""
        return ""
    linhas: list[str] = []
    for item in resultado:
        score = float(item[2]) if len(item) > 2 else 1.0
        if score < CONFIDENCE_OCR_MIN:
            continue
        t = (item[1] or "").strip()
        if t:
            linhas.append(t)
    texto = _limpar_texto("\n".join(linhas))
    _OCR_CACHE[hash_] = texto
    return texto


def _calcular_niveis_listas(blocos_list_item: list[BlocoExtraido]) -> dict[str, int]:
    if not blocos_list_item:
        return {}
    xs_por_id = {b.block_id: b.bbox[0] if b.bbox else 0.0 for b in blocos_list_item}
    if len(set(xs_por_id.values())) <= 1:
        return {bid: 0 for bid in xs_por_id}
    x_min = min(xs_por_id.values())
    xs_distintos = sorted(set(xs_por_id.values()))
    diffs = [xs_distintos[i + 1] - xs_distintos[i] for i in range(len(xs_distintos) - 1)]
    diffs_significativos = [d for d in diffs if d > 5]
    passo = median(diffs_significativos) if diffs_significativos else 20.0
    niveis = {}
    for bid, x in xs_por_id.items():
        nivel = round((x - x_min) / passo)
        nivel = max(0, min(nivel, 5))
        niveis[bid] = nivel
    return niveis


def _gerar_markdown_pagina(blocos: list[BlocoExtraido], niveis_lista: dict[str, int]) -> list[str]:
    partes: list[str] = []
    for b in blocos:
        if b.type in TIPOS_FORA_DO_TEXTO:
            continue
        texto = b.text
        if b.type == "title":
            level = b.level if b.level else 1
            partes.append("#" * level + " " + texto)
        elif b.type == "list_item":
            indent = "  " * niveis_lista.get(b.block_id or "", 0)
            partes.append(f"{indent}- {texto}")
        elif b.type == "table":
            partes.append(f"```\n{texto}\n```")
        elif b.type == "formula":
            partes.append(f"$$\n{texto}\n$$")
        elif b.type == "caption":
            partes.append(f"_{texto}_")
        elif b.type == "image":
            if texto and len(texto) >= 30:
                partes.append(texto)
        else:
            partes.append(texto)
    return partes


# ============================================================
# MATCHERS (interface comum: recebe bloco + regioes, devolve regiao escolhida)
# ============================================================

# Tipo do matcher: (bloco_dict, lista_regioes, img_height_px) -> RegiaoLayout | None
MatcherFn = Callable[[dict, list[RegiaoLayout], int], "RegiaoLayout | None"]


def matcher_best_0(bloco: dict, regioes: list[RegiaoLayout], img_h: int) -> "RegiaoLayout | None":
    """Best match SEM threshold (= hibrido_layout_aware atual)."""
    melhor = None
    melhor_overlap = 0.0
    for r in regioes:
        overlap = _intersecao_relativa(bloco["bbox_img"], r.bbox)
        if overlap > melhor_overlap:
            melhor_overlap = overlap
            melhor = r
    if melhor and melhor_overlap > 0:
        return melhor
    return None


def matcher_best_02(bloco: dict, regioes: list[RegiaoLayout], img_h: int) -> "RegiaoLayout | None":
    """Best match com threshold IoS 0.2 (estilo Docling)."""
    melhor = None
    melhor_overlap = 0.0
    for r in regioes:
        overlap = _intersecao_relativa(bloco["bbox_img"], r.bbox)
        if overlap > melhor_overlap:
            melhor_overlap = overlap
            melhor = r
    if melhor and melhor_overlap >= 0.20:  # ← THRESHOLD DOCLING
        return melhor
    return None


# Estado global pro greedy (reseta entre paginas via funcao auxiliar)
_GREEDY_USED_BLOCKS: set[int] = set()


def matcher_greedy_05(bloco: dict, regioes: list[RegiaoLayout], img_h: int) -> "RegiaoLayout | None":
    """Greedy first-claim com threshold IoS 0.5 (estilo MinerU).

    ATENCAO: comportamento depende da ORDEM dos blocos (e do greedy state).
    Implementacao "best match com threshold 0.5" eh a aproximacao mais simples
    sem precisar reverter a logica do pipeline (que itera por blocos vetoriais).
    """
    melhor = None
    melhor_overlap = 0.0
    for r in regioes:
        overlap = _intersecao_relativa(bloco["bbox_img"], r.bbox)
        if overlap > melhor_overlap:
            melhor_overlap = overlap
            melhor = r
    if melhor and melhor_overlap > 0.50:  # ← THRESHOLD MINERU
        return melhor
    return None


def matcher_confidence_aware(bloco: dict, regioes: list[RegiaoLayout], img_h: int) -> "RegiaoLayout | None":
    """Threshold dinamico baseado em YOLO confidence.

    Regiao com confidence alta -> permissivo (threshold 0.10)
    Regiao com confidence media -> medio (threshold 0.30)
    Regiao com confidence baixa -> rigido (threshold 0.50)

    Intuicao: se YOLO esta confiante na regiao, a gente confia mais e baixa
    a barra. Se YOLO esta inseguro, exige prova mais forte.
    """
    melhor = None
    melhor_overlap = 0.0
    melhor_threshold = 0.50
    for r in regioes:
        overlap = _intersecao_relativa(bloco["bbox_img"], r.bbox)
        # Threshold dinamico por confidence da regiao
        if r.confidence > 0.7:
            thresh = 0.10
        elif r.confidence > 0.4:
            thresh = 0.30
        else:
            thresh = 0.50
        # So considera a regiao se passou o threshold dela
        if overlap > thresh and overlap > melhor_overlap:
            melhor_overlap = overlap
            melhor = r
            melhor_threshold = thresh
    if melhor:
        return melhor
    return None


# ============================================================
# Score composto (#2) — combina IoS com sinais de dominio
# ============================================================

# Pesos pra cada sinal (calibraveis depois via grid search)
PESO_IOS = 1.0           # geometria pura (base)
PESO_FONT_TITLE = 0.30   # bonus pra title se font grande
PESO_REGEX_LIST = 0.40   # bonus pra list_item se tem bullet
PESO_POSITION = 0.20     # bonus pra header/footer pelo Y
PENALIDADE_TITLE_SEM_FONT = 0.30  # penalidade se title MAS font normal
PENALIDADE_NUMERICO_NAO_PG = 0.20  # penalidade se text e numero E tipo nao e page_number

# Threshold minimo do score combinado pra aceitar match
SCORE_MIN_THRESHOLD = 0.30


def _bonus_font(bloco: dict, classe: str, font_mediano: float) -> float:
    """Bonus pra title se font_size grande, penalidade se title com font normal."""
    if classe != "title":
        return 0.0
    if font_mediano <= 0:
        return 0.0
    razao = bloco.get("font_size", 12.0) / font_mediano
    if razao >= 1.25:
        return PESO_FONT_TITLE        # title justificado
    if razao >= 1.10:
        return PESO_FONT_TITLE * 0.5  # title borderline
    return -PENALIDADE_TITLE_SEM_FONT # title com font normal — suspeito


def _bonus_regex(bloco: dict, classe: str) -> float:
    """Bonus pra list_item/list se text comeca com bullet/numero."""
    if classe not in ("list_item", "list", "plain_text"):
        return 0.0
    if _eh_list_item(bloco["text"]):
        if classe in ("list_item", "list"):
            return PESO_REGEX_LIST           # list_item com bullet — alinhado
        else:
            return PESO_REGEX_LIST * 0.3     # bullet em plain_text — fraco bonus
    return 0.0


def _bonus_position(bloco: dict, classe: str, img_h: int) -> float:
    """Bonus pra header/footer/page_number baseado em Y do bloco."""
    if classe != "abandon":
        return 0.0
    if img_h <= 0:
        return 0.0
    bbox_img = bloco.get("bbox_img")
    if not bbox_img:
        return 0.0
    y_centro = (bbox_img[1] + bbox_img[3]) / 2
    razao_y = y_centro / img_h
    # abandon eh header/footer — confirma se esta nas extremidades
    if razao_y < HEADER_Y_MAX or razao_y > FOOTER_Y_MIN:
        return PESO_POSITION
    return 0.0


def _penalidade_numerico_isolado(bloco: dict, classe: str) -> float:
    """Texto numerico isolado (ex: '5', '12') tem penalidade pra tipos
    nao-page_number. Provavel paginacao."""
    text = bloco["text"].strip()
    if len(text) > 5:
        return 0.0
    # so digitos + espacos + barras (ex: "5", "12/24", "Pg 3")
    eh_numerico = all(c in "0123456789 /" for c in text)
    if not eh_numerico:
        return 0.0
    if classe in ("page_number", "abandon", "footer"):
        return 0.0  # ok, eh natural pra esses tipos
    return -PENALIDADE_NUMERICO_NAO_PG


def _score_composto(
    bloco: dict, regiao: RegiaoLayout, font_mediano: float, img_h: int
) -> float:
    """Score combinado: geometria + dominio."""
    ios = _intersecao_relativa(bloco["bbox_img"], regiao.bbox)
    score = PESO_IOS * ios

    # Sinais de dominio só importam se ja houve algum overlap geometrico
    if ios > 0:
        score += _bonus_font(bloco, regiao.classe, font_mediano)
        score += _bonus_regex(bloco, regiao.classe)
        score += _bonus_position(bloco, regiao.classe, img_h)
        score += _penalidade_numerico_isolado(bloco, regiao.classe)

    return score


def matcher_score_composto(bloco: dict, regioes: list[RegiaoLayout], img_h: int) -> "RegiaoLayout | None":
    """Best match usando SCORE COMPOSTO (IoS + font + regex + position)."""
    # font_mediano vem do contexto — aproximacao: usar 12.0 padrao se nao
    # tiver. (Em producao seria injetado do pipeline; aqui simplifico.)
    font_mediano = bloco.get("_font_mediano_pagina", 12.0)

    melhor = None
    melhor_score = 0.0
    for r in regioes:
        s = _score_composto(bloco, r, font_mediano, img_h)
        if s > melhor_score:
            melhor_score = s
            melhor = r
    if melhor and melhor_score >= SCORE_MIN_THRESHOLD:
        return melhor
    return None


# Registry: nome -> (matcher_fn, descricao)
MATCHERS: dict[str, tuple[MatcherFn, str]] = {
    "best_0": (matcher_best_0, "Best match SEM threshold (atual hibrido_layout_aware)"),
    "best_02": (matcher_best_02, "Best match com threshold IoS 0.2 (Docling-style)"),
    "greedy_05": (matcher_greedy_05, "Best match com threshold IoS 0.5 (MinerU-style)"),
    "confidence": (matcher_confidence_aware, "Threshold dinamico por YOLO confidence (0.10 / 0.30 / 0.50)"),
    "score_composto": (matcher_score_composto, "Score IoS + font_size + regex bullet + posicao Y (+penalidades)"),
}


# ============================================================
# Pipeline principal — recebe matcher como parametro
# ============================================================


def _extrair_com_matcher(
    caminho_pdf: str | Path,
    matcher_name: str,
) -> ResultadoExtracao:
    """Pipeline plugavel: recebe nome do matcher e usa ele pra cruzamento."""
    if matcher_name not in MATCHERS:
        raise ValueError(f"Matcher desconhecido: {matcher_name}. Opcoes: {list(MATCHERS)}")
    matcher_fn, _ = MATCHERS[matcher_name]
    nome_extrator = f"hibrido-v2-{matcher_name}"

    inicio = time.perf_counter()
    caminho = Path(caminho_pdf)

    try:
        doc = fitz.open(caminho)
    except Exception as e:
        return ResultadoExtracao(
            extrator=nome_extrator,
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

            try:
                img_bgr = _renderizar_pagina_bgr(page)
                regioes = layout_detector.detectar(img_bgr)
            except Exception:
                regioes = []
                img_bgr = None

            img_h, img_w = (img_bgr.shape[:2] if img_bgr is not None else (0, 0))

            blocos_vetorial = _extrair_blocos_vetorial(page)
            for b in blocos_vetorial:
                if img_w > 0:
                    b["bbox_img"] = _converter_bbox_pdf_para_img(
                        b["bbox_pdf"], page_width_pt, page_height_pt, img_w, img_h
                    )
                else:
                    b["bbox_img"] = b["bbox_pdf"]

            font_sizes = [b["font_size"] for b in blocos_vetorial if b["font_size"]]
            font_size_mediano = median(font_sizes) if font_sizes else 12.0

            # Anota mediano em cada bloco pra matchers que precisam (score_composto)
            for b in blocos_vetorial:
                b["_font_mediano_pagina"] = font_size_mediano

            blocos_pagina: list[BlocoExtraido] = []
            blocos_list_item_da_pagina: list[BlocoExtraido] = []

            # ============================================================
            # MATCHING (a parte que muda entre matchers)
            # ============================================================
            for vi, b in enumerate(blocos_vetorial):
                regiao_escolhida = matcher_fn(b, regioes, img_h)

                if regiao_escolhida:
                    if regiao_escolhida.classe == "abandon":
                        tipo = _refinar_tipo_abandon(regiao_escolhida.bbox, img_h)
                    else:
                        tipo = classe_para_tipo_bloco(regiao_escolhida.classe)
                else:
                    tipo = "paragraph"

                # A4: regex sobrescreve pra list_item
                if _eh_list_item(b["text"]):
                    tipo = "list_item"

                # A2: nivel pra title via font_size
                level: int | None = None
                if tipo == "title":
                    level = _calcular_nivel_titulo(b["font_size"], font_size_mediano)

                bloco = BlocoExtraido(
                    type=tipo,
                    text=b["text"],
                    page_idx=page_idx,
                    bbox=b["bbox_pdf"],
                    block_id=f"p{page_idx}-vec{vi}",
                    level=level,
                )
                blocos_pagina.append(bloco)
                if tipo == "list_item":
                    blocos_list_item_da_pagina.append(bloco)

            # ============================================================
            # OCR pra figure/table/formula sem texto vetorial dentro
            # ============================================================
            if img_bgr is not None:
                for ri, regiao in enumerate(regioes):
                    if regiao.classe not in ("figure", "table", "isolate_formula"):
                        continue
                    tem_vetorial = any(
                        _intersecao_relativa(b["bbox_img"], regiao.bbox) > 0.30
                        for b in blocos_vetorial
                    )
                    if tem_vetorial:
                        continue
                    x1, y1, x2, y2 = (int(round(v)) for v in regiao.bbox)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(img_w, x2); y2 = min(img_h, y2)
                    largura = x2 - x1
                    altura = y2 - y1
                    if largura < 10 or altura < 10:
                        continue
                    recorte = img_bgr[y1:y2, x1:x2].copy()
                    h = _hash_recorte(recorte)

                    if regiao.classe == "table":
                        tipo_regiao = "table"
                    elif regiao.classe == "isolate_formula":
                        tipo_regiao = "formula"
                    else:
                        tipo_regiao = "image"

                    eh_figure_pequena = (
                        tipo_regiao == "image"
                        and (largura < MIN_FIGURE_WIDTH or altura < MIN_FIGURE_HEIGHT)
                    )
                    if eh_figure_pequena:
                        continue

                    texto_ocr = _deduplica_linhas_consecutivas(_ocr_recorte_cached(recorte, h))

                    if tipo_regiao in ("table", "formula") and (
                        not texto_ocr or len(texto_ocr) < LIMIAR_OCR_MIN_CHARS
                    ):
                        continue

                    blocos_pagina.append(
                        BlocoExtraido(
                            type=tipo_regiao,
                            text=texto_ocr,
                            page_idx=page_idx,
                            bbox=regiao.bbox,
                            block_id=f"p{page_idx}-ocr{ri}",
                        )
                    )

            niveis_lista = _calcular_niveis_listas(blocos_list_item_da_pagina)

            blocos_pagina.sort(
                key=lambda b: (
                    round((b.bbox[1] if b.bbox else 0) / 50) * 50,
                    b.bbox[0] if b.bbox else 0,
                )
            )

            blocks_finais.extend(blocos_pagina)

            partes_texto_pagina = [
                b.text for b in blocos_pagina if b.type not in TIPOS_FORA_DO_TEXTO and b.text
            ]
            paginas_texto.append("\n".join(partes_texto_pagina))

            partes_md = _gerar_markdown_pagina(blocos_pagina, niveis_lista)
            if partes_md:
                md_partes.append("\n\n".join(partes_md))

        texto = "\n\n".join(paginas_texto)
        markdown = "\n\n---\n\n".join(md_partes)
    finally:
        doc.close()

    return ResultadoExtracao(
        extrator=nome_extrator,
        texto=texto,
        blocks=blocks_finais,
        markdown=markdown,
        num_paginas=num_paginas,
        num_caracteres=len(texto),
        tempo_segundos=round(time.perf_counter() - inicio, 3),
        erro=None,
    )


# ============================================================
# Wrappers publicos (1 por matcher) pra usar no benchmark
# ============================================================


def extrair_best_0(caminho_pdf: str | Path) -> ResultadoExtracao:
    return _extrair_com_matcher(caminho_pdf, "best_0")


def extrair_best_02(caminho_pdf: str | Path) -> ResultadoExtracao:
    return _extrair_com_matcher(caminho_pdf, "best_02")


def extrair_greedy_05(caminho_pdf: str | Path) -> ResultadoExtracao:
    return _extrair_com_matcher(caminho_pdf, "greedy_05")


def extrair_confidence(caminho_pdf: str | Path) -> ResultadoExtracao:
    return _extrair_com_matcher(caminho_pdf, "confidence")


def extrair_score_composto(caminho_pdf: str | Path) -> ResultadoExtracao:
    return _extrair_com_matcher(caminho_pdf, "score_composto")
