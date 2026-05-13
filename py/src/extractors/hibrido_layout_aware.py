"""
Extrator hibrido layout-aware: PyMuPDF + DocLayout-YOLO + RapidOCR seletivo.

REFATORADO no Sprint 6 com a logica VETORIAL-FIRST:

  - Texto vetorial do PyMuPDF eh SEMPRE preservado no output
  - Pra cada bloco vetorial: acha regiao YOLO com MAIOR sobreposicao (sem
    threshold fixo) e atribui o tipo dessa regiao
  - Se nenhuma regiao sobrepoe (sobreposicao = 0%) -> tipo "paragraph"
  - Regioes YOLO "figure"/"table"/"isolate_formula" SEM texto vetorial
    dentro -> OCR no recorte da imagem (com cache por hash)

Melhorias do Sprint 6:
  A1 - Fix   e   (hard spaces) -> espaco normal
  A2 - Detecta nivel de titulo (#, ##, ###) via font_size do PyMuPDF
  A3 - Refina classe "abandon" do YOLO em header/footer/page_number pelo Y
  A4 - Detecta list_item por regex de bullet/numero
  B  - Hierarquia de listas (nesting) via X do bbox
  C  - Logica vetorial-first sem threshold (acima)
  D1 - Cache OCR por hash SHA-256 do recorte (sem regra de descarte)
  D2 - Salva imagens dos recortes + ![image](path) no markdown
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from statistics import median
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

# ============================================================
# Constantes
# ============================================================

DPI_RENDER = 150
CONFIDENCE_OCR_MIN = 0.5
LIMIAR_OCR_MIN_CHARS = 5

# Filtro: regioes "figure" menores que isso nao geram OCR text
# (provavel falso positivo do YOLO marcando label/caixinha dentro de diagrama)
MIN_FIGURE_WIDTH = 200
MIN_FIGURE_HEIGHT = 150

# Tipos que nao entram no texto puro (so nos blocks)
TIPOS_FORA_DO_TEXTO = {"header", "footer", "page_number"}

# Diretorio padrao pra salvar imagens dos recortes (Fase D2).
# Path relativo a raiz do projeto: <projeto>/resultados/structured-jsons/<extrator>/<pdf>_images/
RAIZ_PROJETO = Path(__file__).resolve().parents[3]
DIR_BASE_IMAGES = RAIZ_PROJETO / "resultados" / "structured-jsons" / NOME_EXTRATOR

# Regex pra detectar list_item (bullets unicode + numeros + letras alfanumericos)
RE_LIST_ITEM = re.compile(
    r"^[\s ]*("
    r"[-–—•·●○◦▪▫►▸→❯>]"           # bullets variados
    r"|\d+[.)]"                       # 1. ou 1)
    r"|[a-zA-Z][.)]"                  # a. ou A)
    r"|\([a-zA-Z0-9]{1,3}\)"          # (1) (a) (iv)
    r")\s+"
)

# Thresholds pra nivel de titulo via font_size (multiplicador do font_size mediano da pagina)
TITULO_LEVEL_1 = 1.5    # font >= 1.5x mediano -> # (H1)
TITULO_LEVEL_2 = 1.25   # font >= 1.25x mediano -> ## (H2)
TITULO_LEVEL_3 = 1.1    # font >= 1.1x mediano -> ### (H3)

# Thresholds pra header/footer (porcao da altura da pagina)
HEADER_Y_MAX = 0.15     # se y_centro < 15% -> header
FOOTER_Y_MIN = 0.85     # se y_centro > 85% -> footer

# Cache OCR (Fase D1) — global pra reuso entre PDFs no mesmo processo
_OCR_CACHE: dict[str, str] = {}
_OCR: "RapidOCR | None" = None


def _get_ocr() -> "RapidOCR":
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


# ============================================================
# Helpers de texto e geometria
# ============================================================


def _limpar_texto(s: str) -> str:
    """A1: troca hard spaces (U+00A0, U+202F) por espaco normal."""
    return s.replace(" ", " ").replace(" ", " ")


def _deduplica_linhas_consecutivas(texto: str) -> str:
    """Remove linhas duplicadas consecutivas (efeito sombra: 'Central\\nCentral')."""
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


# Sprint 9 item 2: regex pra remover bullet do INICIO do texto vetorial
# (evita duplicacao quando o markdown gera "- " e o texto original ja tinha "•")
RE_STRIP_BULLET_INICIAL = re.compile(
    r"^[\s ]*("
    r"[-–—•·●○◦▪▫►▸→❯>]"
    r"|\d+[.)]"
    r"|[a-zA-Z][.)]"
    r"|\([a-zA-Z0-9]{1,3}\)"
    r")\s+"
)


def _strip_bullet_inicial(texto: str) -> str:
    """Sprint 9 item 2: remove bullet/numero/letra do inicio do texto."""
    return RE_STRIP_BULLET_INICIAL.sub("", texto, count=1)


# Sprint 9 item 2 (refino): bullets INLINE (no meio do texto) viram nova linha
# Pra blocos list_item que tem varios items concatenados (ex: "Item1 • Item2 • Item3")
RE_BULLET_INLINE = re.compile(r"\s+([•·●○◦▪▫►▸→❯])\s+")


def _separar_bullets_inline(texto: str) -> str:
    """Substitui bullets inline por quebra de linha (cria multiplos list items)."""
    # Insere \n antes de cada bullet inline pra serem renderizados como sub-itens
    return RE_BULLET_INLINE.sub(r"\n\1 ", texto)


def _normalizar_quebras_intra_paragrafo(texto: str) -> str:
    """Sprint 9 item 3: substitui \\n isolada (intra-paragrafo) por espaco.

    Mantem \\n\\n (separador real de paragrafos). Multiplos espacos viram 1.
    """
    if not texto:
        return texto
    # Marca \n\n com sentinela
    SENT = "\x00"
    # \n+ com 2 ou mais → mantem como separador (vira 1 sentinela)
    texto = re.sub(r"\n\s*\n+", SENT, texto)
    # \n unica → espaco
    texto = texto.replace("\n", " ")
    # Restaura sentinela como \n\n
    texto = texto.replace(SENT, "\n\n")
    # Multiplos espacos → 1
    texto = re.sub(r" +", " ", texto)
    return texto.strip()


def _eh_list_item(texto: str) -> bool:
    """A4: detecta se o texto comeca com bullet/numero de lista."""
    return bool(RE_LIST_ITEM.match(texto))


def _refinar_tipo_abandon(
    bbox_img: tuple[float, float, float, float],
    img_height_px: int,
) -> str:
    """A3: classe 'abandon' do YOLO -> header / footer / page_number pelo Y."""
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
    """Area da intersecao(a,b) / area(a). Mede quanto de A esta dentro de B."""
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


# ============================================================
# Extracao de blocos vetoriais com font_size (A2)
# ============================================================


def _extrair_blocos_vetorial(page: fitz.Page) -> list[dict]:
    """Devolve blocos vetoriais com bbox, texto, block_no E font_size mediano.

    Usa modo 'blocks' (com agrupamento heuristico de paragrafos) como
    estrutura primaria, e cruza com 'dict' (spans) apenas pra inferir
    font_size + font_flags (bold/italic) de cada bloco.
    """
    # 1. Spans com font_size E font_flags (do modo "dict")
    # font_flags bits: 1=italic, 4=bold (PyMuPDF docs)
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
                spans_info.append({
                    "bbox": tuple(bbox_span[:4]),
                    "size": float(sz),
                    "flags": int(span.get("flags", 0)),
                })

    # 2. Blocos agrupados (modo "blocks") — estrutura principal
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

        # Acha font_size + flags dos spans cujo bbox cai dentro deste bloco
        spans_no_bloco = [
            s
            for s in spans_info
            if x0 - 1 <= s["bbox"][0] and s["bbox"][2] <= x1 + 1
            and y0 - 1 <= s["bbox"][1] and s["bbox"][3] <= y1 + 1
        ]
        sizes = [s["size"] for s in spans_no_bloco]
        font_size = median(sizes) if sizes else 12.0

        # Sprint 9 item 4: detecta bold/italic pelo flags da MAIORIA dos spans
        # Threshold conservador (80%) pra evitar marcar paragrafos inteiros
        # como bold quando so algumas palavras sao destacadas
        n_spans = max(1, len(spans_no_bloco))
        n_bold = sum(1 for s in spans_no_bloco if s["flags"] & 4)
        n_italic = sum(1 for s in spans_no_bloco if s["flags"] & 1)
        is_bold = n_bold >= n_spans * 0.80
        is_italic = n_italic >= n_spans * 0.80

        blocos.append(
            {
                "bbox_pdf": (float(x0), float(y0), float(x1), float(y1)),
                "text": texto,
                "block_no": int(block_no),
                "font_size": float(font_size),
                "is_bold": is_bold,
                "is_italic": is_italic,
            }
        )
    return blocos


def _calcular_nivel_titulo(font_size: float, font_size_mediano: float) -> int:
    """A2: dado font_size do bloco e mediano da pagina, devolve level 1-6."""
    if font_size_mediano <= 0:
        return 2
    razao = font_size / font_size_mediano
    if razao >= TITULO_LEVEL_1:
        return 1
    if razao >= TITULO_LEVEL_2:
        return 2
    if razao >= TITULO_LEVEL_3:
        return 3
    return 4  # title detectado por YOLO mas font normal -> H4


# ============================================================
# OCR com cache (D1) e salvamento de imagem (D2)
# ============================================================


def _hash_recorte(recorte: np.ndarray) -> str:
    """Hash SHA-256 dos bytes do recorte (pra cache OCR)."""
    return hashlib.sha256(recorte.tobytes()).hexdigest()


def _ocr_recorte_cached(
    recorte: np.ndarray,
    hash_recorte: str,
) -> str:
    """OCR com cache por hash. Se hash ja foi processado, reusa."""
    if hash_recorte in _OCR_CACHE:
        return _OCR_CACHE[hash_recorte]
    ocr = _get_ocr()
    try:
        resultado, _ = ocr(recorte)
    except Exception:
        _OCR_CACHE[hash_recorte] = ""
        return ""
    if not resultado:
        _OCR_CACHE[hash_recorte] = ""
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
    _OCR_CACHE[hash_recorte] = texto
    return texto


def _salvar_imagem_recorte(recorte: np.ndarray, hash_recorte: str, dir_imagens: Path) -> str:
    """D2: salva recorte como PNG. Devolve path relativo pra usar no markdown."""
    dir_imagens.mkdir(parents=True, exist_ok=True)
    nome = f"{hash_recorte[:16]}.png"
    path = dir_imagens / nome
    if not path.exists():
        cv2.imwrite(str(path), recorte)
    return f"images/{nome}"


# ============================================================
# Hierarquia de listas (B)
# ============================================================


def _calcular_niveis_listas(
    blocos_list_item: list[BlocoExtraido],
) -> dict[str, int]:
    """B: pra blocos list_item de UMA pagina, calcula nivel via x do bbox.

    Retorna: dict {block_id -> nivel (0-5)}
    """
    if not blocos_list_item:
        return {}

    # Extrai x_inicial (bbox[0]) de cada list_item
    xs_por_id = {b.block_id: b.bbox[0] if b.bbox else 0.0 for b in blocos_list_item}

    if len(set(xs_por_id.values())) <= 1:
        # Todos na mesma indentacao -> tudo nivel 0
        return {bid: 0 for bid in xs_por_id}

    x_min = min(xs_por_id.values())
    xs_distintos = sorted(set(xs_por_id.values()))

    # Infere "passo" como mediana das diferencas entre x consecutivos
    diffs = [xs_distintos[i + 1] - xs_distintos[i] for i in range(len(xs_distintos) - 1)]
    diffs_significativos = [d for d in diffs if d > 5]  # ignora ruido
    passo = median(diffs_significativos) if diffs_significativos else 20.0

    niveis = {}
    for bid, x in xs_por_id.items():
        nivel = round((x - x_min) / passo)
        nivel = max(0, min(nivel, 5))
        niveis[bid] = nivel
    return niveis


# ============================================================
# Geracao de markdown
# ============================================================


def _gerar_markdown_pagina(
    blocos: list[BlocoExtraido],
    niveis_lista: dict[str, int],
    style_map: dict[str, tuple[bool, bool]] | None = None,
) -> list[str]:
    """Gera lista de strings markdown pros blocos de uma pagina."""
    # Sprint 9: bold/italic DESATIVADOS — Pedro prefere texto limpo
    partes: list[str] = []
    for b in blocos:
        if b.type in TIPOS_FORA_DO_TEXTO:
            continue
        texto = b.text

        if b.type == "title":
            # Sprint 9 item 1: TODO titulo vira ## (igual Docling padrao)
            partes.append("## " + texto)
        elif b.type == "list_item":
            indent = "  " * niveis_lista.get(b.block_id or "", 0)
            # Sprint 9: se texto tem bullets inline (transformados em \n),
            # quebra em multiplos items separados (igual Docling)
            items = [ln.strip() for ln in texto.split("\n") if ln.strip()]
            for item in items:
                partes.append(f"{indent}- {item}")
        elif b.type == "table":
            # Item 5 desativado: image_path so no JSON pra nao poluir qexc
            partes.append(f"```\n{texto}\n```")
        elif b.type == "formula":
            partes.append(f"$$\n{texto}\n$$")
        elif b.type == "caption":
            partes.append(texto)
        elif b.type == "image":
            # Item 5 desativado: image_path mantido no JSON (RAG),
            # mas NAO inserido no markdown (poluia qexc com hashes).
            # Texto OCR substancial vira paragrafo.
            if texto and len(texto) >= 30:
                partes.append(texto)
        else:
            # paragraph e outros tipos genericos
            partes.append(texto)
    return partes


# ============================================================
# Funcao principal
# ============================================================


def extrair(caminho_pdf: str | Path) -> ResultadoExtracao:
    inicio = time.perf_counter()
    caminho = Path(caminho_pdf)

    # Diretorio pra salvar imagens (D2)
    dir_imagens_pdf = DIR_BASE_IMAGES / f"{caminho.stem}_images"

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

            # 1. Render + 2. Layout (YOLO)
            try:
                img_bgr = _renderizar_pagina_bgr(page)
                regioes = layout_detector.detectar(img_bgr)
            except Exception:
                regioes = []
                img_bgr = None

            img_h, img_w = (img_bgr.shape[:2] if img_bgr is not None else (0, 0))

            # 3. Texto vetorial COMPLETO (com font_size pra A2)
            blocos_vetorial = _extrair_blocos_vetorial(page)

            # Converte bbox vetorial pra coords da imagem (pra cruzar com YOLO)
            for b in blocos_vetorial:
                if img_w > 0:
                    b["bbox_img"] = _converter_bbox_pdf_para_img(
                        b["bbox_pdf"], page_width_pt, page_height_pt, img_w, img_h
                    )
                else:
                    b["bbox_img"] = b["bbox_pdf"]

            # ============================================================
            # FASE C: Logica VETORIAL-FIRST
            # Pra cada bloco vetorial, acha regiao YOLO com MAIOR
            # sobreposicao (sem threshold). Se nenhuma sobrepoe -> paragraph.
            # ============================================================
            font_sizes = [b["font_size"] for b in blocos_vetorial if b["font_size"]]
            font_size_mediano = median(font_sizes) if font_sizes else 12.0

            blocos_pagina: list[BlocoExtraido] = []
            blocos_list_item_da_pagina: list[BlocoExtraido] = []
            # Sprint 8 item 2: dict paralelo com ordem natural do PDF (block_no)
            # pra reading order multi-coluna funcionar corretamente
            pdf_order_map: dict[str, int] = {}
            # Sprint 9 item 4: dict paralelo com style (bold, italic) por block_id
            style_map: dict[str, tuple[bool, bool]] = {}

            for vi, b in enumerate(blocos_vetorial):
                # Sprint 8 item 1: best_02 (threshold IoS >= 0.20, estilo Docling)
                # Acha regiao YOLO mais sobreposta com IoS >= 0.20
                melhor_regiao: RegiaoLayout | None = None
                melhor_overlap = 0.0
                for r in regioes:
                    overlap = _intersecao_relativa(b["bbox_img"], r.bbox)
                    if overlap > melhor_overlap:
                        melhor_overlap = overlap
                        melhor_regiao = r

                # Atribui tipo (threshold 0.20 — best_02 / Docling-style)
                if melhor_regiao and melhor_overlap >= 0.20:
                    if melhor_regiao.classe == "abandon":
                        tipo = _refinar_tipo_abandon(melhor_regiao.bbox, img_h)  # A3
                    else:
                        tipo = classe_para_tipo_bloco(melhor_regiao.classe)
                else:
                    tipo = "paragraph"

                # A4: regex de bullet sobrescreve pra list_item
                if _eh_list_item(b["text"]):
                    tipo = "list_item"

                # Sprint 8 item 3: tipo "image" com texto longo (> 100 chars)
                # eh provavel paragraph sobreposto a regiao figure — reclassifica
                if tipo == "image" and len(b["text"]) > 100:
                    tipo = "paragraph"

                # Sprint 9 item 1: TODO titulo vira level=1 (markdown ##),
                # mesma estrategia do Docling. Removida heuristica de
                # font_size que era inconsistente entre paginas.
                level: int | None = None
                if tipo == "title":
                    level = 1   # sempre H2 no markdown (## vem de level+1)

                # Sprint 9 item 2: strip bullet duplicado em list_item
                # ORDEM IMPORTA:
                #  1. strip bullet inicial
                #  2. normalizar quebras intra-paragrafo (\\n -> espaco)
                #  3. separar bullets inline em quebras (cria items markdown)
                texto_final = b["text"]
                if tipo == "list_item":
                    texto_final = _strip_bullet_inicial(texto_final)

                # Sprint 9 item 3: normaliza quebras de linha intra-paragrafo
                texto_final = _normalizar_quebras_intra_paragrafo(texto_final)

                if tipo == "list_item":
                    # Apos normalizar quebras, bullets inline ficam como ' • '
                    # Substituir por \\n cria items separados no markdown
                    texto_final = re.sub(r"\s+[•·●○◦▪▫►▸→❯]\s+", "\n", texto_final)

                block_id = f"p{page_idx}-vec{vi}"
                bloco = BlocoExtraido(
                    type=tipo,
                    text=texto_final,
                    page_idx=page_idx,
                    bbox=b["bbox_pdf"],
                    block_id=block_id,
                    level=level,
                )
                # Sprint 9 item 4: anota bold/italic pra usar no markdown
                # (campos extras nao sao do schema BlocoExtraido — guardamos no dict)
                style_map[block_id] = (b.get("is_bold", False), b.get("is_italic", False))
                blocos_pagina.append(bloco)
                pdf_order_map[block_id] = b["block_no"]   # Sprint 8 item 2
                if tipo == "list_item":
                    blocos_list_item_da_pagina.append(bloco)

            # ============================================================
            # OCR pra regioes "figure"/"table" SEM texto vetorial dentro
            # (D1: cache por hash; D2: salva imagem)
            # ============================================================
            if img_bgr is not None:
                for ri, regiao in enumerate(regioes):
                    if regiao.classe not in ("figure", "table", "isolate_formula"):
                        continue
                    # Tem algum bloco vetorial sobreposto?
                    tem_vetorial = any(
                        _intersecao_relativa(b["bbox_img"], regiao.bbox) > 0.30
                        for b in blocos_vetorial
                    )
                    if tem_vetorial:
                        continue

                    # Recorta + hash + cache OCR
                    x1, y1, x2, y2 = (int(round(v)) for v in regiao.bbox)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(img_w, x2); y2 = min(img_h, y2)
                    largura = x2 - x1
                    altura = y2 - y1
                    if largura < 10 or altura < 10:
                        continue
                    recorte = img_bgr[y1:y2, x1:x2].copy()
                    h = _hash_recorte(recorte)

                    # Decide tipo do bloco
                    if regiao.classe == "table":
                        tipo_regiao = "table"
                    elif regiao.classe == "isolate_formula":
                        tipo_regiao = "formula"
                    else:
                        tipo_regiao = "image"

                    # Filtro: figures pequenas viram placeholder SEM texto
                    # (provavel falso positivo do YOLO em label dentro de diagrama)
                    eh_figure_pequena = (
                        tipo_regiao == "image"
                        and (largura < MIN_FIGURE_WIDTH or altura < MIN_FIGURE_HEIGHT)
                    )
                    if eh_figure_pequena:
                        # Salva imagem mas nao roda OCR (e nao adiciona ao output)
                        try:
                            _salvar_imagem_recorte(recorte, h, dir_imagens_pdf)
                        except Exception:
                            pass
                        continue

                    # Sprint 8 itens 5 e 6: tabela/formula usam reconhecedor
                    # especializado em vez de OCR cru. Cai pra OCR como fallback
                    # se o reconhecedor falhar (devolve string vazia).
                    if tipo_regiao == "table":
                        from . import table_recognizer
                        texto_ocr = table_recognizer.reconhecer_tabela(recorte)
                        if not texto_ocr:
                            # Fallback: OCR cru
                            texto_ocr = _deduplica_linhas_consecutivas(_ocr_recorte_cached(recorte, h))
                    elif tipo_regiao == "formula":
                        from . import formula_recognizer
                        texto_ocr = formula_recognizer.reconhecer_formula(recorte)
                        if not texto_ocr:
                            texto_ocr = _deduplica_linhas_consecutivas(_ocr_recorte_cached(recorte, h))
                    else:
                        # Image: OCR cru padrao
                        texto_ocr = _deduplica_linhas_consecutivas(_ocr_recorte_cached(recorte, h))

                    # Sprint 8 item 4: SEMPRE salva crop pra image/formula/table
                    # (paridade RAG com MinerU — usuario pode linkar de volta a figura)
                    image_path: str | None = None
                    if tipo_regiao in ("image", "formula", "table"):
                        try:
                            image_path = _salvar_imagem_recorte(recorte, h, dir_imagens_pdf)
                        except Exception:
                            image_path = None

                    # Pra table/formula, exige texto OCR minimo
                    if tipo_regiao in ("table", "formula") and (
                        not texto_ocr or len(texto_ocr) < LIMIAR_OCR_MIN_CHARS
                    ):
                        continue

                    block_id_ocr = f"p{page_idx}-ocr{ri}"
                    blocos_pagina.append(
                        BlocoExtraido(
                            type=tipo_regiao,
                            text=texto_ocr,
                            page_idx=page_idx,
                            bbox=regiao.bbox,
                            block_id=block_id_ocr,
                            # Sprint 8 item 4: image_path em todos os tipos visuais
                            image_path=image_path if tipo_regiao in ("image", "formula", "table") else None,
                        )
                    )
                    # Sprint 8 item 2: blocos OCR vao pro final da pagina,
                    # ordenados por Y do bbox (offset alto pra aparecer depois
                    # de qualquer block_no vetorial)
                    pdf_order_map[block_id_ocr] = 10000 + int(regiao.bbox[1])

            # B: hierarquia de listas (calcula nivel via x do bbox)
            niveis_lista = _calcular_niveis_listas(blocos_list_item_da_pagina)

            # Sprint 8 item 2: Reading order via block_no do PyMuPDF
            # Resolve multi-coluna automaticamente em PDFs com stream limpo
            # (mesma estrategia do Docling com cell.index)
            blocos_pagina.sort(
                key=lambda b: (
                    pdf_order_map.get(b.block_id or "", 99999),
                    b.bbox[1] if b.bbox else 0,  # tiebreak Y
                    b.bbox[0] if b.bbox else 0,  # tiebreak X
                )
            )

            blocks_finais.extend(blocos_pagina)

            # Texto puro (sem header/footer/page_number)
            partes_texto_pagina = [
                b.text for b in blocos_pagina if b.type not in TIPOS_FORA_DO_TEXTO and b.text
            ]
            paginas_texto.append("\n".join(partes_texto_pagina))

            # Markdown
            partes_md = _gerar_markdown_pagina(blocos_pagina, niveis_lista, style_map)
            if partes_md:
                md_partes.append("\n\n".join(partes_md))

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
