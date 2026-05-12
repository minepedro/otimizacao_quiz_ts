"""
Adaptador Docling -> nosso schema (BlocoExtraido + ResultadoExtracao).

Docling (IBM) tem pipeline proprio: layout (DocLayout-YOLO + classificador
RT-DETR), reading order ML, OCR opcional via RapidOCR. Devolve um
DoclingDocument com:

  - doc.texts: items de texto com label semantica (section_header, text,
    list_item, caption, page_footer, etc) + bbox em coords PDF (BOTTOMLEFT)
  - doc.pictures: imagens detectadas com bbox + label="picture"
  - doc.tables: tabelas estruturadas
  - doc.export_to_markdown(): markdown formatado

Aqui mapeamos isso pro `BlocoExtraido` do nosso schema, preservando bbox
e tipo. ATENCAO: bbox do Docling vem em PDF coords (BOTTOMLEFT). Conversao
pra TOPLEFT (igual aos outros) requer altura da pagina:
   y_topleft = page_height - y_bottomleft
"""

from __future__ import annotations

import time
from pathlib import Path

from ..schemas import BlocoExtraido, ResultadoExtracao

NOME_EXTRATOR = "docling-native"


# Mapeia label do Docling -> TipoBloco do nosso schema
_LABEL_PARA_TIPO: dict[str, str] = {
    "section_header": "title",
    "title": "title",
    "subtitle": "title",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list_item",
    "list": "list",
    "caption": "caption",
    "page_footer": "footer",
    "page_header": "header",
    "footnote": "footer",
    "formula": "formula",
    "code": "code",
    "picture": "image",
    "table": "table",
}


def _label_para_tipo(label: object) -> str:
    """Pega .value se for enum, senao str(). Mapeia pra TipoBloco."""
    if label is None:
        return "paragraph"
    nome = getattr(label, "value", None) or str(label)
    nome = nome.lower().strip()
    return _LABEL_PARA_TIPO.get(nome, "paragraph")


def _bbox_pdf_para_topleft(prov, page_height: float | None = None) -> tuple[float, float, float, float] | None:
    """Converte bbox do Docling (BOTTOMLEFT) pra TOPLEFT (igual aos outros).

    Como nao temos page_height aqui (so dentro do loop), retorna no formato
    bruto BOTTOMLEFT se page_height nao fornecido. O modulo de comparacao
    cuida disso.
    """
    if not prov or len(prov) == 0:
        return None
    bb = prov[0].bbox
    if page_height is not None:
        # Conversao BOTTOMLEFT -> TOPLEFT
        return (
            float(bb.l),
            float(page_height - bb.t),
            float(bb.r),
            float(page_height - bb.b),
        )
    return (float(bb.l), float(bb.t), float(bb.r), float(bb.b))


def _page_no_de_prov(prov) -> int:
    """Extrai page_idx (0-based) do primeiro ProvenanceItem."""
    if not prov or len(prov) == 0:
        return 0
    return max(0, int(prov[0].page_no) - 1)  # Docling usa 1-based


def extrair(caminho_pdf: str | Path) -> ResultadoExtracao:
    inicio = time.perf_counter()
    caminho = Path(caminho_pdf)

    try:
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as e:
        return ResultadoExtracao(
            extrator=NOME_EXTRATOR,
            texto="",
            blocks=[],
            markdown="",
            num_paginas=0,
            num_caracteres=0,
            tempo_segundos=round(time.perf_counter() - inicio, 3),
            erro=f"docling nao instalado: {e}",
        )

    try:
        # Backend pypdfium2 evita "Inconsistent number of pages" do docling-parse
        # (bug conhecido em alguns PDFs, ja descoberto no TS lab)
        conv = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(backend=PyPdfiumDocumentBackend),
            }
        )
        result = conv.convert(caminho)
        doc = result.document
    except Exception as e:
        return ResultadoExtracao(
            extrator=NOME_EXTRATOR,
            texto="",
            blocks=[],
            markdown="",
            num_paginas=0,
            num_caracteres=0,
            tempo_segundos=round(time.perf_counter() - inicio, 3),
            erro=f"{type(e).__name__}: {str(e)[:300]}",
        )

    # Mapa page_idx -> altura pra conversao bbox (algumas paginas podem
    # nao ter altura disponivel — fica em None)
    paginas_altura: dict[int, float | None] = {}
    if hasattr(doc, "pages") and doc.pages:
        for page_no, page in doc.pages.items():
            try:
                paginas_altura[page_no - 1] = float(page.size.height) if page.size else None
            except Exception:
                paginas_altura[page_no - 1] = None

    blocks: list[BlocoExtraido] = []

    # Texts
    for i, t in enumerate(getattr(doc, "texts", [])):
        text = (getattr(t, "text", "") or "").strip()
        if not text:
            continue
        page_idx = _page_no_de_prov(t.prov)
        bbox = _bbox_pdf_para_topleft(t.prov, paginas_altura.get(page_idx))
        tipo = _label_para_tipo(getattr(t, "label", None))
        level = None
        if tipo == "title":
            level = int(getattr(t, "level", 1) or 1)
            if level < 1: level = 1
            if level > 6: level = 6
        blocks.append(
            BlocoExtraido(
                type=tipo,
                text=text,
                page_idx=page_idx,
                bbox=bbox,
                level=level,
                block_id=f"docling-t{i}",
            )
        )

    # Pictures
    for i, p in enumerate(getattr(doc, "pictures", [])):
        page_idx = _page_no_de_prov(p.prov)
        bbox = _bbox_pdf_para_topleft(p.prov, paginas_altura.get(page_idx))
        # Caption: primeiro item de p.captions se houver
        caption_txt = ""
        try:
            for cap_ref in getattr(p, "captions", []) or []:
                cap = cap_ref.resolve(doc) if hasattr(cap_ref, "resolve") else cap_ref
                if hasattr(cap, "text") and cap.text:
                    caption_txt = cap.text
                    break
        except Exception:
            pass
        blocks.append(
            BlocoExtraido(
                type="image",
                text=caption_txt or "",
                page_idx=page_idx,
                bbox=bbox,
                image_caption=caption_txt or None,
                block_id=f"docling-p{i}",
            )
        )

    # Tables
    for i, t in enumerate(getattr(doc, "tables", [])):
        page_idx = _page_no_de_prov(t.prov)
        bbox = _bbox_pdf_para_topleft(t.prov, paginas_altura.get(page_idx))
        # Texto da tabela: tenta export_to_markdown pra ter formato pipe
        texto_tabela = ""
        try:
            texto_tabela = t.export_to_markdown(doc) if hasattr(t, "export_to_markdown") else str(t)
        except Exception:
            texto_tabela = ""
        blocks.append(
            BlocoExtraido(
                type="table",
                text=texto_tabela,
                page_idx=page_idx,
                bbox=bbox,
                block_id=f"docling-tab{i}",
            )
        )

    # Markdown
    try:
        markdown = doc.export_to_markdown()
    except Exception:
        markdown = ""

    # Texto puro: concatena texts (sem tables/pictures vazios)
    partes_texto: list[str] = []
    blocks_por_pagina: dict[int, list[BlocoExtraido]] = {}
    for b in blocks:
        blocks_por_pagina.setdefault(b.page_idx, []).append(b)
    for page_idx in sorted(blocks_por_pagina):
        for b in blocks_por_pagina[page_idx]:
            if b.type in ("header", "footer", "page_number"):
                continue
            if b.text:
                partes_texto.append(b.text)
    texto = "\n".join(partes_texto)

    num_paginas = len(getattr(doc, "pages", {})) or max((b.page_idx + 1 for b in blocks), default=0)

    return ResultadoExtracao(
        extrator=NOME_EXTRATOR,
        texto=texto,
        blocks=blocks,
        markdown=markdown,
        num_paginas=num_paginas,
        num_caracteres=len(texto),
        tempo_segundos=round(time.perf_counter() - inicio, 3),
        erro=None,
    )
