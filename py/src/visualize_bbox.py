"""
Visualizacao de bboxes detectados pelos 3 extratores.

Pra cada pagina de UM PDF: renderiza 3 paineis lado a lado, cada um
mostrando os bboxes do respectivo extrator (Docling, MinerU, nosso
hibrido-layout-aware). Cores diferentes por TIPO de bloco.

Saida: 1 PNG por pagina em `resultados/bbox-overlay/<pdf_stem>/`.

ATENCAO: bboxes estao em coords diferentes por extrator:
  - Docling: PDF points (TOPLEFT, ja convertido)
  - MinerU: pixels da imagem renderizada por ele (DPI 200 padrao)
  - Nosso: pixels da imagem renderizada por nos (DPI 150)

Pra visualizar correto, normalizamos cada um pra coords da pagina
em PROPORCAO 0..1 (relative coords) e depois multiplicamos pelo
tamanho da imagem renderizada agora (DPI fixo).

Uso:
  py/.venv/Scripts/python.exe -m src.visualize_bbox --apenas "Aula 04 - Redes"
  py/.venv/Scripts/python.exe -m src.visualize_bbox --apenas Realismo --max-paginas 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .schemas import BlocoExtraido, ResultadoExtracao

RAIZ = Path(__file__).resolve().parents[2]
DIR_PDFS = RAIZ / "dados" / "pdfs"
DIR_JSONS = RAIZ / "resultados" / "structured-jsons"
DIR_OUT = RAIZ / "resultados" / "bbox-overlay"

DPI_RENDER = 150
EXTRATORES = ["hibrido-layout-aware", "docling-native", "mineru-native"]

# Cores fixas por tipo (consistente nos 3 paineis)
COR_POR_TIPO: dict[str, str] = {
    "title": "#e41a1c",        # vermelho
    "paragraph": "#377eb8",    # azul
    "list": "#984ea3",         # roxo
    "list_item": "#984ea3",
    "table": "#ff7f00",        # laranja
    "image": "#4daf4a",        # verde
    "caption": "#a65628",      # marrom
    "header": "#999999",       # cinza
    "footer": "#999999",
    "page_number": "#999999",
    "formula": "#f781bf",      # rosa
    "code": "#000000",         # preto
    "definition": "#ffff33",   # amarelo
    "exercise": "#a6cee3",
    "quote": "#b15928",
}


def _carregar(extrator: str, pdf_stem: str) -> ResultadoExtracao | None:
    arq = DIR_JSONS / extrator / f"{pdf_stem}.json"
    if not arq.exists():
        return None
    try:
        return ResultadoExtracao.model_validate(json.loads(arq.read_text(encoding="utf-8")))
    except Exception:
        return None


def _normalizar_bbox(
    bbox: tuple[float, float, float, float] | None,
    extrator: str,
    page_width_pt: float,
    page_height_pt: float,
) -> tuple[float, float, float, float] | None:
    """Converte bbox de qualquer extrator pra COORDS RELATIVAS (0..1)."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    # Heuristica de qual escala usar
    if extrator == "docling-native":
        # PDF points (TOPLEFT, ja convertido em docling_native.py)
        return (x1 / page_width_pt, y1 / page_height_pt, x2 / page_width_pt, y2 / page_height_pt)
    elif extrator == "mineru-native":
        # MinerU usa DPI 200 padrao -> 200/72 ~= 2.78x
        # Mas o melhor eh detectar pelos limites: se max(x) > page_width_pt * 1.5, eh pixel
        max_dim = max(x2, y2)
        if max_dim > max(page_width_pt, page_height_pt) * 1.2:
            # Esta em pixels; normaliza pelo bbox max do extrator
            # Assume escala consistente com DPI 200
            scale_x = page_width_pt * (200 / 72)
            scale_y = page_height_pt * (200 / 72)
            return (x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y)
        return (x1 / page_width_pt, y1 / page_height_pt, x2 / page_width_pt, y2 / page_height_pt)
    elif extrator == "hibrido-layout-aware":
        # Pixels da imagem renderizada @ DPI 150 -> escala = 150/72
        scale_x = page_width_pt * (150 / 72)
        scale_y = page_height_pt * (150 / 72)
        return (x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y)
    return (x1 / page_width_pt, y1 / page_height_pt, x2 / page_width_pt, y2 / page_height_pt)


def _renderizar_pagina_rgb(page: fitz.Page, dpi: int = DPI_RENDER) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return img[:, :, :3]
    if pix.n == 1:
        return np.stack([img[:, :, 0]] * 3, axis=-1)
    return img


def desenhar_pagina(
    pdf_path: Path,
    page_idx: int,
    blocks_por_extrator: dict[str, list[BlocoExtraido]],
    out_path: Path,
) -> None:
    """Desenha 3 paineis com bboxes pra UMA pagina."""
    doc = fitz.open(pdf_path)
    if page_idx >= doc.page_count:
        doc.close()
        return
    page = doc[page_idx]
    page_w_pt = page.rect.width
    page_h_pt = page.rect.height
    img = _renderizar_pagina_rgb(page)
    img_h, img_w = img.shape[:2]
    doc.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, extrator in zip(axes, EXTRATORES):
        ax.imshow(img)
        ax.set_title(f"{extrator}\n({len([b for b in blocks_por_extrator.get(extrator, []) if b.page_idx == page_idx])} blocks)")
        ax.set_xticks([]); ax.set_yticks([])
        for b in blocks_por_extrator.get(extrator, []):
            if b.page_idx != page_idx:
                continue
            bb_rel = _normalizar_bbox(b.bbox, extrator, page_w_pt, page_h_pt)
            if bb_rel is None:
                continue
            x1, y1, x2, y2 = bb_rel
            x_px = x1 * img_w
            y_px = y1 * img_h
            w_px = (x2 - x1) * img_w
            h_px = (y2 - y1) * img_h
            cor = COR_POR_TIPO.get(b.type, "#999999")
            rect = mpatches.Rectangle(
                (x_px, y_px), w_px, h_px,
                linewidth=1.5, edgecolor=cor, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x_px, y_px - 2, b.type,
                fontsize=6, color=cor,
                bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="none"),
            )

    # Legenda compartilhada
    tipos_presentes = {b.type for blocks in blocks_por_extrator.values() for b in blocks if b.page_idx == page_idx}
    handles = [
        mpatches.Patch(color=COR_POR_TIPO.get(t, "#999999"), label=t)
        for t in sorted(tipos_presentes)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), fontsize=8)
    fig.suptitle(f"{pdf_path.name} — pagina {page_idx + 1}", fontsize=12)
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apenas", type=str, default=None, help="Filtra PDF por substring.")
    parser.add_argument("--max-paginas", type=int, default=None, help="Limita N primeiras paginas por PDF.")
    args = parser.parse_args()

    pdfs = sorted(DIR_PDFS.glob("*.pdf"))
    if args.apenas:
        pdfs = [p for p in pdfs if args.apenas.lower() in p.name.lower()]
    if not pdfs:
        print("Nenhum PDF.")
        return 1

    for pdf in pdfs:
        print(f"\n=== {pdf.stem} ===")
        blocks_por_extrator: dict[str, list[BlocoExtraido]] = {}
        for nome in EXTRATORES:
            r = _carregar(nome, pdf.stem)
            if r is None:
                print(f"  [{nome}] sem cache, pulando")
                blocks_por_extrator[nome] = []
            else:
                blocks_por_extrator[nome] = r.blocks

        # Quantas paginas tem
        doc = fitz.open(pdf)
        n_paginas = doc.page_count
        doc.close()
        if args.max_paginas:
            n_paginas = min(n_paginas, args.max_paginas)

        out_dir = DIR_OUT / pdf.stem
        for i in range(n_paginas):
            out_path = out_dir / f"page-{i + 1:03d}.png"
            desenhar_pagina(pdf, i, blocks_por_extrator, out_path)
            print(f"  pg {i + 1}/{n_paginas} -> {out_path.name}")

    print(f"\nVisualizacoes salvas em: {DIR_OUT.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
