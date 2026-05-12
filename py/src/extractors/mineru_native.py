"""
Adaptador MinerU -> nosso schema (BlocoExtraido + ResultadoExtracao).

MinerU 3.0+ usa `mineru.cli.common.aio_do_parse` (async) que processa
PDFs e SALVA em disco varios artefatos:

  <output_dir>/<pdf_stem>/auto/
    ├── <pdf_stem>.md                   # markdown final (mm_markdown)
    ├── <pdf_stem>_content_list.json    # lista de blocks tipados
    ├── <pdf_stem>_middle.json          # estrutura intermediaria
    ├── <pdf_stem>_model.json           # output cru dos modelos
    └── images/                         # figuras extraidas

Aqui nos:
  1. Chamamos `aio_do_parse` em um diretorio temporario
  2. Lemos `<stem>_content_list.json` (formato comum do MinerU)
  3. Mapeamos blocks pro nosso BlocoExtraido[]
  4. Lemos o markdown e devolvemos junto

Modelos baixados on-demand pro cache padrao do MinerU
(`~/.cache/mineru/` ou similar). Pode demorar na primeira chamada.

Backend: 'pipeline' (modular, mais leve, CPU-friendly). Existe tambem
'vlm-engine' (MinerU2.5-Pro VLM, requer GPU).

Nota: MinerU eh LENTO em CPU pra PDFs grandes (1-5 min/PDF).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path

from ..schemas import BlocoExtraido, ResultadoExtracao

NOME_EXTRATOR = "mineru-native"


# Mapeia type do content_list.json do MinerU -> TipoBloco
_TYPE_PARA_TIPO: dict[str, str] = {
    "text": "paragraph",
    "title": "title",
    "equation": "formula",
    "image": "image",
    "table": "table",
    "image_caption": "caption",
    "table_caption": "caption",
    "image_footnote": "caption",
    "table_footnote": "caption",
    "list": "list",
    "code": "code",
}


async def _parse_async(pdf_bytes: bytes, pdf_name: str, out_dir: Path) -> None:
    from mineru.cli.common import aio_do_parse
    await aio_do_parse(
        output_dir=str(out_dir),
        pdf_file_names=[pdf_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=["pt"],
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        # Desliga PDFs/imagens debug pra economizar IO
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
        image_analysis=False,
    )


def extrair(caminho_pdf: str | Path) -> ResultadoExtracao:
    inicio = time.perf_counter()
    caminho = Path(caminho_pdf)
    pdf_bytes = caminho.read_bytes()
    pdf_stem = caminho.stem

    # Diretorio temporario isolado por execucao
    out_dir = Path(tempfile.mkdtemp(prefix="mineru-out-"))

    try:
        try:
            asyncio.run(_parse_async(pdf_bytes, pdf_stem, out_dir))
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

        # MinerU salva em <out_dir>/<pdf_stem>/auto/
        auto_dir = out_dir / pdf_stem / "auto"
        if not auto_dir.exists():
            # Fallback: alguns backends usam outra subpasta
            possiveis = list(out_dir.rglob("*_content_list.json"))
            if possiveis:
                auto_dir = possiveis[0].parent
            else:
                return ResultadoExtracao(
                    extrator=NOME_EXTRATOR,
                    texto="",
                    blocks=[],
                    markdown="",
                    num_paginas=0,
                    num_caracteres=0,
                    tempo_segundos=round(time.perf_counter() - inicio, 3),
                    erro=f"saida do mineru nao encontrada em {out_dir}",
                )

        content_list_path = auto_dir / f"{pdf_stem}_content_list.json"
        md_path = auto_dir / f"{pdf_stem}.md"

        # Le content_list.json
        if not content_list_path.exists():
            possiveis = list(auto_dir.glob("*_content_list.json"))
            if possiveis:
                content_list_path = possiveis[0]
        items = []
        if content_list_path.exists():
            items = json.loads(content_list_path.read_text(encoding="utf-8"))

        # Le markdown
        markdown = ""
        if md_path.exists():
            markdown = md_path.read_text(encoding="utf-8")
        else:
            possiveis = list(auto_dir.glob("*.md"))
            if possiveis:
                markdown = possiveis[0].read_text(encoding="utf-8")

        blocks: list[BlocoExtraido] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            raw_type = item.get("type", "text")
            # MinerU marca title via text_level (1-6) em vez de type="title"
            if raw_type == "text" and item.get("text_level"):
                tipo = "title"
            else:
                tipo = _TYPE_PARA_TIPO.get(raw_type, "paragraph")
            text = item.get("text") or item.get("img_caption") or ""
            if isinstance(text, list):
                text = " ".join(str(x) for x in text if x)
            text = str(text).strip()
            page_idx = int(item.get("page_idx", 0))
            bbox_raw = item.get("bbox") or item.get("img_bbox") or item.get("table_bbox")
            bbox = None
            if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                try:
                    bbox = (float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]))
                except (TypeError, ValueError):
                    bbox = None
            level = item.get("text_level")
            if level is not None:
                try:
                    level = int(level)
                    if level < 1: level = 1
                    if level > 6: level = 6
                except (TypeError, ValueError):
                    level = None
            if not text and tipo not in ("image", "table"):
                continue
            blocks.append(
                BlocoExtraido(
                    type=tipo,
                    text=text,
                    page_idx=page_idx,
                    bbox=bbox,
                    level=level,
                    block_id=f"mineru-i{i}",
                )
            )

        # Texto puro: concatena blocks visiveis (sem header/footer)
        partes_texto: list[str] = []
        for b in blocks:
            if b.type in ("header", "footer", "page_number"):
                continue
            if b.text:
                partes_texto.append(b.text)
        texto = "\n".join(partes_texto)

        num_paginas = max((b.page_idx + 1 for b in blocks), default=0)

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
    finally:
        # Limpa temp dir
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass
