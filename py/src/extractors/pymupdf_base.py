"""
Extrator base com PyMuPDF (fitz) — texto vetorial puro.

Equivalente Python do `mupdf` TS, mas usando o binding nativo
(PyMuPDF) em vez do WASM. Mais rapido e com acesso completo ao
modelo do MuPDF (bbox, fontes, imagens com bbox).

Esse extrator faz APENAS extracao de texto vetorial em ordem
natural — nao roda OCR, nao classifica blocos. Serve como
baseline pra medir o teto do que da pra extrair sem ML.

Output: `ResultadoExtracao` com:
  - texto: texto puro concatenado pagina por pagina (separador \\n\\n)
  - markdown: vazio (esse extrator nao gera markdown)
  - blocks: lista de BlocoExtraido tipo "paragraph" (1 por bloco
            do PyMuPDF; sem classificacao semantica)
  - num_paginas, num_caracteres, tempo_segundos, erro
"""

from __future__ import annotations

import time
from pathlib import Path

import fitz

from ..schemas import BlocoExtraido, ResultadoExtracao

NOME_EXTRATOR = "pymupdf"


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

    paginas_texto: list[str] = []
    blocks: list[BlocoExtraido] = []

    try:
        for page_idx, page in enumerate(doc):
            # Modo 'blocks' devolve [(x0, y0, x1, y1, text, block_no, block_type), ...]
            # block_type 0 = texto, 1 = imagem
            page_blocks = page.get_text("blocks", sort=True)
            partes_pagina: list[str] = []
            for bx in page_blocks:
                if len(bx) < 7:
                    continue
                x0, y0, x1, y1, texto_bloco, block_no, block_type = bx[:7]
                if block_type != 0:
                    continue  # imagem; tratamos depois quando integrar OCR
                texto_limpo = (texto_bloco or "").strip()
                if not texto_limpo:
                    continue
                partes_pagina.append(texto_limpo)
                blocks.append(
                    BlocoExtraido(
                        type="paragraph",
                        text=texto_limpo,
                        page_idx=page_idx,
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                        block_id=f"p{page_idx}-b{block_no}",
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
