"""
Schemas Pydantic do pipeline Python.

`BlocoExtraido` eh o tipo central — todo extrator devolve uma lista
desses blocos. Inspirado no `content_list.json` do MinerU mas
enriquecido com campos pra rastreabilidade RAG no tutor-ai.

`GabaritoV2` eh o que o gerador de gabaritos v2 (Claude API) produz —
combina markdown + lista de blocos pro mesmo PDF.

`ResultadoExtracao` eh o equivalente Python do `ResultadoExtracao` TS
(em `ts/src/utils/tipos.ts`), mantendo compatibilidade conceitual.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TipoBloco = Literal[
    "title",
    "paragraph",
    "list",
    "list_item",
    "table",
    "image",
    "caption",
    "exercise",
    "definition",
    "formula",
    "header",
    "footer",
    "page_number",
    "code",
    "quote",
]


class BlocoExtraido(BaseModel):
    """Bloco semantico extraido de uma pagina do PDF.

    Campos minimos: type + text + page_idx.
    Campos opcionais (level, bbox, parent_id) habilitam recursos
    avancados: hierarquia, abrir PDF na regiao exata, RAG com
    rastreabilidade.
    """

    model_config = ConfigDict(extra="forbid")

    type: TipoBloco
    text: str = Field(min_length=0)
    page_idx: int = Field(ge=0)
    # Nivel de cabecalho 1-6 (so faz sentido pra type=title) ou nivel
    # de aninhamento de lista (so pra type=list/list_item).
    level: int | None = Field(default=None, ge=1, le=6)
    # Bbox em coordenadas da imagem renderizada (pixels), [x1, y1, x2, y2]
    bbox: tuple[float, float, float, float] | None = None
    # ID estavel pra referencia (ex: "ex4-pg2") — gerado pelo extrator
    block_id: str | None = None
    # Pra hierarquia (ex: list_item dentro de list)
    parent_id: str | None = None
    # Imagens: caminho do arquivo extraido (relativo ao output dir)
    image_path: str | None = None
    image_caption: str | None = None
    # OCR confidence (0-100), so faz sentido em blocos vindos de OCR
    ocr_confidence: float | None = Field(default=None, ge=0, le=100)


class GabaritoV2(BaseModel):
    """Gabarito v2 gerado pelo Claude — markdown + JSON paralelos."""

    model_config = ConfigDict(extra="forbid")

    pdf_name: str
    markdown: str
    blocks: list[BlocoExtraido]
    # Metadata de geracao
    model: str = "claude-sonnet-4-6"
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0


class ResultadoExtracao(BaseModel):
    """Resultado de UM extrator rodado em UM PDF.

    Equivalente ao `ResultadoExtracao` TS (`ts/src/utils/tipos.ts`).
    Mantem campos compativeis pra facilitar comparacao cross-stack.
    """

    model_config = ConfigDict(extra="forbid")

    extrator: str
    texto: str  # texto puro (compatibilidade com benchmark TS)
    blocks: list[BlocoExtraido] = Field(default_factory=list)
    markdown: str = ""
    num_paginas: int = 0
    num_caracteres: int = 0
    tempo_segundos: float = 0.0
    erro: str | None = None
