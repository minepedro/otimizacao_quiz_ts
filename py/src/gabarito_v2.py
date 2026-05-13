"""
Gerador de gabarito v2 — markdown + JSON estruturado.

Porte ampliado do `ts/src/extratores/claude.ts`. Mesma logica de
chunking recursivo + detector de recusa + retry rate limit, mas o
prompt agora pede DOIS formatos paralelos:

1. markdown (str): estrutura visual com #, ##, -, tabelas, etc.
2. blocks (list[BlocoExtraido]): JSON tipado pra rastreabilidade RAG

A geracao usa `tool_use` da API Anthropic — o modelo eh forcado a
chamar a tool `salvar_gabarito` cujo schema bate com `GabaritoV2`,
o que garante JSON estruturado direto (sem parsing de markdown).

Uso:
    py/.venv/Scripts/python.exe -m src.gabarito_v2
    py/.venv/Scripts/python.exe -m src.gabarito_v2 --apenas "Aula 04"
    py/.venv/Scripts/python.exe -m src.gabarito_v2 --forcar
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError
from pypdf import PdfReader, PdfWriter

from .schemas import BlocoExtraido, GabaritoV2

# ============================================================
# Constantes (espelham ts/src/extratores/claude.ts)
# ============================================================

MODELO = "claude-sonnet-4-6"

# Precos oficiais Sonnet 4.6 (USD por milhao de tokens)
INPUT_COST_PER_MTOKEN = 3.0
OUTPUT_COST_PER_MTOKEN = 15.0

MAX_TOKENS_RESPOSTA = 32_000  # tool_use ocupa mais tokens, deixei folga

# Heuristica de deteccao de recusa silenciosa
RECUSA_INPUT_MIN = 10_000
RECUSA_OUTPUT_MAX = 500
PREFIXO_ERRO_RECUSA = "RECUSA:"

PREFIXO_ERRO_RATE_LIMIT = "RATE_LIMIT:"
MAX_RETRIES_SDK = 6
MAX_RETRIES_MANUAIS = 3
SLEEP_BASE_SEG = 30  # 30s, 60s, 120s

# Caminhos relativos a raiz do repo (este arquivo: py/src/gabarito_v2.py)
RAIZ = Path(__file__).resolve().parents[2]
DIR_PDFS = RAIZ / "dados" / "pdfs"
DIR_GAB_MD = RAIZ / "dados" / "gabaritos-v2-md"
DIR_GAB_JSON = RAIZ / "dados" / "gabaritos-v2-json"
PATH_METADATA = RAIZ / "resultados" / "gabaritos_v2_metadata.json"


# ============================================================
# Prompt + tool schema
# ============================================================

PROMPT_GABARITO_V2 = """\
Voce eh um extrator de PDF de altissima fidelidade pra alimentar pipelines de RAG \
(Retrieval-Augmented Generation). Sua saida vira a fonte da verdade contra a qual \
outros extratores de PDF serao avaliados — ela precisa ser COMPLETA e ESTRUTURADA.

Extraia TODO o conteudo do PDF nos dois formatos paralelos da tool `salvar_gabarito`:

## REGRAS GERAIS
- NAO resuma, NAO interprete, NAO adicione preambulo. Apenas transcreva o que esta la.
- INCLUA texto dentro de imagens/screenshots (NAO pule!) — eh o ponto fraco dos \
extratores que estamos comparando.
- INCLUA texto vetorial completo, na ordem natural de leitura (titulo antes de corpo, \
coluna esquerda antes da direita).
- Headers/rodapes/numeros de pagina: incluir, mas marcar com type=header/footer/page_number.
- Formulas matematicas: transcrever em LaTeX inline ($...$) ou bloco ($$...$$) \
quando possivel; ou texto literal se nao der.
- Tabelas: usar pipe markdown no campo `markdown`, e type=table no campo `blocks` \
com o texto da celula concatenado.

## CAMPO `markdown`
String unica em markdown bem-formada cobrindo o PDF inteiro. Use:
- `# titulo`, `## subtitulo`, `### sub-sub` (nivel correspondente)
- `- item` pra listas, `1. item` pra listas numeradas
- tabelas em pipe markdown
- `![descricao](image_<idx>)` pra imagens (idx referenciando bloco type=image)
- separador `\\n\\n---\\n\\n` entre paginas eh OPCIONAL — a quebra de pagina natural \
do markdown serve.

## CAMPO `blocks`
Lista de blocos tipados, em ORDEM DE LEITURA. Cada bloco tem:
- `type`: title | paragraph | list | list_item | table | image | caption | exercise | \
definition | formula | header | footer | page_number | code | quote
- `text`: o texto literal do bloco (sem markdown — o markdown vai no outro campo)
- `page_idx`: indice da pagina onde o bloco aparece (0-based)
- `level` (opcional, 1-6): pra type=title indica nivel do cabecalho; pra type=list \
e list_item indica nivel de aninhamento
- `image_caption` (opcional): pra type=image, o texto da legenda se houver

NAO inclua bbox nem block_id nem parent_id — esses campos sao pros extratores \
automatizados, nao pro gabarito humano (Claude).

Comece direto pela tool — nao escreva nada antes."""


# Schema do tool_use — bate com GabaritoV2 mas sem campos de metadata
# (que sao preenchidos por nos, nao pelo Claude).
TOOL_SALVAR_GABARITO: dict[str, Any] = {
    "name": "salvar_gabarito",
    "description": (
        "Salva o gabarito completo do PDF em dois formatos paralelos: "
        "markdown (string formatada) e blocks (lista de blocos tipados em ordem)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "markdown": {
                "type": "string",
                "description": "Conteudo COMPLETO do PDF em markdown bem-formado.",
            },
            "blocks": {
                "type": "array",
                "description": "Lista de blocos tipados em ordem de leitura.",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "title", "paragraph", "list", "list_item",
                                "table", "image", "caption", "exercise",
                                "definition", "formula", "header", "footer",
                                "page_number", "code", "quote",
                            ],
                        },
                        "text": {"type": "string"},
                        "page_idx": {"type": "integer", "minimum": 0},
                        "level": {"type": "integer", "minimum": 1, "maximum": 6},
                        "image_caption": {"type": "string"},
                    },
                    "required": ["type", "text", "page_idx"],
                },
            },
        },
        "required": ["markdown", "blocks"],
    },
}


# ============================================================
# Tipos internos
# ============================================================


@dataclass
class ChunkInfo:
    paginas: str
    depth: int
    stop_reason: str | None
    erro: str | None
    foi_dividido: bool = False


@dataclass
class RetornoChamada:
    markdown: str
    blocks: list[dict[str, Any]]
    tokens_input: int
    tokens_output: int
    custo_usd: float
    stop_reason: str | None
    erro: str | None


@dataclass
class StatsExtracao:
    tokens_input: int = 0
    tokens_output: int = 0
    custo_usd: float = 0.0
    num_chunks: int = 0
    chunks: list[ChunkInfo] = field(default_factory=list)
    recusa_detectada: bool = False


# ============================================================
# Helpers
# ============================================================


def eh_erro_rate_limit(msg: str) -> bool:
    return "429" in msg or "rate_limit" in msg.lower()


def fatiar_pdf(bytes_pdf: bytes, start: int, end: int) -> bytes:
    """Extrai paginas [start, end) do PDF em bytes, retorna novo PDF em bytes."""
    reader = PdfReader(io.BytesIO(bytes_pdf))
    writer = PdfWriter()
    for i in range(start, end):
        writer.add_page(reader.pages[i])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def contar_paginas(bytes_pdf: bytes) -> int:
    try:
        return len(PdfReader(io.BytesIO(bytes_pdf)).pages)
    except Exception:
        return 0


def calc_custo(tokens_input: int, tokens_output: int) -> float:
    return round(
        (tokens_input * INPUT_COST_PER_MTOKEN + tokens_output * OUTPUT_COST_PER_MTOKEN)
        / 1_000_000,
        4,
    )


# ============================================================
# Chamada Claude com tool_use
# ============================================================


def chamar_claude_uma_vez(
    client: anthropic.Anthropic,
    pdf_bytes: bytes,
) -> RetornoChamada:
    try:
        import base64
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")

        # Streaming necessario pra max_tokens alto
        with client.messages.stream(
            model=MODELO,
            max_tokens=MAX_TOKENS_RESPOSTA,
            tools=[TOOL_SALVAR_GABARITO],
            tool_choice={"type": "tool", "name": "salvar_gabarito"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {"type": "text", "text": PROMPT_GABARITO_V2},
                    ],
                },
            ],
        ) as stream:
            for _ in stream:
                pass
            final = stream.get_final_message()

        tokens_input = final.usage.input_tokens
        tokens_output = final.usage.output_tokens
        custo = calc_custo(tokens_input, tokens_output)

        # Procura o tool_use bloco
        markdown = ""
        blocks: list[dict[str, Any]] = []
        for bloco in final.content:
            if bloco.type == "tool_use" and bloco.name == "salvar_gabarito":
                inp = bloco.input
                if isinstance(inp, dict):
                    md_raw = inp.get("markdown", "")
                    bl_raw = inp.get("blocks", [])
                    # Defensivo: o modelo as vezes alucina tipos errados
                    markdown = md_raw if isinstance(md_raw, str) else ""
                    blocks = bl_raw if isinstance(bl_raw, list) else []
                break

        # Detector de recusa: se nao chamou a tool ou retornou vazio com input alto
        eh_recusa = (
            final.stop_reason == "end_turn"
            and tokens_input >= RECUSA_INPUT_MIN
            and tokens_output < RECUSA_OUTPUT_MAX
            and not markdown
            and not blocks
        )

        erro: str | None = None
        if eh_recusa:
            erro = (
                f"{PREFIXO_ERRO_RECUSA} output_tokens={tokens_output} pra "
                f"input_tokens={tokens_input}. Provavel recusa do modelo "
                "(copyright/policy)."
            )
        elif final.stop_reason == "max_tokens":
            # Tool_use truncou — precisa dividir
            erro = None  # nao eh erro, mas o caller vai ver stop_reason

        return RetornoChamada(
            markdown=markdown,
            blocks=blocks,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            custo_usd=custo,
            stop_reason=final.stop_reason,
            erro=erro,
        )
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        erro = f"{PREFIXO_ERRO_RATE_LIMIT} {msg}" if eh_erro_rate_limit(msg) else msg
        return RetornoChamada(
            markdown="",
            blocks=[],
            tokens_input=0,
            tokens_output=0,
            custo_usd=0.0,
            stop_reason=None,
            erro=erro,
        )


def chamar_claude_com_retry(
    pdf_bytes: bytes,
    chave: str,
    verbose: bool = False,
) -> RetornoChamada:
    client = anthropic.Anthropic(api_key=chave, max_retries=MAX_RETRIES_SDK)

    for i in range(MAX_RETRIES_MANUAIS + 1):
        r = chamar_claude_uma_vez(client, pdf_bytes)
        if not r.erro or not r.erro.startswith(PREFIXO_ERRO_RATE_LIMIT):
            return r
        if i == MAX_RETRIES_MANUAIS:
            return r
        sleep = SLEEP_BASE_SEG * (2**i)
        if verbose:
            print(
                f"        rate limit: aguardando {sleep}s antes de re-tentar "
                f"({i + 1}/{MAX_RETRIES_MANUAIS})..."
            )
        time.sleep(sleep)

    return RetornoChamada(
        markdown="",
        blocks=[],
        tokens_input=0,
        tokens_output=0,
        custo_usd=0.0,
        stop_reason=None,
        erro=f"{PREFIXO_ERRO_RATE_LIMIT} retries esgotados",
    )


# ============================================================
# Chunking recursivo
# ============================================================


@dataclass
class ResultadoExtracaoGabarito:
    markdown: str
    blocks: list[BlocoExtraido]
    num_paginas: int
    tempo_segundos: float
    erro: str | None
    stats: StatsExtracao


def extrair_com_chunking(
    caminho_pdf: Path,
    chave: str,
    max_profundidade: int = 6,
    verbose: bool = True,
) -> ResultadoExtracaoGabarito:
    inicio = time.perf_counter()
    pdf_bytes = caminho_pdf.read_bytes()
    num_paginas = contar_paginas(pdf_bytes)

    if verbose:
        print(f"Extracao recursiva: {caminho_pdf.name} ({num_paginas} pg)")

    stats = StatsExtracao()

    def recurse(start: int, end: int, depth: int) -> tuple[str, list[dict[str, Any]]]:
        n = end - start
        label = f"pg {start + 1}-{end}"
        prefixo = "  " * depth
        if verbose:
            print(f"{prefixo}[chunk {label}] {n} pg, depth={depth}")

        slice_bytes = fatiar_pdf(pdf_bytes, start, end)
        r = chamar_claude_com_retry(slice_bytes, chave, verbose)

        stats.tokens_input += r.tokens_input
        stats.tokens_output += r.tokens_output
        stats.custo_usd = round(stats.custo_usd + r.custo_usd, 4)
        stats.num_chunks += 1
        registro = ChunkInfo(
            paginas=label,
            depth=depth,
            stop_reason=r.stop_reason,
            erro=r.erro,
            foi_dividido=False,
        )
        stats.chunks.append(registro)

        # Rate limit nao se resolve dividindo
        if r.erro and r.erro.startswith(PREFIXO_ERRO_RATE_LIMIT):
            if verbose:
                print(f"{prefixo}  -> rate limit persistiu, NAO dividindo")
            return r.markdown, r.blocks

        # Recusa nao se resolve dividindo
        eh_recusa = bool(r.erro) and r.erro.startswith(PREFIXO_ERRO_RECUSA)
        if eh_recusa:
            if verbose:
                print(f"{prefixo}  -> recusa detectada, NAO dividindo")
            stats.recusa_detectada = True
            return r.markdown, r.blocks

        precisa_dividir = bool(r.erro) or r.stop_reason == "max_tokens"

        if precisa_dividir and n > 1 and depth < max_profundidade:
            registro.foi_dividido = True
            if verbose:
                motivo = "erro" if r.erro else "max_tokens"
                print(f"{prefixo}  -> dividindo ({motivo})")
            meio = start + n // 2
            md_esq, bl_esq = recurse(start, meio, depth + 1)
            md_dir, bl_dir = recurse(meio, end, depth + 1)
            # Reajusta page_idx do lado direito (offset = meio - start absoluto)
            # Na verdade, page_idx ja vem relativo ao chunk (0-based dentro do chunk).
            # Precisamos somar `start` (absoluto) pra ficar global.
            # Faremos isso APENAS no nivel raiz pra evitar re-aplicar em cascata.
            return (md_esq + "\n\n" + md_dir), (bl_esq + bl_dir)

        if precisa_dividir:
            if verbose:
                print(f"{prefixo}  -> chunk indivisivel, retornando como esta")

        return r.markdown, r.blocks

    # Chamada inicial: rastreamos tambem o offset por chunk pra ajustar page_idx
    # Solucao mais simples: fazer recursao plana (lista de chunks finais com offset)
    # e ajustar no final. Mas pra v1 vamos manter a logica TS fielmente — o
    # ajuste de page_idx fica como TODO se virar problema.
    markdown, blocks_raw = recurse(0, num_paginas, 0)

    # Valida cada bloco com Pydantic
    blocks_validados: list[BlocoExtraido] = []
    erros_validacao: list[str] = []
    for i, b in enumerate(blocks_raw):
        try:
            blocks_validados.append(BlocoExtraido.model_validate(b))
        except ValidationError as e:
            erros_validacao.append(f"bloco {i}: {e.errors()[:1]}")

    erro_final: str | None = None
    if stats.recusa_detectada:
        erro_final = (
            f"{PREFIXO_ERRO_RECUSA} modelo recusou transcrever este PDF "
            "(provavel copyright/policy)."
        )
    elif erros_validacao:
        # Nao bloqueia salvamento — apenas avisa
        if verbose:
            print(
                f"  AVISO: {len(erros_validacao)} blocos invalidados (de {len(blocks_raw)}):"
            )
            for msg in erros_validacao[:3]:
                print(f"    - {msg}")

    return ResultadoExtracaoGabarito(
        markdown=markdown,
        blocks=blocks_validados,
        num_paginas=num_paginas,
        tempo_segundos=round(time.perf_counter() - inicio, 2),
        erro=erro_final,
        stats=stats,
    )


# ============================================================
# Orquestracao: rodar nos PDFs
# ============================================================


def carregar_metadata_existente() -> dict[str, Any]:
    if PATH_METADATA.exists():
        return json.loads(PATH_METADATA.read_text(encoding="utf-8"))
    return {}


def salvar_metadata(metadata: dict[str, Any]) -> None:
    PATH_METADATA.parent.mkdir(parents=True, exist_ok=True)
    PATH_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def gerar_para_pdf(caminho_pdf: Path, chave: str, forcar: bool) -> dict[str, Any]:
    """Gera gabarito v2 pra UM PDF. Retorna metadata da geracao."""
    nome = caminho_pdf.stem
    path_md = DIR_GAB_MD / f"{nome}.md"
    path_json = DIR_GAB_JSON / f"{nome}.json"

    if not forcar and path_md.exists() and path_json.exists():
        print(f"  [SKIP] {nome} ja tem gabarito v2 (use --forcar pra refazer)")
        return {"skip": True}

    DIR_GAB_MD.mkdir(parents=True, exist_ok=True)
    DIR_GAB_JSON.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {nome} ===")
    resultado = extrair_com_chunking(caminho_pdf, chave)

    # Monta GabaritoV2 (valida tudo)
    gabarito = GabaritoV2(
        pdf_name=caminho_pdf.name,
        markdown=resultado.markdown,
        blocks=resultado.blocks,
        model=MODELO,
        tokens_input=resultado.stats.tokens_input,
        tokens_output=resultado.stats.tokens_output,
        cost_usd=resultado.stats.custo_usd,
        elapsed_seconds=resultado.tempo_segundos,
    )

    # Salva
    path_md.write_text(gabarito.markdown, encoding="utf-8")
    path_json.write_text(
        gabarito.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(
        f"  OK: {len(gabarito.markdown)} chars md, {len(gabarito.blocks)} blocks, "
        f"${gabarito.cost_usd:.4f}, {gabarito.elapsed_seconds:.1f}s"
    )
    if resultado.erro:
        print(f"  ERRO: {resultado.erro}")

    return {
        "nome_arquivo": caminho_pdf.name,
        "num_paginas": resultado.num_paginas,
        "num_chars_md": len(gabarito.markdown),
        "num_blocks": len(gabarito.blocks),
        "tokens_input": gabarito.tokens_input,
        "tokens_output": gabarito.tokens_output,
        "custo_usd": gabarito.cost_usd,
        "tempo_segundos": gabarito.elapsed_seconds,
        "num_chunks": resultado.stats.num_chunks,
        "chunks": [
            {
                "paginas": c.paginas,
                "depth": c.depth,
                "stop_reason": c.stop_reason,
                "erro": c.erro,
                "foi_dividido": c.foi_dividido,
            }
            for c in resultado.stats.chunks
        ],
        "recusa_detectada": resultado.stats.recusa_detectada,
        "erro_final": resultado.erro,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera gabarito v2 (markdown + JSON) pros PDFs do lab.",
    )
    parser.add_argument(
        "--apenas",
        type=str,
        default=None,
        help="Filtra PDFs por substring no nome do arquivo.",
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Re-gera mesmo se gabarito v2 ja existir.",
    )
    args = parser.parse_args()

    load_dotenv(RAIZ / ".env")
    import os
    chave = os.getenv("ANTHROPIC_API_KEY")
    if not chave:
        print("ERRO: ANTHROPIC_API_KEY nao encontrada no .env nem no env.")
        return 1

    if not DIR_PDFS.exists():
        print(f"ERRO: {DIR_PDFS} nao existe.")
        return 1

    pdfs = sorted(DIR_PDFS.glob("*.pdf"))
    if args.apenas:
        antes = len(pdfs)
        pdfs = [p for p in pdfs if args.apenas.lower() in p.name.lower()]
        print(f"Filtro --apenas '{args.apenas}': {len(pdfs)} de {antes} PDFs")

    if not pdfs:
        print("Nenhum PDF pra processar.")
        return 0

    print(f"Vou processar {len(pdfs)} PDF(s):")
    for p in pdfs:
        print(f"  - {p.name}")

    metadata = carregar_metadata_existente()
    custo_total = 0.0
    inicio = time.perf_counter()

    for p in pdfs:
        try:
            info = gerar_para_pdf(p, chave, args.forcar)
            if info.get("skip"):
                continue
            metadata[p.stem] = info
            custo_total += info.get("custo_usd", 0.0)
            # Salva metadata incrementalmente (caso de crash)
            salvar_metadata(metadata)
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuario.")
            salvar_metadata(metadata)
            return 130
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  ERRO inesperado em {p.name}: {type(e).__name__}: {e}")
            print(tb)
            metadata[p.stem] = {
                "nome_arquivo": p.name,
                "erro_final": f"{type(e).__name__}: {e}",
                "traceback": tb,
            }
            salvar_metadata(metadata)

    print(
        f"\n=== FIM === custo total: ${custo_total:.4f}, "
        f"tempo: {time.perf_counter() - inicio:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
