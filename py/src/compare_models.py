"""
Comparacao 3-vias entre Docling x MinerU x nosso pipeline.

Pra cada PDF: roda os 3 extratores, salva os 3 JSONs estruturados, e
imprime tabela de concordancia par-a-par.

Uso:
  py/.venv/Scripts/python.exe -m src.compare_models
  py/.venv/Scripts/python.exe -m src.compare_models --apenas "Aula 04 - Redes"

Output:
  - resultados/structured-jsons/<extrator>/<pdf_stem>.json
  - resultados/structured-comparison.json (matriz de concordancia)
  - stdout: tabela legivel
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .extractors import docling_native, hibrido_layout_aware, mineru_native
from .schemas import BlocoExtraido, ResultadoExtracao
from .structured_compare import (
    CompararPar,
    comparar_tres_vias,
    formatar_matriz,
)

RAIZ = Path(__file__).resolve().parents[2]
DIR_PDFS = RAIZ / "dados" / "pdfs"
DIR_OUT_JSON = RAIZ / "resultados" / "structured-jsons"
PATH_COMPARACAO = RAIZ / "resultados" / "structured-comparison.json"


EXTRATORES = [
    ("hibrido-layout-aware", hibrido_layout_aware.extrair),
    ("docling-native", docling_native.extrair),
    ("mineru-native", mineru_native.extrair),
]


def salvar_resultado(resultado: ResultadoExtracao, pdf_stem: str) -> None:
    """Salva JSON completo + markdown + texto puro pra reuso futuro."""
    out = DIR_OUT_JSON / resultado.extrator
    out.mkdir(parents=True, exist_ok=True)
    # JSON completo (com texto e markdown embutidos pra reuso single-file)
    (out / f"{pdf_stem}.json").write_text(
        resultado.model_dump_json(indent=2),
        encoding="utf-8",
    )
    # Markdown separado (mais facil de inspecionar a olho nu)
    if resultado.markdown:
        (out / f"{pdf_stem}.md").write_text(resultado.markdown, encoding="utf-8")
    # Texto puro separado (idem)
    if resultado.texto:
        (out / f"{pdf_stem}.txt").write_text(resultado.texto, encoding="utf-8")


def carregar_resultado(extrator: str, pdf_stem: str) -> ResultadoExtracao | None:
    """Carrega ResultadoExtracao salvo previamente. Retorna None se nao existir."""
    arq = DIR_OUT_JSON / extrator / f"{pdf_stem}.json"
    if not arq.exists():
        return None
    try:
        data = json.loads(arq.read_text(encoding="utf-8"))
        return ResultadoExtracao.model_validate(data)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Comparacao 3-vias estruturada.")
    parser.add_argument("--apenas", type=str, default=None, help="Filtra PDFs por substring.")
    parser.add_argument(
        "--apenas-extrator",
        type=str,
        default=None,
        help="Roda so um extrator (ainda salva o JSON, comparacao precisa dos 3).",
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Re-extrai mesmo se ja tem JSON salvo (default: reusa).",
    )
    args = parser.parse_args()

    pdfs = sorted(DIR_PDFS.glob("*.pdf"))
    if args.apenas:
        pdfs = [p for p in pdfs if args.apenas.lower() in p.name.lower()]

    extratores = EXTRATORES
    if args.apenas_extrator:
        extratores = [e for e in EXTRATORES if args.apenas_extrator.lower() in e[0].lower()]

    if not pdfs:
        print("Nenhum PDF.")
        return 0

    DIR_OUT_JSON.mkdir(parents=True, exist_ok=True)

    comparacao_global: dict[str, dict] = {}
    inicio_global = time.perf_counter()

    for pdf in pdfs:
        print(f"\n=== {pdf.stem} ===")
        blocks_por_extrator: dict[str, list[BlocoExtraido]] = {}

        for nome, extrair in extratores:
            cached = carregar_resultado(nome, pdf.stem) if not args.forcar else None
            if cached is not None and not cached.erro:
                blocks = cached.blocks
                print(f"  [{nome}] (cache) {len(blocks)} blocks, {cached.tempo_segundos}s")
            else:
                print(f"  [{nome}] extraindo ...", end=" ", flush=True)
                t0 = time.time()
                try:
                    resultado = extrair(pdf)
                except Exception as e:
                    print(f"ERRO: {type(e).__name__}: {e}")
                    continue
                print(
                    f"OK em {time.time()-t0:.1f}s, {len(resultado.blocks)} blocks, "
                    f"erro={resultado.erro}"
                )
                if resultado.erro:
                    # Salva mesmo com erro pra nao re-rodar (a menos que --forcar)
                    salvar_resultado(resultado, pdf.stem)
                    continue
                salvar_resultado(resultado, pdf.stem)
                blocks = resultado.blocks

            blocks_por_extrator[nome] = blocks

        # Comparacao 3-vias (so se temos pelo menos 2 extratores OK)
        if len(blocks_por_extrator) < 2:
            print(f"  Pulando comparacao: {len(blocks_por_extrator)} extratores OK")
            continue

        matriz = comparar_tres_vias(blocks_por_extrator)
        print()
        print(formatar_matriz(matriz))

        # Acumula matriz pra arquivo
        comparacao_global[pdf.stem] = {
            a: {
                b: {
                    "n_blocks_a": p.n_blocks_a,
                    "n_blocks_b": p.n_blocks_b,
                    "n_casados": p.n_casados,
                    "n_tipo_concorda": p.n_tipo_concorda,
                    "taxa_match": round(p.taxa_match, 4),
                    "taxa_tipo": round(p.taxa_tipo, 4),
                    "iou_medio_pares": p.iou_medio_pares,
                    "distribuicao_a": p.distribuicao_a,
                    "distribuicao_b": p.distribuicao_b,
                }
                for b, p in matriz[a].items()
            }
            for a in matriz
        }
        PATH_COMPARACAO.write_text(
            json.dumps(comparacao_global, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\n=== FIM === {time.perf_counter() - inicio_global:.1f}s")
    print(f"JSONs estruturados: {DIR_OUT_JSON.relative_to(RAIZ)}")
    print(f"Matriz comparacao: {PATH_COMPARACAO.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
