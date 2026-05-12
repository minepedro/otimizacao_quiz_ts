"""
Benchmark dos extratores Python contra o gabarito v2.

Equivalente do `ts/src/benchmark.ts`. Pra cada PDF em
`dados/pdfs/` que tenha gabarito em `dados/gabaritos-v2-md/`,
roda os extratores Python registrados, calcula as 5 metricas e
escreve resultados em:

  - resultados/benchmark-v2.csv  (1 linha por par PDF x extrator)
  - resultados/benchmark-v2.json (agregados por extrator)

Uso:
  py/.venv/Scripts/python.exe -m src.benchmark
  py/.venv/Scripts/python.exe -m src.benchmark --apenas "Aula 04"
  py/.venv/Scripts/python.exe -m src.benchmark --apenas-extrator pymupdf

MERGE: igual ao TS, quando rodado com --apenas ou
--apenas-extrator, mescla com o CSV existente — linhas pra pares
(PDF, extrator) que NAO foram filtrados sao preservadas. Sem
filtro, sobrescreve tudo.

Comparacao com o gabarito: por padrao usa o markdown do v2
(`dados/gabaritos-v2-md/<nome>.md`). Eh o equivalente "texto
puro de referencia" mais fiel disponivel hoje.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

from .extractors import (
    hibrido_layout_aware,
    hibrido_pymupdf_rapidocr,
    paddleocr_full,
    pymupdf_base,
    rapidocr_full,
    tesseract_full,
)
from .metrics import calcular_metricas
from .schemas import ResultadoExtracao

# Caminhos relativos a raiz do repo (este arquivo: py/src/benchmark.py)
RAIZ = Path(__file__).resolve().parents[2]
DIR_PDFS = RAIZ / "dados" / "pdfs"
DIR_GABARITOS = RAIZ / "dados" / "gabaritos-v2-md"
DIR_RESULTADOS = RAIZ / "resultados"
PATH_CSV = DIR_RESULTADOS / "benchmark-v2.csv"
PATH_JSON = DIR_RESULTADOS / "benchmark-v2.json"


# Header igual ao TS (`ts/src/benchmark.ts`) pra facilitar comparacao
COLUNAS_CSV = [
    "pdf",
    "extrator",
    "num_paginas",
    "tempo_seg",
    "num_caracteres",
    "sim_levenshtein",
    "cobertura_palavras",
    "sim_bigramas",
    "tamanho_relativo",
    "qualidade_excesso",
    "erro",
]


# Registro de extratores. Cada item: (nome, callable que retorna ResultadoExtracao)
EXTRATORES: list[tuple[str, Callable[[Path], ResultadoExtracao]]] = [
    (pymupdf_base.NOME_EXTRATOR, pymupdf_base.extrair),
    (rapidocr_full.NOME_EXTRATOR, rapidocr_full.extrair),
    (paddleocr_full.NOME_EXTRATOR, paddleocr_full.extrair),
    (tesseract_full.NOME_EXTRATOR, tesseract_full.extrair),
    (hibrido_pymupdf_rapidocr.NOME_EXTRATOR, hibrido_pymupdf_rapidocr.extrair),
    (hibrido_layout_aware.NOME_EXTRATOR, hibrido_layout_aware.extrair),
]


def carregar_csv_existente() -> list[dict[str, str]]:
    if not PATH_CSV.exists():
        return []
    with PATH_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def salvar_csv(linhas: Iterable[dict]) -> None:
    PATH_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PATH_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUNAS_CSV)
        writer.writeheader()
        for linha in linhas:
            writer.writerow(linha)


def fmt_metric(v: float, casas: int = 4) -> str:
    return f"{v:.{casas}f}"


def rodar_extracao(
    pdf: Path,
    nome_extrator: str,
    extrair: Callable[[Path], ResultadoExtracao],
    gabarito: str,
) -> dict[str, str]:
    print(f"  [{nome_extrator}] {pdf.name}", end=" ... ", flush=True)
    try:
        resultado = extrair(pdf)
    except Exception as e:
        print(f"ERRO: {type(e).__name__}: {e}")
        return {
            "pdf": pdf.stem,
            "extrator": nome_extrator,
            "num_paginas": "0",
            "tempo_seg": "0",
            "num_caracteres": "0",
            "sim_levenshtein": "",
            "cobertura_palavras": "",
            "sim_bigramas": "",
            "tamanho_relativo": "",
            "qualidade_excesso": "",
            "erro": f"{type(e).__name__}: {e}",
        }

    if resultado.erro:
        print(f"ERRO: {resultado.erro}")
        return {
            "pdf": pdf.stem,
            "extrator": nome_extrator,
            "num_paginas": str(resultado.num_paginas),
            "tempo_seg": fmt_metric(resultado.tempo_segundos, 3),
            "num_caracteres": str(resultado.num_caracteres),
            "sim_levenshtein": "",
            "cobertura_palavras": "",
            "sim_bigramas": "",
            "tamanho_relativo": "",
            "qualidade_excesso": "",
            "erro": resultado.erro,
        }

    m = calcular_metricas(resultado.texto, gabarito)
    print(
        f"cob={fmt_metric(m.cobertura_palavras, 3)} "
        f"lev={fmt_metric(m.sim_levenshtein, 3)} "
        f"bi={fmt_metric(m.sim_bigramas, 3)} "
        f"tam={fmt_metric(m.tamanho_relativo, 2)} "
        f"qexc={fmt_metric(m.qualidade_excesso, 3)} "
        f"({resultado.tempo_segundos:.1f}s)"
    )

    return {
        "pdf": pdf.stem,
        "extrator": nome_extrator,
        "num_paginas": str(resultado.num_paginas),
        "tempo_seg": fmt_metric(resultado.tempo_segundos, 3),
        "num_caracteres": str(resultado.num_caracteres),
        "sim_levenshtein": fmt_metric(m.sim_levenshtein),
        "cobertura_palavras": fmt_metric(m.cobertura_palavras),
        "sim_bigramas": fmt_metric(m.sim_bigramas),
        "tamanho_relativo": fmt_metric(m.tamanho_relativo),
        "qualidade_excesso": fmt_metric(m.qualidade_excesso),
        "erro": "",
    }


def calcular_agregados(linhas: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Media das metricas por extrator (apenas linhas sem erro)."""
    por_extrator: dict[str, list[dict[str, str]]] = {}
    for ln in linhas:
        if ln.get("erro"):
            continue
        por_extrator.setdefault(ln["extrator"], []).append(ln)

    agregados: dict[str, dict[str, float]] = {}
    for nome, ll in por_extrator.items():
        if not ll:
            continue
        n = len(ll)

        def media(col: str) -> float:
            return round(sum(float(x[col]) for x in ll) / n, 4)

        agregados[nome] = {
            "n_pdfs_sem_erro": n,
            "tempo_seg_medio": media("tempo_seg"),
            "sim_levenshtein": media("sim_levenshtein"),
            "cobertura_palavras": media("cobertura_palavras"),
            "sim_bigramas": media("sim_bigramas"),
            "tamanho_relativo": media("tamanho_relativo"),
            "qualidade_excesso": media("qualidade_excesso"),
        }
    return agregados


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark dos extratores Python contra gabarito v2.",
    )
    parser.add_argument(
        "--apenas",
        type=str,
        default=None,
        help="Filtra PDFs por substring no nome.",
    )
    parser.add_argument(
        "--apenas-extrator",
        type=str,
        default=None,
        help="Filtra extratores por substring no nome.",
    )
    args = parser.parse_args()

    if not DIR_PDFS.exists():
        print(f"ERRO: {DIR_PDFS} nao existe.")
        return 1
    if not DIR_GABARITOS.exists():
        print(
            f"ERRO: {DIR_GABARITOS} nao existe. Rode src.gabarito_v2 primeiro."
        )
        return 1

    pdfs = sorted(DIR_PDFS.glob("*.pdf"))
    if args.apenas:
        pdfs = [p for p in pdfs if args.apenas.lower() in p.name.lower()]

    extratores = EXTRATORES
    if args.apenas_extrator:
        extratores = [
            e for e in EXTRATORES if args.apenas_extrator.lower() in e[0].lower()
        ]

    if not pdfs:
        print("Nenhum PDF pra processar.")
        return 0
    if not extratores:
        print("Nenhum extrator pra rodar.")
        return 0

    print(f"PDFs: {len(pdfs)} | Extratores: {len(extratores)}")
    print(f"Total de extracoes: {len(pdfs) * len(extratores)}")
    print()

    # Merge: se filtros aplicados, preserva linhas existentes que nao casam
    linhas_existentes = (
        carregar_csv_existente() if (args.apenas or args.apenas_extrator) else []
    )
    pares_a_refazer = {(p.stem, e[0]) for p in pdfs for e in extratores}
    linhas_preservadas = [
        ln for ln in linhas_existentes if (ln["pdf"], ln["extrator"]) not in pares_a_refazer
    ]
    print(f"Linhas existentes preservadas: {len(linhas_preservadas)}")

    novas: list[dict[str, str]] = []
    inicio = time.perf_counter()

    for pdf in pdfs:
        gabarito_path = DIR_GABARITOS / f"{pdf.stem}.md"
        if not gabarito_path.exists():
            print(f"  [SKIP] {pdf.name} sem gabarito v2 em {gabarito_path}")
            continue
        gabarito = gabarito_path.read_text(encoding="utf-8")

        print(f"\n=== {pdf.stem} ===")
        for nome_extrator, extrair in extratores:
            linha = rodar_extracao(pdf, nome_extrator, extrair, gabarito)
            novas.append(linha)

    todas = linhas_preservadas + novas
    salvar_csv(todas)

    agregados = calcular_agregados(todas)
    PATH_JSON.write_text(
        json.dumps(agregados, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"\n=== FIM === {len(novas)} novas extracoes em "
        f"{time.perf_counter() - inicio:.1f}s"
    )
    print(f"CSV: {PATH_CSV.relative_to(RAIZ)}")
    print(f"JSON: {PATH_JSON.relative_to(RAIZ)}")
    print()
    print("Agregados (media por extrator):")
    for nome, ag in agregados.items():
        print(
            f"  {nome:30s}  cob={ag['cobertura_palavras']:.3f}  "
            f"lev={ag['sim_levenshtein']:.3f}  bi={ag['sim_bigramas']:.3f}  "
            f"tam={ag['tamanho_relativo']:.2f}  qexc={ag['qualidade_excesso']:.3f}  "
            f"({ag['tempo_seg_medio']:.1f}s, n={int(ag['n_pdfs_sem_erro'])})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
