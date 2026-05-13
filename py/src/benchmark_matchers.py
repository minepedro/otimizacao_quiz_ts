"""
Benchmark dos matchers do hibrido_v2 contra gabarito v2.

Roda os 4 matchers em todos os 9 PDFs e gera tabela comparativa.

Uso:
  py/.venv/Scripts/python.exe -m src.benchmark_matchers
  py/.venv/Scripts/python.exe -m src.benchmark_matchers --apenas "Aula 04"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

from .extractors import hibrido_v2
from .extractors.hibrido_v2 import MATCHERS
from .metrics import calcular_metricas

RAIZ = Path(__file__).resolve().parents[2]
DIR_PDFS = RAIZ / "dados" / "pdfs"
DIR_GAB = RAIZ / "dados" / "gabaritos-v2-md"
DIR_RESULTADOS = RAIZ / "resultados"
PATH_CSV = DIR_RESULTADOS / "benchmark-matchers.csv"
PATH_JSON = DIR_RESULTADOS / "benchmark-matchers.json"


COLUNAS_CSV = [
    "matcher",
    "pdf",
    "num_paginas",
    "num_blocks",
    "num_tipos_distintos",
    "tempo_seg",
    "num_caracteres",
    "sim_levenshtein",
    "cobertura_palavras",
    "sim_bigramas",
    "tamanho_relativo",
    "qualidade_excesso",
    "erro",
]


def fmt(v: float, casas: int = 4) -> str:
    return f"{v:.{casas}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark dos matchers do hibrido_v2.")
    parser.add_argument("--apenas", type=str, default=None, help="Filtra PDFs por substring.")
    parser.add_argument("--apenas-matcher", type=str, default=None, help="Filtra matchers.")
    args = parser.parse_args()

    pdfs = sorted(DIR_PDFS.glob("*.pdf"))
    if args.apenas:
        pdfs = [p for p in pdfs if args.apenas.lower() in p.name.lower()]
    if not pdfs:
        print("Nenhum PDF.")
        return 1

    matcher_names = list(MATCHERS.keys())
    if args.apenas_matcher:
        matcher_names = [n for n in matcher_names if args.apenas_matcher.lower() in n.lower()]
    if not matcher_names:
        print(f"Nenhum matcher (opcoes: {list(MATCHERS)})")
        return 1

    print(f"Matchers: {matcher_names}")
    print(f"PDFs: {len(pdfs)}")
    print(f"Total extracoes: {len(matcher_names) * len(pdfs)}")
    print()

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    linhas_csv: list[dict[str, str]] = []
    inicio = time.perf_counter()

    extrair_fns = {
        "best_0": hibrido_v2.extrair_best_0,
        "best_02": hibrido_v2.extrair_best_02,
        "greedy_05": hibrido_v2.extrair_greedy_05,
        "confidence": hibrido_v2.extrair_confidence,
    }

    for matcher_name in matcher_names:
        extrair = extrair_fns[matcher_name]
        print(f"=== Matcher: {matcher_name} ===")

        for pdf in pdfs:
            gab_path = DIR_GAB / f"{pdf.stem}.md"
            if not gab_path.exists():
                continue
            gabarito = gab_path.read_text(encoding="utf-8")

            print(f"  [{pdf.stem[:50]:50s}]", end=" ", flush=True)
            try:
                t0 = time.time()
                r = extrair(pdf)
                elapsed = time.time() - t0
            except Exception as e:
                print(f"ERRO: {type(e).__name__}: {e}")
                linhas_csv.append({
                    "matcher": matcher_name, "pdf": pdf.stem,
                    "num_paginas": "0", "num_blocks": "0", "num_tipos_distintos": "0",
                    "tempo_seg": "0", "num_caracteres": "0",
                    "sim_levenshtein": "", "cobertura_palavras": "",
                    "sim_bigramas": "", "tamanho_relativo": "",
                    "qualidade_excesso": "", "erro": f"{type(e).__name__}: {e}",
                })
                continue

            if r.erro:
                print(f"ERRO: {r.erro}")
                linhas_csv.append({
                    "matcher": matcher_name, "pdf": pdf.stem,
                    "num_paginas": str(r.num_paginas), "num_blocks": str(len(r.blocks)),
                    "num_tipos_distintos": "0",
                    "tempo_seg": fmt(r.tempo_segundos, 3), "num_caracteres": str(r.num_caracteres),
                    "sim_levenshtein": "", "cobertura_palavras": "",
                    "sim_bigramas": "", "tamanho_relativo": "",
                    "qualidade_excesso": "", "erro": r.erro[:200],
                })
                continue

            m = calcular_metricas(r.markdown, gabarito)
            tipos = Counter(b.type for b in r.blocks)
            print(
                f"cob={fmt(m.cobertura_palavras, 3)} "
                f"lev={fmt(m.sim_levenshtein, 3)} "
                f"bi={fmt(m.sim_bigramas, 3)} "
                f"qexc={fmt(m.qualidade_excesso, 3)} "
                f"({elapsed:.1f}s, {len(r.blocks)} blocks, {len(tipos)} tipos)"
            )
            linhas_csv.append({
                "matcher": matcher_name,
                "pdf": pdf.stem,
                "num_paginas": str(r.num_paginas),
                "num_blocks": str(len(r.blocks)),
                "num_tipos_distintos": str(len(tipos)),
                "tempo_seg": fmt(r.tempo_segundos, 3),
                "num_caracteres": str(r.num_caracteres),
                "sim_levenshtein": fmt(m.sim_levenshtein),
                "cobertura_palavras": fmt(m.cobertura_palavras),
                "sim_bigramas": fmt(m.sim_bigramas),
                "tamanho_relativo": fmt(m.tamanho_relativo),
                "qualidade_excesso": fmt(m.qualidade_excesso),
                "erro": "",
            })
        print()

    # Salva CSV
    with PATH_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUNAS_CSV)
        writer.writeheader()
        for ln in linhas_csv: writer.writerow(ln)

    # Agregados por matcher
    agregados: dict[str, dict[str, float]] = {}
    for matcher_name in matcher_names:
        linhas = [ln for ln in linhas_csv if ln["matcher"] == matcher_name and not ln["erro"]]
        if not linhas: continue
        n = len(linhas)
        def media(col: str) -> float:
            return round(sum(float(ln[col]) for ln in linhas) / n, 4)
        agregados[matcher_name] = {
            "n_pdfs_sem_erro": n,
            "tempo_seg_medio": media("tempo_seg"),
            "num_blocks_medio": round(sum(int(ln["num_blocks"]) for ln in linhas) / n, 1),
            "num_tipos_medio": round(sum(int(ln["num_tipos_distintos"]) for ln in linhas) / n, 1),
            "sim_levenshtein": media("sim_levenshtein"),
            "cobertura_palavras": media("cobertura_palavras"),
            "sim_bigramas": media("sim_bigramas"),
            "tamanho_relativo": media("tamanho_relativo"),
            "qualidade_excesso": media("qualidade_excesso"),
        }

    PATH_JSON.write_text(json.dumps(agregados, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 100)
    print(f"AGREGADO ({time.perf_counter() - inicio:.1f}s total)")
    print("=" * 100)
    print(f"{'matcher':14s}  {'cob':>6s}  {'lev':>6s}  {'bi':>6s}  {'tam':>5s}  {'qexc':>6s}  {'tempo':>6s}  {'blocks':>7s}  {'tipos':>6s}")
    for nome, ag in agregados.items():
        print(
            f"{nome:14s}  "
            f"{ag['cobertura_palavras']:.3f}   "
            f"{ag['sim_levenshtein']:.3f}   "
            f"{ag['sim_bigramas']:.3f}   "
            f"{ag['tamanho_relativo']:.2f}    "
            f"{ag['qualidade_excesso']:.3f}   "
            f"{ag['tempo_seg_medio']:5.1f}s   "
            f"{ag['num_blocks_medio']:>6.1f}   "
            f"{ag['num_tipos_medio']:>5.1f}"
        )
    print()
    print(f"CSV: {PATH_CSV.relative_to(RAIZ)}")
    print(f"JSON: {PATH_JSON.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
