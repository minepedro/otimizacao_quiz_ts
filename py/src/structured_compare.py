"""
Comparacao 2-a-2 entre JSONs estruturados de extratores diferentes.

Sem ground truth — apenas mede a CONCORDANCIA par-a-par. Util pra
saber onde Docling, MinerU e nosso pipeline divergem na estruturacao.

Estrategia:
  1. Pra cada bloco do extrator A, encontra o bloco do extrator B mais
     proximo (por bbox + similaridade textual).
  2. Conta quantos pares "casaram" (acima do threshold).
  3. Dos casados, conta quantos tem o MESMO tipo.
  4. Reporta:
     - taxa_match: |pares casados| / |blocks de A|
     - taxa_tipo: |pares com mesmo tipo| / |pares casados|
     - distribuicao de tipos por extrator (independente de match)

Output: dict comparavel + tabela legivel.

ATENCAO: bbox de extratores diferentes podem estar em coords diferentes
(PDF points vs pixels da imagem). A comparacao por TEXTO eh mais
confiavel; por BBOX eh complementar quando ambos estao nas mesmas coords.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .schemas import BlocoExtraido


# ============================================================
# Helpers
# ============================================================


def _remover_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _normalizar(s: str) -> str:
    s = _remover_acentos(s).lower()
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^\w]+", _normalizar(s)) if len(t) >= 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return 0.0 if union == 0 else inter / union


def _iou_bbox(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU de dois bboxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1.0)
    area_b = max((bx2 - bx1) * (by2 - by1), 1.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ============================================================
# Comparacao 2-a-2
# ============================================================


# Threshold pra considerar 2 blocos como "o mesmo bloco"
JACCARD_TEXTO_MIN = 0.5  # 50% de tokens comuns


@dataclass
class CompararPar:
    """Resultado de comparar extrator A vs extrator B."""

    extrator_a: str
    extrator_b: str
    n_blocks_a: int
    n_blocks_b: int
    n_casados: int
    n_tipo_concorda: int
    iou_medio_pares: float
    distribuicao_a: dict[str, int] = field(default_factory=dict)
    distribuicao_b: dict[str, int] = field(default_factory=dict)

    @property
    def taxa_match(self) -> float:
        return self.n_casados / self.n_blocks_a if self.n_blocks_a else 0.0

    @property
    def taxa_tipo(self) -> float:
        return self.n_tipo_concorda / self.n_casados if self.n_casados else 0.0


def _melhor_par_por_pagina(
    bloco_a: BlocoExtraido,
    blocos_b_pagina: list[BlocoExtraido],
) -> tuple[BlocoExtraido | None, float, float]:
    """Acha o bloco em B mais similar ao A. Retorna (bloco, jaccard, iou_bbox)."""
    if not blocos_b_pagina:
        return None, 0.0, 0.0
    tok_a = _tokens(bloco_a.text)
    melhor: BlocoExtraido | None = None
    melhor_jac = 0.0
    melhor_iou = 0.0
    for b in blocos_b_pagina:
        jac = _jaccard(tok_a, _tokens(b.text))
        iou = _iou_bbox(bloco_a.bbox, b.bbox) if (bloco_a.bbox and b.bbox) else 0.0
        # Score combinado privilegia texto; bbox eh tie-breaker
        score = jac + 0.05 * iou
        if score > melhor_jac:
            melhor_jac = score
            melhor = b
            melhor_iou = iou
    # Recalcula jaccard puro pra retornar
    if melhor:
        return melhor, _jaccard(tok_a, _tokens(melhor.text)), melhor_iou
    return None, 0.0, 0.0


def comparar_pares(
    blocks_a: list[BlocoExtraido],
    blocks_b: list[BlocoExtraido],
    nome_a: str,
    nome_b: str,
) -> CompararPar:
    """Compara 2 listas de BlocoExtraido. Retorna CompararPar."""
    blocks_b_por_pagina: dict[int, list[BlocoExtraido]] = {}
    for b in blocks_b:
        blocks_b_por_pagina.setdefault(b.page_idx, []).append(b)

    n_casados = 0
    n_tipo = 0
    iou_acum = 0.0
    iou_count = 0

    for ba in blocks_a:
        if not ba.text or len(ba.text) < 5:
            continue
        candidatos = blocks_b_por_pagina.get(ba.page_idx, [])
        match, jac, iou = _melhor_par_por_pagina(ba, candidatos)
        if match is None or jac < JACCARD_TEXTO_MIN:
            continue
        n_casados += 1
        if ba.type == match.type:
            n_tipo += 1
        if iou > 0:
            iou_acum += iou
            iou_count += 1

    iou_medio = iou_acum / iou_count if iou_count else 0.0

    return CompararPar(
        extrator_a=nome_a,
        extrator_b=nome_b,
        n_blocks_a=len(blocks_a),
        n_blocks_b=len(blocks_b),
        n_casados=n_casados,
        n_tipo_concorda=n_tipo,
        iou_medio_pares=round(iou_medio, 3),
        distribuicao_a=dict(Counter(b.type for b in blocks_a)),
        distribuicao_b=dict(Counter(b.type for b in blocks_b)),
    )


def comparar_tres_vias(
    blocks_por_extrator: dict[str, list[BlocoExtraido]],
) -> dict[str, dict[str, CompararPar]]:
    """Comparacao all-pairs entre N extratores. Retorna matriz [a][b] -> CompararPar."""
    nomes = list(blocks_por_extrator.keys())
    matriz: dict[str, dict[str, CompararPar]] = {a: {} for a in nomes}
    for a in nomes:
        for b in nomes:
            if a == b:
                continue
            matriz[a][b] = comparar_pares(
                blocks_por_extrator[a], blocks_por_extrator[b], a, b
            )
    return matriz


def formatar_matriz(matriz: dict[str, dict[str, CompararPar]]) -> str:
    """Formata matriz de comparacao numa tabela texto."""
    nomes = list(matriz.keys())
    out = []

    out.append("=== Distribuicao de tipos por extrator ===")
    tipos_todos: set[str] = set()
    for a in nomes:
        for b, par in matriz[a].items():
            tipos_todos.update(par.distribuicao_a.keys())
            tipos_todos.update(par.distribuicao_b.keys())
            break  # so precisa de 1 par
    tipos_lista = sorted(tipos_todos)
    cab = f"{'tipo':15s} | " + " | ".join(f"{n:18s}" for n in nomes)
    out.append(cab)
    out.append("-" * len(cab))
    distrib_por_extrator: dict[str, dict[str, int]] = {n: {} for n in nomes}
    for a in nomes:
        for b in matriz[a].values():
            distrib_por_extrator[a] = b.distribuicao_a
            break
    for tipo in tipos_lista:
        linha = f"{tipo:15s} | " + " | ".join(
            f"{distrib_por_extrator[n].get(tipo, 0):18d}" for n in nomes
        )
        out.append(linha)

    out.append("")
    out.append("=== Concordancia par-a-par ===")
    out.append(
        f"{'A vs B':40s} | {'casados/A':12s} | {'taxa_match':10s} | {'tipo_ok':10s} | {'taxa_tipo':10s} | {'iou_medio':10s}"
    )
    out.append("-" * 110)
    for a in nomes:
        for b in nomes:
            if a == b:
                continue
            par = matriz[a][b]
            par_nome = f"{a:18s} vs {b:18s}"
            out.append(
                f"{par_nome:40s} | {par.n_casados:>5d}/{par.n_blocks_a:<5d} | "
                f"{par.taxa_match:.3f}      | {par.n_tipo_concorda:>5d}      | "
                f"{par.taxa_tipo:.3f}      | {par.iou_medio_pares:.3f}"
            )

    return "\n".join(out)
