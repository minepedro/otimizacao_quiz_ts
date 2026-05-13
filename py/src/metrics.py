"""
Metricas de similaridade entre texto extraido e gabarito.

Port direto de `ts/src/utils/metricas.ts`. Mesmas 5 metricas,
mesmo comportamento, mesmos thresholds. Mantido fiel ao TS pra
permitir comparacao cross-stack via mesmo CSV
(`resultados/benchmark*.csv`).

Cinco metricas, todas calculadas sobre texto normalizado
(lowercase, sem acento, espacos colapsados):

  - sim_levenshtein: 1 - dist/max(len). Caractere por caractere.
  - cobertura_palavras: |palavras_extraido inter palavras_gabarito|
                       / |palavras_gabarito|.
  - sim_bigramas: |bigramas em comum| / |union bigramas|.
                  Robusto a reordenacao.
  - tamanho_relativo: len(extraido) / len(gabarito).
                     1.0 = igual; <1 truncou; >1 ruido.
  - qualidade_excesso: das palavras que existem APENAS no extraido
                      (nao no gabarito), que % parecem palavras pt-BR
                      validas? Mede se o "extra" eh conteudo legitimo
                      (que o gabarito deixou passar) ou ruido de OCR.
                      1.0 = todo extra parece valido; 0.0 = todo lixo.
                      Sigla curta (<=5 chars maiusculas) eh aceita
                      pra nao punir STP/PDCA/JIT/5S.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class Metricas:
    sim_levenshtein: float
    cobertura_palavras: float
    sim_bigramas: float
    tamanho_relativo: float
    qualidade_excesso: float


# Pre-compiladas (mais rapido em loops)
_RE_TOKENS = re.compile(r"[^\w]+", flags=re.UNICODE)  # split por nao-letra/digito
_RE_VOGAL = re.compile(r"[aeiouy]")
_RE_CONSOANTES = re.compile(r"[bcdfghjklmnpqrstvwxz]{5,}")
_RE_SO_LETRAS = re.compile(r"^[a-z]+$")
_RE_TEM_MAIUSCULA = re.compile(r"[A-Z]")


def _remover_acentos(s: str) -> str:
    """NFD + remove combining marks. Equivalente ao .normalize('NFD').replace(/\\p{M}/gu, '') do TS."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalizar(s: str) -> str:
    s = _remover_acentos(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenizar(s: str) -> set[str]:
    """Split por nao-letra/nao-digito (Unicode), filtra vazios, retorna set."""
    return {t for t in _RE_TOKENS.split(s) if t}


def bigramas(s: str) -> set[str]:
    return {s[i : i + 2] for i in range(len(s) - 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return 0.0 if union == 0 else inter / union


def cobertura(extraido: set[str], gabarito: set[str]) -> float:
    if not gabarito:
        return 1.0 if not extraido else 0.0
    inter = len(gabarito & extraido)
    return inter / len(gabarito)


def parece_valido_pt_br(palavra_original: str) -> bool:
    """Heuristica pra "parece palavra pt-BR valida" sem dicionario.

    Trade-off: aceita siglas curtas (STP, PDCA), rejeita lixo OCR
    (eca, mmr, pttiige). Pode errar em ambos sentidos mas eh util como
    proxy. Recebe palavra crua (preserva caso pra detectar sigla).
    """
    n = len(palavra_original)
    if n < 2 or n > 25:
        return False

    # Sigla curta (<=5 chars) toda em maiusculo: aceita (STP, PDCA, OEE, JIT, 5S)
    if n <= 5 and palavra_original == palavra_original.upper():
        if _RE_TEM_MAIUSCULA.search(palavra_original):
            return True

    # Pra outras, normaliza e checa heuristica de "vogais e consoantes"
    p = _remover_acentos(palavra_original).lower()

    # So letras (sem digitos)
    if not _RE_SO_LETRAS.match(p):
        return False

    # Pelo menos 1 vogal
    if not _RE_VOGAL.search(p):
        return False

    # Sem 5+ consoantes consecutivas (pttiige, mmr)
    if _RE_CONSOANTES.search(p):
        return False

    # Razao de vogais entre 0.15 e 0.85
    vogais = len(_RE_VOGAL.findall(p))
    razao = vogais / len(p)
    if razao < 0.15 or razao > 0.85:
        return False

    return True


def qualidade_do_excesso(texto_extraido: str, texto_gabarito: str) -> float:
    """% das palavras "extras" (so no extraido) que parecem pt-BR valido.

    Sem palavras exclusivas → retorna 1.0 (qualidade perfeita por
    construcao — nao ha excesso pra ser ruim).
    """
    pal_gab = {
        t for t in _RE_TOKENS.split(_remover_acentos(texto_gabarito).lower()) if t
    }
    palavras_originais_extraido = [
        t for t in _RE_TOKENS.split(texto_extraido) if t
    ]

    vistas: set[str] = set()
    exclusivas = 0
    validas = 0
    for original in palavras_originais_extraido:
        norm = _remover_acentos(original).lower()
        if norm in pal_gab:
            continue
        if norm in vistas:
            continue
        vistas.add(norm)
        exclusivas += 1
        if parece_valido_pt_br(original):
            validas += 1

    if exclusivas == 0:
        return 1.0
    return validas / exclusivas


def calcular_metricas(extraido: str, gabarito: str) -> Metricas:
    ext = normalizar(extraido)
    gab = normalizar(gabarito)

    max_len = max(len(ext), len(gab))
    sim_lev = 1.0 if max_len == 0 else 1.0 - Levenshtein.distance(ext, gab) / max_len

    cob = cobertura(tokenizar(ext), tokenizar(gab))
    sim_bi = jaccard(bigramas(ext), bigramas(gab))
    tam_rel = 0.0 if not gab else len(ext) / len(gab)

    # qualidade_excesso usa texto ORIGINAL (nao normalizado) pra detectar sigla
    qual_exc = qualidade_do_excesso(extraido, gabarito)

    return Metricas(
        sim_levenshtein=sim_lev,
        cobertura_palavras=cob,
        sim_bigramas=sim_bi,
        tamanho_relativo=tam_rel,
        qualidade_excesso=qual_exc,
    )
