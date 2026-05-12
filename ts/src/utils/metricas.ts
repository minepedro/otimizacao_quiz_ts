/**
 * Metricas de similaridade entre texto extraido e gabarito.
 *
 * Cinco metricas, todas calculadas sobre texto normalizado (lowercase,
 * sem acento, espacos colapsados):
 *
 *  - similaridadeLevenshtein: 1 - dist/max(len). Caractere por caractere.
 *  - coberturaPalavras: |palavras_extraido inter palavras_gabarito| / |palavras_gabarito|.
 *  - similaridadeBigramas: |bigramas em comum| / |union bigramas|. Robusto a reordenacao.
 *  - tamanhoRelativo: len(extraido) / len(gabarito). 1.0 = igual; <1 truncou; >1 ruido.
 *  - qualidadeExcesso: das palavras que existem APENAS no extraido (nao
 *    no gabarito), que % parecem palavras pt-BR validas? Mede se o
 *    "extra" eh conteudo legitimo (que o gabarito deixou passar) ou
 *    ruido de OCR. 1.0 = todo o extra parece valido; 0.0 = todo lixo.
 *    Sigla curta (<=5 chars maiusculas) eh aceita como valida pra nao
 *    punir STP/PDCA/JIT/5S.
 */

import { distance as levenshteinDistance } from 'fastest-levenshtein';

export interface Metricas {
  similaridadeLevenshtein: number;
  coberturaPalavras: number;
  similaridadeBigramas: number;
  tamanhoRelativo: number;
  qualidadeExcesso: number;
}

function normalizar(s: string): string {
  return s
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenizar(s: string): Set<string> {
  const tokens = s.split(/[^\p{L}\p{N}]+/u).filter((t) => t.length > 0);
  return new Set(tokens);
}

function bigramas(s: string): Set<string> {
  const set = new Set<string>();
  for (let i = 0; i < s.length - 1; i++) set.add(s.slice(i, i + 2));
  return set;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter += 1;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
}

function cobertura(extraido: Set<string>, gabarito: Set<string>): number {
  if (gabarito.size === 0) return extraido.size === 0 ? 1 : 0;
  let inter = 0;
  for (const x of gabarito) if (extraido.has(x)) inter += 1;
  return inter / gabarito.size;
}

/** Heuristica pra "parece palavra pt-BR valida" sem dicionario.
 *  Trade-off: aceita siglas curtas (STP, PDCA), rejeita lixo OCR
 *  (eca, mmr, pttiige). Pode errar em ambos sentidos mas eh util como
 *  proxy. Recebe palavra crua (preserva caso pra detectar sigla). */
function pareceValidoPtBr(palavraOriginal: string): boolean {
  if (palavraOriginal.length < 2 || palavraOriginal.length > 25) return false;

  // Sigla curta (<=5 chars) toda em maiusculo: aceita (STP, PDCA, OEE, JIT, 5S)
  if (palavraOriginal.length <= 5 && palavraOriginal === palavraOriginal.toUpperCase()) {
    // mas exige que tenha pelo menos 1 letra (nao so digitos)
    if (/[A-Z]/.test(palavraOriginal)) return true;
  }

  // Pra outras, normaliza e checa heuristica de "vogais e consoantes"
  const p = palavraOriginal
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .toLowerCase();

  // So letras (sem digitos) — palavras com numero misturado raramente sao validas
  if (!/^[a-z]+$/.test(p)) return false;

  // Pelo menos 1 vogal
  if (!/[aeiouy]/.test(p)) return false;

  // Sem 5+ consoantes consecutivas (pttiige, mmr, etc)
  if (/[bcdfghjklmnpqrstvwxz]{5,}/.test(p)) return false;

  // Razao de vogais entre 0.15 e 0.85
  const vogais = (p.match(/[aeiouy]/g) ?? []).length;
  const razao = vogais / p.length;
  if (razao < 0.15 || razao > 0.85) return false;

  return true;
}

/** Das palavras que existem so no extraido (nao no gabarito), quantas %
 *  parecem validas? Devolve null se nao ha palavras exclusivas (nao da
 *  pra calcular). */
function qualidadeDoExcesso(
  textoExtraido: string,
  textoGabarito: string,
): number {
  const palGab = new Set(
    textoGabarito
      .normalize('NFD')
      .replace(/\p{M}/gu, '')
      .toLowerCase()
      .split(/[^\p{L}\p{N}]+/u)
      .filter((t) => t.length > 0),
  );
  const palavrasOriginaisExtraido = textoExtraido
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length > 0);

  // Pega palavras unicas exclusivas (em forma normalizada vs gabarito,
  // mas mantem original pra checar capitalizacao/sigla)
  const vistas = new Set<string>();
  let exclusivas = 0;
  let validas = 0;
  for (const original of palavrasOriginaisExtraido) {
    const norm = original
      .normalize('NFD')
      .replace(/\p{M}/gu, '')
      .toLowerCase();
    if (palGab.has(norm)) continue;
    if (vistas.has(norm)) continue;
    vistas.add(norm);
    exclusivas += 1;
    if (pareceValidoPtBr(original)) validas += 1;
  }

  if (exclusivas === 0) return 1; // sem excesso eh "qualidade perfeita"
  return validas / exclusivas;
}

export function calcularMetricas(extraido: string, gabarito: string): Metricas {
  const ext = normalizar(extraido);
  const gab = normalizar(gabarito);

  const maxLen = Math.max(ext.length, gab.length);
  const simLev =
    maxLen === 0 ? 1 : 1 - levenshteinDistance(ext, gab) / maxLen;

  const cob = cobertura(tokenizar(ext), tokenizar(gab));
  const simBi = jaccard(bigramas(ext), bigramas(gab));
  const tamRel = gab.length === 0 ? 0 : ext.length / gab.length;

  // qualidadeExcesso usa texto ORIGINAL (nao normalizado) pra detectar
  // siglas pela capitalizacao
  const qualExc = qualidadeDoExcesso(extraido, gabarito);

  return {
    similaridadeLevenshtein: simLev,
    coberturaPalavras: cob,
    similaridadeBigramas: simBi,
    tamanhoRelativo: tamRel,
    qualidadeExcesso: qualExc,
  };
}
