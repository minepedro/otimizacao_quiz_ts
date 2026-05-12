/**
 * Combinacao building blocks 2 + 1: confidence + tessdata_best.
 *
 * Tese: best le mais (cobertura ligeiramente maior, qexc menor — pega
 * mais palavras incluindo borderline). Conf filtra as borderline com
 * confidence baixa, deixando so as confiaveis. Combinacao "pegar mais e
 * filtrar".
 *
 * Custo: tempo ~10-12s/PDF (best eh ~40% mais lento que default).
 */

import { extrairComOpcoes } from './_hibrido-img-base.js';
import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido-clean-conf-best';

export function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  return extrairComOpcoes(
    caminhoPdf,
    { confidenceMin: 60, tessdataBest: true },
    NOME_EXTRATOR,
  );
}
