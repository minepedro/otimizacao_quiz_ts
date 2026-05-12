/**
 * Combinacao building blocks 2 + 4: confidence + multi-escala.
 *
 * Tese: multi captura MAIS conteudo (cobertura alta), conf filtra o
 * RUIDO que o multi adiciona. Pode ser o melhor dos dois mundos.
 *
 * Custo: tempo ~22s/PDF (multi-escala domina).
 */

import { extrairComOpcoes } from './_hibrido-img-base.js';
import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido-clean-conf-multi';

export function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  return extrairComOpcoes(
    caminhoPdf,
    { confidenceMin: 60, escalasOcr: [1, 2] },
    NOME_EXTRATOR,
  );
}
