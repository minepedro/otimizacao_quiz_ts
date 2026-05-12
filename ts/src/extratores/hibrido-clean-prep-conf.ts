/**
 * Combinacao building blocks 3 + 2: pre-processamento + confidence.
 *
 * Tese: prep isolado deu MUITO ruim (tam=2.67) porque binarizacao gera
 * texto inventado. Conf filtra esse lixo. Hipotese: o conf pode salvar
 * o prep, fazendo binarizacao funcionar so quando vale.
 *
 * Custo: tempo ~30s/PDF (prep eh o gargalo).
 */

import { extrairComOpcoes } from './_hibrido-img-base.js';
import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido-clean-prep-conf';

export function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  return extrairComOpcoes(
    caminhoPdf,
    { preProcessar: true, confidenceMin: 60 },
    NOME_EXTRATOR,
  );
}
