/**
 * Building block 4 isolado: OCR em multiplas escalas.
 *
 * Igual ao hibrido-unpdf-img-hash-clean, mas roda OCR em duas escalas
 * (1x e 2x) e mescla resultados. Pega o texto mais longo como base e
 * adiciona palavras exclusivas do outro. Custo: ~2x mais tempo de OCR.
 * Util quando texto pequeno (capa, rodape, label) nao eh captado na
 * escala original.
 */

import { extrairComOpcoes } from './_hibrido-img-base.js';
import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido-clean-multi';

export function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  return extrairComOpcoes(caminhoPdf, { escalasOcr: [1, 2] }, NOME_EXTRATOR);
}
