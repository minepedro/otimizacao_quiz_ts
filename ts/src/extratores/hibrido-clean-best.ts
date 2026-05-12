/**
 * Building block 1 isolado: tessdata_best.
 *
 * Igual ao hibrido-unpdf-img-hash-clean, mas usa o tessdata_best
 * (modelos LSTM mais precisos do tesseract). Carregamento inicial ~100ms
 * mais lento, qualidade melhor em fontes incomuns/manuscrito.
 */

import { extrairComOpcoes } from './_hibrido-img-base.js';
import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido-clean-best';

export function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  return extrairComOpcoes(caminhoPdf, { tessdataBest: true }, NOME_EXTRATOR);
}
