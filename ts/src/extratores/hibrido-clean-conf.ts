/**
 * Building block 2 isolado: filtro por confidence.
 *
 * Igual ao hibrido-unpdf-img-hash-clean, mas descarta palavras OCR com
 * confidence < 60 (tesseract devolve 0-100). Ataca diretamente o
 * ruido visual ("pttiige", "eea", "mmr") que o tesseract reconhece com
 * baixa certeza.
 */

import { extrairComOpcoes } from './_hibrido-img-base.js';
import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido-clean-conf';

export function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  return extrairComOpcoes(caminhoPdf, { confidenceMin: 60 }, NOME_EXTRATOR);
}
