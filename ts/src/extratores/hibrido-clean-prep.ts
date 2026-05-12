/**
 * Building block 3 isolado: pre-processamento (binarizacao + contraste).
 *
 * Igual ao hibrido-unpdf-img-hash-clean, mas antes do OCR converte a
 * imagem pra preto-e-branco usando limiar adaptativo. Tesseract foi
 * treinado em texto preto sobre fundo branco — pode ajudar em imagens
 * coloridas / baixo contraste, pode atrapalhar em screenshots com cores
 * informativas.
 */

import { extrairComOpcoes } from './_hibrido-img-base.js';
import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido-clean-prep';

export function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  return extrairComOpcoes(caminhoPdf, { preProcessar: true }, NOME_EXTRATOR);
}
