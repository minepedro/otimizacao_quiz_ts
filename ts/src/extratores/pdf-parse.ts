/**
 * Extrator pdf-parse: lib mais simples e popular do ecossistema Node.
 *
 * Internamente usa pdf.js da Mozilla (versao antiga). API trivial: passa
 * buffer, recebe texto. Sem informacao de layout.
 *
 * Baseline do benchmark.
 */

import { readFileSync } from 'node:fs';

import pdfParse from 'pdf-parse';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'pdf-parse';

export async function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  const inicio = performance.now();
  try {
    const buffer = readFileSync(caminhoPdf);
    const data = await pdfParse(buffer);
    return {
      extrator: NOME_EXTRATOR,
      texto: data.text,
      paginas: data.text.split(/\f|\n{2,}/),
      numPaginas: data.numpages,
      numCaracteres: data.text.length,
      tempoSegundos: (performance.now() - inicio) / 1000,
      erro: null,
    };
  } catch (e) {
    const msg = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    return {
      extrator: NOME_EXTRATOR,
      texto: '',
      paginas: [],
      numPaginas: 0,
      numCaracteres: 0,
      tempoSegundos: (performance.now() - inicio) / 1000,
      erro: msg,
    };
  }
}
