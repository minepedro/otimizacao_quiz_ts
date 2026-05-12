/**
 * Extrator unpdf: wrapper moderno em cima do pdf.js, otimizado pra
 * runtimes serverless (Cloudflare Workers, Vercel Edge etc).
 *
 * API minima: extractText(uint8array, { mergePages: true }) -> { totalPages, text }.
 * Mantemos paginas individuais tambem chamando sem mergePages.
 */

import { readFileSync } from 'node:fs';

import { extractText } from 'unpdf';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'unpdf';

export async function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  const inicio = performance.now();
  try {
    const buffer = readFileSync(caminhoPdf);
    const data = new Uint8Array(buffer);
    const porPagina = await extractText(data);
    const merged = porPagina.text.join('\n\n');
    return {
      extrator: NOME_EXTRATOR,
      texto: merged,
      paginas: porPagina.text,
      numPaginas: porPagina.totalPages,
      numCaracteres: merged.length,
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
