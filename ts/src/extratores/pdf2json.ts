/**
 * Extrator pdf2json: parser baseado em eventos, retorna o PDF inteiro
 * como JSON estruturado (Pages -> Texts -> Runs).
 *
 * Usamos `getRawTextContent()` apos parse, que ja faz a remontagem em
 * texto plano (com page-break "----------------Page N----------------").
 *
 * `parseBuffer` eh sincrono (dispara eventos), entao envolvemos em
 * Promise. `needRawText=true` no construtor ativa o getRawTextContent.
 */

import { readFileSync } from 'node:fs';

import PDFParser from 'pdf2json';
import type { Output } from 'pdf2json';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'pdf2json';

function parsearBuffer(buffer: Buffer): Promise<{ texto: string; numPaginas: number }> {
  return new Promise((resolve, reject) => {
    const parser = new PDFParser(null, true);
    parser.on('pdfParser_dataError', (err) => {
      const e = 'parserError' in err ? err.parserError : err;
      reject(e);
    });
    parser.on('pdfParser_dataReady', (data: Output) => {
      const texto = parser.getRawTextContent();
      resolve({ texto, numPaginas: data.Pages.length });
    });
    parser.parseBuffer(buffer);
  });
}

export async function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  const inicio = performance.now();
  try {
    const buffer = readFileSync(caminhoPdf);
    const { texto, numPaginas } = await parsearBuffer(buffer);
    const paginas = texto.split(/----------------Page \(\d+\) Break----------------\r?\n?/);
    return {
      extrator: NOME_EXTRATOR,
      texto,
      paginas: paginas.filter((p) => p.length > 0),
      numPaginas,
      numCaracteres: texto.length,
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
