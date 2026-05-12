/**
 * Extrator mupdf.js: port WebAssembly do MuPDF (Artifex).
 *
 * Eh o motor por tras do PyMuPDF, que venceu o lab Python. Aposta forte
 * pra repetir a vitoria em TS.
 *
 * Por pagina: page.toStructuredText('preserve-whitespace').asText().
 *
 * Atencao: licenca AGPL-3.0-or-later (diferente das outras MIT/Apache).
 * Implicacoes pra distribuir junto do tutor-ai precisam ser avaliadas
 * antes de ir pra producao.
 */

import { readFileSync } from 'node:fs';

import * as mupdf from 'mupdf';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'mupdf';

export async function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  const inicio = performance.now();
  try {
    const buffer = readFileSync(caminhoPdf);
    const doc = mupdf.Document.openDocument(
      new Uint8Array(buffer),
      'application/pdf',
    );
    const numPaginas = doc.countPages();
    const paginas: string[] = [];
    for (let i = 0; i < numPaginas; i++) {
      const page = doc.loadPage(i);
      const stext = page.toStructuredText('preserve-whitespace');
      paginas.push(stext.asText());
      stext.destroy();
      page.destroy();
    }
    doc.destroy();
    const texto = paginas.join('\n\n');
    return {
      extrator: NOME_EXTRATOR,
      texto,
      paginas,
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
