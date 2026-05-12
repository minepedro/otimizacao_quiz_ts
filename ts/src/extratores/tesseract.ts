/**
 * Extrator OCR puro: renderiza cada pagina como PNG (via mupdf, 2x scale)
 * e passa pro tesseract.js (Tesseract OCR rodando em WebAssembly, idioma
 * portugues + ingles).
 *
 * MUITO mais lento que extratores de texto puro (~5-15s por pagina).
 *
 * Util pra:
 *   1. PDF que eh scan (so imagem) — extratores de texto retornam vazio.
 *   2. Comparar contra extratores de texto pra ver "quanto OCR ganha".
 *
 * Pra producao real, usar o `hibrido` (mupdf onde tem texto, OCR onde
 * nao tem). OCR puro aqui eh baseline.
 */

import { readFileSync } from 'node:fs';

import * as mupdf from 'mupdf';
import { createWorker, type Worker } from 'tesseract.js';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'tesseract';

const LANG = 'por+eng';
const ESCALA_RENDER = 2; // 2x = melhor precisao do OCR sem inflar muito

async function ocrPagina(worker: Worker, pngBuffer: Buffer): Promise<string> {
  const { data } = await worker.recognize(pngBuffer);
  return data.text;
}

function renderizarPagina(
  doc: mupdf.Document,
  indice: number,
  escala: number,
): Buffer {
  const page = doc.loadPage(indice);
  try {
    const matriz = mupdf.Matrix.scale(escala, escala);
    const pixmap = page.toPixmap(matriz, mupdf.ColorSpace.DeviceRGB, false, true);
    const png = pixmap.asPNG();
    pixmap.destroy();
    return Buffer.from(png);
  } finally {
    page.destroy();
  }
}

export async function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  const inicio = performance.now();
  let worker: Worker | null = null;
  let doc: mupdf.Document | null = null;
  try {
    const buffer = readFileSync(caminhoPdf);
    doc = mupdf.Document.openDocument(
      new Uint8Array(buffer),
      'application/pdf',
    );
    const numPaginas = doc.countPages();

    worker = await createWorker(LANG, 1, { logger: () => {} });

    const paginas: string[] = [];
    for (let i = 0; i < numPaginas; i++) {
      const png = renderizarPagina(doc, i, ESCALA_RENDER);
      const texto = await ocrPagina(worker, png);
      paginas.push(texto);
    }

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
  } finally {
    if (worker) await worker.terminate();
    if (doc) doc.destroy();
  }
}
