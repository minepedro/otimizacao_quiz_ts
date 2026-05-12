/**
 * Extrator OCR puro 100% MIT: renderiza cada pagina como PNG via unpdf
 * (que usa pdf.js + @napi-rs/canvas) e passa pro tesseract.js.
 *
 * Equivalente funcional ao `tesseract.ts`, mas sem dependencia AGPL —
 * pra ser usado no tutor-ai sem dor de cabeca de licenca.
 *
 * Trade-off vs `tesseract.ts` (mupdf-based):
 *   - Render via canvas pode ter pequena diferenca de qualidade vs
 *     mupdf (a comparar no benchmark).
 *   - Performance similar.
 *   - Licenca: pdfjs-dist (Apache-2.0) + @napi-rs/canvas (MIT) +
 *     tesseract.js (Apache-2.0) — todas permissivas.
 */

import { readFileSync } from 'node:fs';

import { createIsomorphicCanvasFactory, getDocumentProxy, renderPageAsImage } from 'unpdf';
import { createWorker, type Worker } from 'tesseract.js';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'tesseract-unpdf';

const LANG = 'por+eng';
const ESCALA_RENDER = 2;

export async function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  const inicio = performance.now();
  let worker: Worker | null = null;
  try {
    const buffer = readFileSync(caminhoPdf);
    const data = new Uint8Array(buffer);

    // Setup canvas pra Node uma vez, e cria o doc JA com a CanvasFactory.
    // Reutilizar o mesmo doc nas chamadas de renderPageAsImage evita o
    // DataCloneError (pdf.js "transfere" o buffer entre chamadas, ele
    // fica indisponivel a partir da segunda).
    const canvasImport = () => import('@napi-rs/canvas');
    const CanvasFactory = await createIsomorphicCanvasFactory(canvasImport);
    const doc = await getDocumentProxy(data, { CanvasFactory });
    const numPaginas = doc.numPages;

    worker = await createWorker(LANG, 1, { logger: () => {} });

    const paginas: string[] = [];
    for (let i = 1; i <= numPaginas; i++) {
      const png = await renderPageAsImage(doc, i, {
        scale: ESCALA_RENDER,
        canvasImport,
      });
      const buf = Buffer.from(png);
      const { data: ocrData } = await worker.recognize(buf);
      paginas.push(ocrData.text);
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
  }
}
