/**
 * Extrator hibrido: mupdf pra texto + tesseract.js so nas paginas
 * pobres em texto.
 *
 * Estrategia: pra cada pagina, extrai texto via mupdf. Se a pagina tem
 * mais que LIMIAR_CHARS de texto, usa o texto direto. Se tem menos,
 * roda OCR (renderiza com mupdf, joga no tesseract.js).
 *
 * Vantagem sobre OCR puro: rapido nas paginas que ja tem texto (>90%
 * dos casos em slides academicos), so paga o custo de OCR onde
 * realmente precisa.
 *
 * Vantagem sobre extrator de texto puro: nao perde os ~22% de
 * conteudo que esta em screenshots.
 */

import { readFileSync } from 'node:fs';

import * as mupdf from 'mupdf';
import { createWorker, type Worker } from 'tesseract.js';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido';

const LANG = 'por+eng';
const ESCALA_RENDER = 2;
const LIMIAR_CHARS = 100; // abaixo disso, considera "pagina pobre" e roda OCR

async function ocrDoPng(worker: Worker, pngBuffer: Buffer): Promise<string> {
  const { data } = await worker.recognize(pngBuffer);
  return data.text;
}

function renderizarPagina(page: mupdf.PDFPage | mupdf.Page, escala: number): Buffer {
  const matriz = mupdf.Matrix.scale(escala, escala);
  const pixmap = page.toPixmap(matriz, mupdf.ColorSpace.DeviceRGB, false, true);
  const png = pixmap.asPNG();
  pixmap.destroy();
  return Buffer.from(png);
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

    // Primeiro passo: extrai texto de todas as paginas via mupdf.
    // Marca quais precisam de OCR.
    const textosTexto: string[] = [];
    const precisamOcr: number[] = [];
    for (let i = 0; i < numPaginas; i++) {
      const page = doc.loadPage(i);
      const stext = page.toStructuredText('preserve-whitespace');
      const t = stext.asText();
      stext.destroy();
      textosTexto.push(t);
      if (t.length < LIMIAR_CHARS) precisamOcr.push(i);
      page.destroy();
    }

    // So cria o worker do tesseract se precisa.
    if (precisamOcr.length > 0) {
      worker = await createWorker(LANG, 1, { logger: () => {} });
      for (const i of precisamOcr) {
        const page = doc.loadPage(i);
        try {
          const png = renderizarPagina(page, ESCALA_RENDER);
          const textoOcr = await ocrDoPng(worker, png);
          // Mantem o que mupdf extraiu (pode ter algum texto curto) +
          // o que OCR achou.
          textosTexto[i] =
            textosTexto[i]!.length === 0
              ? textoOcr
              : `${textosTexto[i]}\n${textoOcr}`;
        } finally {
          page.destroy();
        }
      }
    }

    const texto = textosTexto.join('\n\n');
    return {
      extrator: NOME_EXTRATOR,
      texto,
      paginas: textosTexto,
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
