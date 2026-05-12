/**
 * Extrator hibrido v2: combina AREA de imagem + DENSIDADE de texto pra
 * decidir quando rodar OCR.
 *
 * Lacuna do v1 (`hibrido.ts`): usava so chars/pagina, nao detectava
 * paginas com screenshot grande + legenda em volta (Aula 2).
 *
 * Lacuna da primeira tentativa do v2: usava so area de imagem, mas
 * slides PowerPoint tem background cobrindo 100% da pagina (mupdf
 * reporta como image block), entao TODA pagina disparava OCR e o
 * texto duplicava (mupdf + OCR concatenados, tam relativo virou 2.0).
 *
 * Estrategia atual:
 *   1. Pra cada pagina, calcula razaoArea = area_imagem / area_pagina
 *      (usando o JSON do mupdf structured text com 'preserve-images').
 *   2. Pra cada pagina, conta chars extraidos por mupdf.
 *   3. Roda OCR se razaoArea >= 0.3 E chars < 500 — pagina
 *      "visualmente cheia mas pouco texto vetorial" = provavel
 *      screenshot/imagem com conteudo.
 *   4. Quando dispara OCR, USA o texto do OCR no lugar do mupdf (nao
 *      soma) — evita duplicacao do texto que ja estava na imagem.
 *
 * Pra paginas onde mupdf pegou bem (texto vetorial puro), nao roda OCR
 * — economiza tempo e evita ruido.
 */

import { readFileSync } from 'node:fs';

import * as mupdf from 'mupdf';
import { createWorker, type Worker } from 'tesseract.js';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido2';

const LANG = 'por+eng';
const ESCALA_RENDER = 2;
const LIMIAR_AREA_IMAGEM = 0.3; // >= 30% da area em imagem
const LIMIAR_CHARS_PAGINA_POBRE = 500; // E < 500 chars de texto vetorial

interface BboxObj {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface BlocoStructuredText {
  type?: string;
  bbox?: BboxObj;
}

function areaImagemNaPagina(page: mupdf.PDFPage | mupdf.Page): {
  razaoArea: number;
  areaPagina: number;
} {
  const bounds = page.getBounds();
  const areaPagina = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
  if (areaPagina <= 0) return { razaoArea: 0, areaPagina: 0 };

  let areaImagem = 0;
  try {
    const stext = page.toStructuredText('preserve-images');
    const json = JSON.parse(stext.asJSON()) as { blocks?: BlocoStructuredText[] };
    if (json.blocks) {
      for (const b of json.blocks) {
        if (b.type === 'image' && b.bbox) {
          areaImagem += b.bbox.w * b.bbox.h;
        }
      }
    }
    stext.destroy();
  } catch {
    // ignora se asJSON nao funcionar
  }

  return { razaoArea: areaImagem / areaPagina, areaPagina };
}

function renderizarPagina(page: mupdf.PDFPage | mupdf.Page, escala: number): Buffer {
  const matriz = mupdf.Matrix.scale(escala, escala);
  const pixmap = page.toPixmap(matriz, mupdf.ColorSpace.DeviceRGB, false, true);
  const png = pixmap.asPNG();
  pixmap.destroy();
  return Buffer.from(png);
}

async function ocrDoPng(worker: Worker, pngBuffer: Buffer): Promise<string> {
  const { data } = await worker.recognize(pngBuffer);
  return data.text;
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

    // Passo 1: extrai texto via mupdf E decide quais precisam de OCR.
    // Criterio: razaoArea de imagem alta E texto vetorial baixo
    // (pagina "visualmente cheia mas pouco texto" = provavel imagem
    // grande com conteudo).
    const textosTexto: string[] = [];
    const precisamOcr: number[] = [];
    for (let i = 0; i < numPaginas; i++) {
      const page = doc.loadPage(i);
      const stext = page.toStructuredText('preserve-whitespace');
      const t = stext.asText();
      stext.destroy();
      textosTexto.push(t);
      const { razaoArea } = areaImagemNaPagina(page);
      if (razaoArea >= LIMIAR_AREA_IMAGEM && t.length < LIMIAR_CHARS_PAGINA_POBRE) {
        precisamOcr.push(i);
      }
      page.destroy();
    }

    // Passo 2: roda OCR nas paginas marcadas e SUBSTITUI o texto
    // (nao concatena), pra evitar duplicacao do texto que ja estava
    // na imagem.
    if (precisamOcr.length > 0) {
      worker = await createWorker(LANG, 1, { logger: () => {} });
      for (const i of precisamOcr) {
        const page = doc.loadPage(i);
        try {
          const png = renderizarPagina(page, ESCALA_RENDER);
          const textoOcr = await ocrDoPng(worker, png);
          textosTexto[i] = textoOcr;
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
