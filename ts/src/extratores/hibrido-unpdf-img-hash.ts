/**
 * Extrator hibrido v3.1 (img-hash): refinamento do hibrido-unpdf-img.
 *
 * Duas mudancas importantes vs hibrido-unpdf-img:
 *
 * 1. DETECCAO DE BACKGROUND POR HASH (em vez de tamanho).
 *    - v1: marcava como background a imagem com TAMANHO mais frequente.
 *      Falhava em PDFs onde cada pagina tem uma imagem do mesmo tamanho
 *      mas com CONTEUDO diferente (resolucao manuscrita: cada scan tem
 *      ~2541x3498 mas hash diferente). Algoritmo via como background e
 *      ignorava tudo, perdendo OCR de tudo.
 *    - v2 (este): marca background pelo SHA-256 dos bytes. Imagens com
 *      o mesmo conteudo binario (=mesmo hash) que aparecem em >= 50%
 *      das paginas viram background. Imagens com conteudos diferentes
 *      mesmo se mesmo tamanho NAO viram background.
 *
 * 2. CACHE DE OCR POR HASH.
 *    - Se a mesma imagem (mesmo hash) aparece em multiplas paginas
 *      (ex: screenshot reusado), so OCR-a uma vez. Resultado eh
 *      reusado nas demais. Salva tempo e evita inconsistencia entre
 *      OCRs do mesmo conteudo.
 *
 * Resultado esperado: melhora cobertura na resolucao manuscrita
 * (que estava em 0.119) e mesmas metricas em PDFs onde v1 ja ia bem.
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import { createCanvas, ImageData } from '@napi-rs/canvas';
import {
  createIsomorphicCanvasFactory,
  extractImages,
  extractText,
  getDocumentProxy,
} from 'unpdf';
import { createWorker, type Worker } from 'tesseract.js';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido-unpdf-img-hash';

const LANG = 'por+eng';
const LIMIAR_PIXELS = 100_000;
const FRACAO_PARA_BACKGROUND = 0.5;
const LIMIAR_OCR_MIN_CHARS = 5;

interface ImgMeta {
  width: number;
  height: number;
  pixels: number;
  data: Uint8ClampedArray;
  channels: 1 | 3 | 4;
  hash: string;
}

function calcularHash(data: Uint8ClampedArray): string {
  return createHash('sha256').update(Buffer.from(data.buffer)).digest('hex');
}

async function listarImagens(
  doc: Awaited<ReturnType<typeof getDocumentProxy>>,
  pageNum: number,
): Promise<ImgMeta[]> {
  try {
    const imgs = await extractImages(doc, pageNum);
    return imgs.map((im) => ({
      width: im.width,
      height: im.height,
      pixels: im.width * im.height,
      data: im.data,
      channels: im.channels,
      hash: calcularHash(im.data),
    }));
  } catch {
    return [];
  }
}

/** Detecta hash de imagens que aparecem como "background" — mesmo conteudo
 *  em >= FRACAO_PARA_BACKGROUND das paginas. Retorna Set de hashes. */
function detectarHashesBackground(imgsPorPagina: ImgMeta[][]): Set<string> {
  const total = imgsPorPagina.length;
  if (total === 0) return new Set();
  const counts = new Map<string, number>();
  for (const imgs of imgsPorPagina) {
    const vistos = new Set<string>();
    for (const im of imgs) {
      if (vistos.has(im.hash)) continue; // conta no max 1x por pagina
      vistos.add(im.hash);
      counts.set(im.hash, (counts.get(im.hash) ?? 0) + 1);
    }
  }
  const limiar = total * FRACAO_PARA_BACKGROUND;
  const backgrounds = new Set<string>();
  for (const [hash, count] of counts) {
    if (count >= limiar) backgrounds.add(hash);
  }
  return backgrounds;
}

function rawParaRGBA(
  data: Uint8ClampedArray,
  channels: 1 | 3 | 4,
): Uint8ClampedArray {
  if (channels === 4) return data;
  const pixels = data.length / channels;
  const out = new Uint8ClampedArray(pixels * 4);
  if (channels === 1) {
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = data[i]!;
      out[i * 4 + 1] = data[i]!;
      out[i * 4 + 2] = data[i]!;
      out[i * 4 + 3] = 255;
    }
  } else {
    for (let i = 0; i < pixels; i++) {
      out[i * 4] = data[i * 3]!;
      out[i * 4 + 1] = data[i * 3 + 1]!;
      out[i * 4 + 2] = data[i * 3 + 2]!;
      out[i * 4 + 3] = 255;
    }
  }
  return out;
}

function imagemParaPng(im: ImgMeta): Buffer {
  const canvas = createCanvas(im.width, im.height);
  const ctx = canvas.getContext('2d');
  const rgba = rawParaRGBA(im.data, im.channels);
  const imageData = new ImageData(rgba, im.width, im.height);
  ctx.putImageData(imageData, 0, 0);
  return canvas.toBuffer('image/png');
}

export async function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  const inicio = performance.now();
  let worker: Worker | null = null;
  try {
    const buffer = readFileSync(caminhoPdf);
    const data = new Uint8Array(buffer);

    const canvasImport = () => import('@napi-rs/canvas');
    const CanvasFactory = await createIsomorphicCanvasFactory(canvasImport);
    const doc = await getDocumentProxy(data, { CanvasFactory });
    const numPaginas = doc.numPages;

    const txt = await extractText(doc);
    const textosVetorial: string[] = txt.text.slice();
    while (textosVetorial.length < numPaginas) textosVetorial.push('');

    const imgsPorPagina: ImgMeta[][] = [];
    for (let i = 1; i <= numPaginas; i++) {
      imgsPorPagina.push(await listarImagens(doc, i));
    }
    const hashesBackground = detectarHashesBackground(imgsPorPagina);

    // Identifica imagens-alvo de OCR: nao-background E grandes.
    // Deduplicar por hash dentro do conjunto-alvo (evita OCR-ar a mesma
    // imagem 2x quando aparece em paginas diferentes).
    interface AlvoOcr {
      hash: string;
      imagem: ImgMeta;
      paginas: number[];
    }
    const alvosPorHash = new Map<string, AlvoOcr>();
    for (let i = 0; i < numPaginas; i++) {
      for (const im of imgsPorPagina[i]!) {
        if (im.pixels < LIMIAR_PIXELS) continue;
        if (hashesBackground.has(im.hash)) continue;
        const existe = alvosPorHash.get(im.hash);
        if (existe) {
          existe.paginas.push(i);
        } else {
          alvosPorHash.set(im.hash, { hash: im.hash, imagem: im, paginas: [i] });
        }
      }
    }

    // Roda OCR uma vez por hash, depois aplica resultado em todas as
    // paginas que tem aquela imagem.
    const textos = textosVetorial.slice();
    if (alvosPorHash.size > 0) {
      worker = await createWorker(LANG, 1, { logger: () => {} });
      // cache eh implicito: cada entrada do Map roda OCR uma vez.
      for (const { imagem, paginas } of alvosPorHash.values()) {
        const png = imagemParaPng(imagem);
        const { data: ocrData } = await worker.recognize(png);
        const t = ocrData.text.trim();
        if (t.length < LIMIAR_OCR_MIN_CHARS) continue;
        for (const i of paginas) {
          textos[i] =
            textos[i]!.length > 0 ? `${textos[i]}\n${t}` : t;
        }
      }
    }

    const texto = textos.join('\n\n');
    return {
      extrator: NOME_EXTRATOR,
      texto,
      paginas: textos,
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
