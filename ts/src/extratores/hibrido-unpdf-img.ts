/**
 * Extrator hibrido v3 (img-direto): OCR roda APENAS nos bytes das imagens
 * extras (nao-background), sem renderizar a pagina inteira.
 *
 * Diferenca vs hibrido-unpdf:
 *   - hibrido-unpdf renderiza a pagina inteira como PNG e roda OCR.
 *     Pode duplicar texto vetorial e adicionar ruido (header/rodape).
 *   - hibrido-unpdf-img extrai só as imagens via extractImages do unpdf
 *     (devolve buffer RGB raw), converte pra PNG via @napi-rs/canvas,
 *     joga no tesseract. Texto vetorial (perfeito) eh PRESERVADO e
 *     somado ao texto do OCR.
 *
 * Estrategia:
 *   1. extractText pra todas as paginas (1 chamada).
 *   2. extractImages por pagina + detecta background.
 *   3. Pra cada imagem nao-background com >= LIMIAR_PIXELS:
 *      converte raw RGB -> PNG -> OCR. Concatena resultado.
 *   4. Texto final por pagina = texto_vetorial + textos_ocr_das_imagens.
 *
 * Nao OCR-a a pagina inteira nunca; nao tem problema de duplicar
 * texto vetorial; preserva qualidade do texto que ja era perfeito.
 */

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

export const NOME_EXTRATOR = 'hibrido-unpdf-img';

const LANG = 'por+eng';
const LIMIAR_PIXELS = 100_000;
const FRACAO_PARA_BACKGROUND = 0.5;
const LIMIAR_OCR_MIN_CHARS = 5; // ignora resultados de OCR muito curtos (ruido)

interface ImgMeta {
  width: number;
  height: number;
  pixels: number;
  data: Uint8ClampedArray;
  channels: 1 | 3 | 4;
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
    }));
  } catch {
    return [];
  }
}

function detectarTamanhoBackground(
  imgsPorPagina: ImgMeta[][],
): { width: number; height: number } | null {
  const totalPaginas = imgsPorPagina.length;
  if (totalPaginas === 0) return null;
  const counts = new Map<string, { count: number; w: number; h: number }>();
  for (const imgs of imgsPorPagina) {
    const vistos = new Set<string>();
    for (const im of imgs) {
      const key = `${im.width}x${im.height}`;
      if (vistos.has(key)) continue;
      vistos.add(key);
      const atual = counts.get(key) ?? { count: 0, w: im.width, h: im.height };
      atual.count += 1;
      counts.set(key, atual);
    }
  }
  let melhor: { count: number; w: number; h: number } | null = null;
  for (const v of counts.values()) {
    if (!melhor || v.count > melhor.count) melhor = v;
  }
  if (!melhor) return null;
  if (melhor.count / totalPaginas < FRACAO_PARA_BACKGROUND) return null;
  return { width: melhor.w, height: melhor.h };
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
    // channels === 3
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

async function ocrDaImagem(worker: Worker, im: ImgMeta): Promise<string> {
  const png = imagemParaPng(im);
  const { data } = await worker.recognize(png);
  return data.text.trim();
}

function ehNaoBackground(
  im: ImgMeta,
  bg: { width: number; height: number } | null,
): boolean {
  return !bg || im.width !== bg.width || im.height !== bg.height;
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
    const background = detectarTamanhoBackground(imgsPorPagina);

    // Identifica imagens-alvo de OCR: nao-background E grandes.
    const ocrPlanejado: { pagina: number; imagens: ImgMeta[] }[] = [];
    let totalImagensOcr = 0;
    for (let i = 0; i < numPaginas; i++) {
      const alvos = imgsPorPagina[i]!.filter(
        (im) => im.pixels >= LIMIAR_PIXELS && ehNaoBackground(im, background),
      );
      if (alvos.length > 0) {
        ocrPlanejado.push({ pagina: i, imagens: alvos });
        totalImagensOcr += alvos.length;
      }
    }

    // Roda OCR nas imagens-alvo (so cria worker se precisa).
    const textos = textosVetorial.slice();
    if (totalImagensOcr > 0) {
      worker = await createWorker(LANG, 1, { logger: () => {} });
      for (const { pagina, imagens } of ocrPlanejado) {
        const partes: string[] = [];
        for (const im of imagens) {
          const t = await ocrDaImagem(worker, im);
          if (t.length >= LIMIAR_OCR_MIN_CHARS) partes.push(t);
        }
        if (partes.length > 0) {
          textos[pagina] =
            textos[pagina]!.length > 0
              ? `${textos[pagina]}\n${partes.join('\n')}`
              : partes.join('\n');
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
