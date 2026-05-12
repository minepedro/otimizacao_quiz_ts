/**
 * BASE configuravel pros extratores hibridos da familia `hibrido-unpdf-img-hash-clean`.
 *
 * Em vez de copiar e colar o codigo do extrator pra cada variante, esse
 * arquivo eh uma BASE que aceita 4 building blocks ligaveis:
 *
 *   1. tessdataBest  — usa tessdata_best (modelos LSTM mais precisos)
 *   2. confidenceMin — filtra palavras OCR com confidence < limiar
 *   3. preProcessar  — binariza + contraste antes do OCR
 *   4. escalasOcr    — roda OCR em multiplas escalas e mescla
 *
 * Cada extrator publico (clean, clean-best, clean-conf, etc) eh um
 * WRAPPER que chama `extrairComOpcoes(caminhoPdf, options)` com opcoes
 * diferentes. Mantem 1 fonte de verdade.
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import { createCanvas, ImageData, loadImage } from '@napi-rs/canvas';
import {
  createIsomorphicCanvasFactory,
  extractImages,
  extractText,
  getDocumentProxy,
} from 'unpdf';
import { createWorker, type Worker } from 'tesseract.js';

import type { ResultadoExtracao } from '../utils/tipos.js';

export interface OpcoesHibrido {
  /** Usar tessdata_best (LSTM melhor) em vez do default. */
  tessdataBest?: boolean;
  /** Filtra palavras com confidence < N (0..100). 0 = desligado. */
  confidenceMin?: number;
  /** Binariza + aumenta contraste antes do OCR. */
  preProcessar?: boolean;
  /** Lista de escalas pra OCR. Default [1]. Pra multi: [1, 2]. */
  escalasOcr?: number[];
}

const LANG = 'por+eng';
const LIMIAR_PIXELS = 100_000;
const FRACAO_PARA_BACKGROUND = 0.5;
const LIMIAR_OCR_MIN_CHARS = 5;
const LIMIAR_SOBREPOSICAO = 0.6;
const URL_TESSDATA_BEST = 'https://tessdata.projectnaptha.com/4.0.0_best';

interface ImgMeta {
  width: number;
  height: number;
  pixels: number;
  data: Uint8ClampedArray;
  channels: 1 | 3 | 4;
  hash: string;
}

// ============================================================
// Helpers comuns (iguais ao hibrido-unpdf-img-hash-clean)
// ============================================================

function calcularHash(data: Uint8ClampedArray): string {
  return createHash('sha256').update(Buffer.from(data.buffer)).digest('hex');
}

function normalizar(s: string): string {
  return s.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase();
}

function tokenizar(s: string): string[] {
  return normalizar(s)
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length >= 2);
}

function quebrarEmFrases(s: string): string[] {
  return s
    .split(/[.!?\n\r]+/)
    .map((f) => f.trim())
    .filter((f) => f.length > 0);
}

function limparOcrSobreposto(textoOcr: string, textoVetorial: string): string {
  const tokensVetorial = new Set(tokenizar(textoVetorial));
  if (tokensVetorial.size === 0) return textoOcr;
  const frases = quebrarEmFrases(textoOcr);
  const mantidas: string[] = [];
  for (const frase of frases) {
    const tokens = tokenizar(frase);
    if (tokens.length === 0) continue;
    let sobrepostas = 0;
    for (const t of tokens) if (tokensVetorial.has(t)) sobrepostas += 1;
    if (sobrepostas / tokens.length < LIMIAR_SOBREPOSICAO) mantidas.push(frase);
  }
  return mantidas.join(' ');
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

function detectarHashesBackground(imgsPorPagina: ImgMeta[][]): Set<string> {
  const total = imgsPorPagina.length;
  if (total === 0) return new Set();
  const counts = new Map<string, number>();
  for (const imgs of imgsPorPagina) {
    const vistos = new Set<string>();
    for (const im of imgs) {
      if (vistos.has(im.hash)) continue;
      vistos.add(im.hash);
      counts.set(im.hash, (counts.get(im.hash) ?? 0) + 1);
    }
  }
  const limiar = total * FRACAO_PARA_BACKGROUND;
  const result = new Set<string>();
  for (const [hash, count] of counts) if (count >= limiar) result.add(hash);
  return result;
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

// ============================================================
// Building block 3: pre-processamento (binarizacao por limiar adaptativo)
// ============================================================

/** Converte PNG pra preto-e-branco usando limiar adaptativo (Otsu simplificado).
 *  Tesseract foi treinado em texto preto sobre fundo branco — binarizar
 *  pode melhorar OCR em imagens coloridas ou com baixo contraste. */
async function preProcessarPng(pngBuffer: Buffer): Promise<Buffer> {
  const img = await loadImage(pngBuffer);
  const w = img.width;
  const h = img.height;
  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;

  // Calcula media de luminance pra usar como limiar
  let somaLum = 0;
  const pixels = w * h;
  for (let i = 0; i < pixels; i++) {
    const r = data[i * 4]!;
    const g = data[i * 4 + 1]!;
    const b = data[i * 4 + 2]!;
    somaLum += 0.299 * r + 0.587 * g + 0.114 * b;
  }
  const limiar = somaLum / pixels;

  // Binariza: pixel >= limiar -> branco, senao preto
  for (let i = 0; i < pixels; i++) {
    const r = data[i * 4]!;
    const g = data[i * 4 + 1]!;
    const b = data[i * 4 + 2]!;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    const v = lum >= limiar ? 255 : 0;
    data[i * 4] = v;
    data[i * 4 + 1] = v;
    data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toBuffer('image/png');
}

// ============================================================
// Building block 4: redimensionar pra outra escala
// ============================================================

async function redimensionarPng(pngBuffer: Buffer, escala: number): Promise<Buffer> {
  if (escala === 1) return pngBuffer;
  const img = await loadImage(pngBuffer);
  const novoW = Math.round(img.width * escala);
  const novoH = Math.round(img.height * escala);
  const canvas = createCanvas(novoW, novoH);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, novoW, novoH);
  return canvas.toBuffer('image/png');
}

// ============================================================
// Building block 2: filtrar OCR por confidence
// ============================================================

interface PalavraOcr {
  text: string;
  confidence: number;
}

function textoComFiltroConfidence(
  ocrData: { text: string; words?: PalavraOcr[] },
  confidenceMin: number,
): string {
  if (confidenceMin <= 0) return ocrData.text.trim();
  const words = ocrData.words ?? [];
  if (words.length === 0) return ocrData.text.trim();
  const mantidas = words
    .filter((w) => w.confidence >= confidenceMin)
    .map((w) => w.text)
    .filter((t) => t.length > 0);
  return mantidas.join(' ').trim();
}

// ============================================================
// Building block 1: criar worker (com ou sem tessdata_best)
// ============================================================

async function criarWorkerConfigurado(opts: OpcoesHibrido): Promise<Worker> {
  // PEGADINHA tesseract.js: ele cachea o `por.traineddata` na pasta atual
  // (cwd) e nas chamadas seguintes IGNORA langPath se o cache existir.
  // Pra forcar usar tessdata_best, precisa de cachePath SEPARADO.
  const workerOpts: { logger: () => void; langPath?: string; cachePath?: string } = {
    logger: () => {},
  };
  if (opts.tessdataBest) {
    workerOpts.langPath = URL_TESSDATA_BEST;
    workerOpts.cachePath = '.tesseract-cache-best';
  }
  return await createWorker(LANG, 1, workerOpts as never);
}

// ============================================================
// OCR de uma imagem com TODAS as opcoes aplicadas
// ============================================================

async function ocrImagemComOpcoes(
  worker: Worker,
  im: ImgMeta,
  opts: OpcoesHibrido,
): Promise<string> {
  let png = imagemParaPng(im);
  if (opts.preProcessar) {
    png = await preProcessarPng(png);
  }

  const escalas = opts.escalasOcr && opts.escalasOcr.length > 0 ? opts.escalasOcr : [1];
  const confMin = opts.confidenceMin ?? 0;

  if (escalas.length === 1) {
    const escala = escalas[0]!;
    const pngFinal = escala === 1 ? png : await redimensionarPng(png, escala);
    const { data } = await worker.recognize(pngFinal);
    return textoComFiltroConfidence(data as { text: string; words?: PalavraOcr[] }, confMin);
  }

  // Multiplas escalas — roda OCR em cada e mescla por uniao de tokens.
  const textosPorEscala: string[] = [];
  for (const escala of escalas) {
    const pngFinal = escala === 1 ? png : await redimensionarPng(png, escala);
    const { data } = await worker.recognize(pngFinal);
    textosPorEscala.push(
      textoComFiltroConfidence(data as { text: string; words?: PalavraOcr[] }, confMin),
    );
  }
  return mesclarMultiEscalas(textosPorEscala);
}

/** Mescla textos OCR de varias escalas. Estrategia: pega o texto MAIS
 *  LONGO como base (provavelmente captou mais), depois adiciona palavras
 *  exclusivas dos demais (que a escala maior pegou e a menor nao, ou
 *  vice-versa). */
function mesclarMultiEscalas(textos: string[]): string {
  if (textos.length === 0) return '';
  if (textos.length === 1) return textos[0]!;
  const ordenados = [...textos].sort((a, b) => b.length - a.length);
  const base = ordenados[0]!;
  const tokensBase = new Set(tokenizar(base));
  const extras: string[] = [];
  for (let i = 1; i < ordenados.length; i++) {
    for (const palavra of ordenados[i]!.split(/\s+/)) {
      const t = normalizar(palavra);
      if (t.length >= 2 && !tokensBase.has(t)) {
        extras.push(palavra);
        tokensBase.add(t);
      }
    }
  }
  return extras.length > 0 ? `${base} ${extras.join(' ')}` : base;
}

// ============================================================
// FUNCAO PRINCIPAL CONFIGURAVEL
// ============================================================

export async function extrairComOpcoes(
  caminhoPdf: string,
  opts: OpcoesHibrido,
  nomeExtrator: string,
): Promise<ResultadoExtracao> {
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
        if (existe) existe.paginas.push(i);
        else alvosPorHash.set(im.hash, { hash: im.hash, imagem: im, paginas: [i] });
      }
    }

    const textos = textosVetorial.slice();
    if (alvosPorHash.size > 0) {
      worker = await criarWorkerConfigurado(opts);
      for (const { imagem, paginas } of alvosPorHash.values()) {
        const textoOcrBruto = await ocrImagemComOpcoes(worker, imagem, opts);
        if (textoOcrBruto.length < LIMIAR_OCR_MIN_CHARS) continue;
        for (const i of paginas) {
          const limpo = limparOcrSobreposto(textoOcrBruto, textosVetorial[i] ?? '');
          if (limpo.length < LIMIAR_OCR_MIN_CHARS) continue;
          textos[i] = textos[i]!.length > 0 ? `${textos[i]}\n${limpo}` : limpo;
        }
      }
    }

    const texto = textos.join('\n\n');
    return {
      extrator: nomeExtrator,
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
      extrator: nomeExtrator,
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
