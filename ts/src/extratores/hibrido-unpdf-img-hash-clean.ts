/**
 * Extrator hibrido v3.2 (img-hash-clean): refinamento do hibrido-unpdf-img-hash.
 *
 * Adiciona DEDUPLICACAO entre o texto OCR e o texto vetorial da pagina.
 *
 * Motivacao do problema:
 *   No EPM 309 STP, o img-hash gerou tam=1.66 (66% de texto extra vs
 *   gabarito). Investigando: o PDF tem imagens de FUNDO com texto
 *   vetorial por cima. Em teoria nao deveria duplicar (vetorial pega o
 *   texto vetor, OCR pega o que esta DENTRO dos pixels da imagem). Mas
 *   na pratica algumas imagens TEM texto embutido que tambem esta
 *   presente como vetorial (PowerPoint exportado costuma fazer isso),
 *   ou o OCR le ruido visual da imagem que sobrepoe palavras do
 *   vetorial.
 *
 * Solucao: pra cada texto OCR, divide em frases. Cada frase eh aceita
 * APENAS se < LIMIAR_SOBREPOSICAO das suas palavras ja existem no texto
 * vetorial da mesma pagina. Frases com sobreposicao alta (>=60%
 * provavelmente sao duplicatas) sao descartadas.
 *
 * Resultado esperado: menos inflado em PDFs onde imagens sao decorativas
 * mas tem texto embutido similar ao vetorial. Mantem ganho em PDFs onde
 * imagens tem informacao GENUINAMENTE nova (screenshots).
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

export const NOME_EXTRATOR = 'hibrido-unpdf-img-hash-clean';

const LANG = 'por+eng';
const LIMIAR_PIXELS = 100_000;
const FRACAO_PARA_BACKGROUND = 0.5;
const LIMIAR_OCR_MIN_CHARS = 5;
const LIMIAR_SOBREPOSICAO = 0.6; // >= 60% das palavras da frase OCR ja estao no vetorial = descarta

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

function normalizar(s: string): string {
  return s
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .toLowerCase();
}

function tokenizar(s: string): string[] {
  return normalizar(s)
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length >= 2);
}

function quebrarEmFrases(s: string): string[] {
  // Separa por pontuacao final ou quebras de linha. Mantem espacos
  // internos. Filtra strings vazias.
  return s
    .split(/[.!?\n\r]+/)
    .map((f) => f.trim())
    .filter((f) => f.length > 0);
}

/** Remove do texto OCR as frases que tem >= LIMIAR_SOBREPOSICAO de palavras
 *  ja presentes no texto vetorial. Retorna texto OCR "limpo". */
function limparOcrSobreposto(textoOcr: string, textoVetorial: string): string {
  const tokensVetorial = new Set(tokenizar(textoVetorial));
  if (tokensVetorial.size === 0) return textoOcr;

  const frases = quebrarEmFrases(textoOcr);
  const frasesMantidas: string[] = [];

  for (const frase of frases) {
    const tokens = tokenizar(frase);
    if (tokens.length === 0) continue;
    let sobrepostas = 0;
    for (const t of tokens) {
      if (tokensVetorial.has(t)) sobrepostas += 1;
    }
    const ratio = sobrepostas / tokens.length;
    if (ratio < LIMIAR_SOBREPOSICAO) {
      frasesMantidas.push(frase);
    }
  }

  return frasesMantidas.join(' ');
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

    const textos = textosVetorial.slice();
    if (alvosPorHash.size > 0) {
      worker = await createWorker(LANG, 1, { logger: () => {} });
      // OCR uma vez por hash, depois aplica em cada pagina LIMPANDO contra
      // o texto vetorial daquela pagina especifica.
      for (const { imagem, paginas } of alvosPorHash.values()) {
        const png = imagemParaPng(imagem);
        const { data: ocrData } = await worker.recognize(png);
        const textoOcrBruto = ocrData.text.trim();
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
