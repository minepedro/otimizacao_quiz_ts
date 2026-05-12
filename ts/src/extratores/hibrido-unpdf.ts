/**
 * Extrator hibrido MIT-only: combinacao de unpdf (texto + render +
 * extractImages) + tesseract.js (OCR), sem dependencia AGPL do mupdf.
 *
 * Heuristica pra decidir quais paginas precisam de OCR (sem ter bbox
 * de imagem como o mupdf):
 *
 *   1. PASSO 0 — detecta o "tamanho do background" do PDF: a dimensao
 *      de imagem (em pixels) que aparece na maioria das paginas
 *      (>= 50%). Em slides PowerPoint, isso eh tipicamente o template
 *      visual repetido.
 *
 *   2. PASSO 1 — pra cada pagina, extrai texto via unpdf e conta chars.
 *
 *   3. PASSO 2 — pra cada pagina, lista as imagens (extractImages) e
 *      separa em "background" (mesmo tamanho do detectado no passo 0)
 *      e "extras" (qualquer outro tamanho).
 *
 *   4. PASSO 3 — roda OCR se:
 *        (a) chars < LIMIAR_CHARS_VAZIA E existe pelo menos 1 imagem
 *            (background ou extra) com >= LIMIAR_PIXELS pixels
 *            -> pagina visualmente cheia mas com pouco texto vetorial
 *
 *      OU
 *        (b) tem imagem extra (NAO-background) com >= LIMIAR_PIXELS
 *            -> screenshot misturado com texto, mesmo se tem chars
 *
 *      O criterio (a) cobre slides "screenshot full-page" e
 *      capas/separadores com elemento visual. O (b) cobre paginas
 *      mistas (texto + screenshot). Combinados evitam OCR em paginas
 *      genuinamente vazias (capas sem nada).
 */

import { readFileSync } from 'node:fs';

import {
  createIsomorphicCanvasFactory,
  extractImages,
  extractText,
  getDocumentProxy,
  renderPageAsImage,
} from 'unpdf';
import { createWorker, type Worker } from 'tesseract.js';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido-unpdf';

const LANG = 'por+eng';
const ESCALA_RENDER = 2;
const LIMIAR_CHARS_VAZIA = 200;
const LIMIAR_PIXELS = 100_000; // ~316x316 px = imagem grande
const FRACAO_PARA_BACKGROUND = 0.5; // imagem precisa aparecer em >= 50% das pgs

interface ImgMeta {
  width: number;
  height: number;
  pixels: number;
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
    // conta cada tamanho NO MAXIMO uma vez por pagina (evita PDF com
    // varias copias da mesma imagem na mesma pagina)
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

function deveRodarOcr(
  chars: number,
  imgs: ImgMeta[],
  background: { width: number; height: number } | null,
): boolean {
  const naoBackground = imgs.filter(
    (im) =>
      !background ||
      im.width !== background.width ||
      im.height !== background.height,
  );
  const temImagemExtraGrande = naoBackground.some((im) => im.pixels >= LIMIAR_PIXELS);
  // Caso (b): screenshot misturado com texto
  if (temImagemExtraGrande) return true;
  // Caso (a): pagina pobre de texto MAS visualmente cheia (tem imagem grande)
  const temAlgumaImagemGrande = imgs.some((im) => im.pixels >= LIMIAR_PIXELS);
  if (chars < LIMIAR_CHARS_VAZIA && temAlgumaImagemGrande) return true;
  return false;
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

    // Texto vetorial por pagina (uma chamada so).
    const txt = await extractText(doc);
    const textos: string[] = txt.text.slice();
    while (textos.length < numPaginas) textos.push('');

    // Imagens por pagina + deteccao de background.
    const imgsPorPagina: ImgMeta[][] = [];
    for (let i = 1; i <= numPaginas; i++) {
      imgsPorPagina.push(await listarImagens(doc, i));
    }
    const background = detectarTamanhoBackground(imgsPorPagina);

    // Decide quais paginas precisam de OCR.
    const precisamOcr: number[] = [];
    for (let i = 0; i < numPaginas; i++) {
      if (deveRodarOcr(textos[i]!.length, imgsPorPagina[i]!, background)) {
        precisamOcr.push(i);
      }
    }

    // Roda OCR e SUBSTITUI o texto da pagina (mesma estrategia do hibrido2).
    if (precisamOcr.length > 0) {
      worker = await createWorker(LANG, 1, { logger: () => {} });
      for (const i of precisamOcr) {
        const png = await renderPageAsImage(doc, i + 1, {
          scale: ESCALA_RENDER,
          canvasImport,
        });
        const buf = Buffer.from(png);
        const { data: ocrData } = await worker.recognize(buf);
        textos[i] = ocrData.text;
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
