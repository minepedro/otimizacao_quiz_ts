/**
 * Extrator hibrido v3 (bbox): OCR roda na REGIAO da pagina onde estao
 * as imagens grandes (nao na pagina inteira).
 *
 * Diferencas vs hibrido-unpdf e hibrido-unpdf-img:
 *   - hibrido-unpdf: renderiza pagina inteira, OCR na pagina inteira,
 *     SUBSTITUI texto vetorial. Pode perder texto vetorial perfeito.
 *   - hibrido-unpdf-img: OCR direto nos bytes da imagem (sem renderizar).
 *     SOMA com texto vetorial. Nao preserva contexto visual da imagem
 *     na pagina (como ela aparece em escala/rotacao real).
 *   - hibrido-unpdf-bbox (este): renderiza pagina inteira UMA vez,
 *     descobre onde cada imagem esta via pdfjs.getOperatorList()
 *     (rastreia CTM = current transformation matrix), passa rectangle
 *     pro tesseract pra OCR-ar so a regiao da imagem na pagina
 *     renderizada. SOMA com texto vetorial.
 *
 * Vantagens teoricas:
 *   - Preserva contexto visual (escala/rotacao aplicadas)
 *   - Nao perde texto vetorial (soma)
 *   - OCR mais focado (so na imagem, nao na pagina inteira)
 *
 * Custos:
 *   - Renderiza pagina inteira mesmo usando so parte
 *   - Codigo de rastreamento de CTM eh nao-trivial
 */

import { readFileSync } from 'node:fs';

import {
  createIsomorphicCanvasFactory,
  extractImages,
  extractText,
  getDocumentProxy,
  renderPageAsImage,
} from 'unpdf';
import { OPS } from 'pdfjs-dist/legacy/build/pdf.mjs';
import { createWorker, type Worker } from 'tesseract.js';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'hibrido-unpdf-bbox';

const LANG = 'por+eng';
const ESCALA_RENDER = 2;
const LIMIAR_PIXELS = 100_000;
const FRACAO_PARA_BACKGROUND = 0.5;
const LIMIAR_OCR_MIN_CHARS = 5;
const MARGEM_RECT_PX = 4; // pequena margem ao redor da imagem pra OCR pegar texto na borda

type Matriz = [number, number, number, number, number, number]; // [a, b, c, d, e, f]

interface BboxPDF {
  x: number;
  y: number;
  w: number;
  h: number;
  width: number; // dimensao original da imagem em pixels
  height: number;
}

function multiplicar(m1: Matriz, m2: Matriz): Matriz {
  // Matriz 3x3 representada como [a, b, c, d, e, f]:
  // | a c e |
  // | b d f |
  // | 0 0 1 |
  return [
    m1[0] * m2[0] + m1[2] * m2[1],
    m1[1] * m2[0] + m1[3] * m2[1],
    m1[0] * m2[2] + m1[2] * m2[3],
    m1[1] * m2[2] + m1[3] * m2[3],
    m1[0] * m2[4] + m1[2] * m2[5] + m1[4],
    m1[1] * m2[4] + m1[3] * m2[5] + m1[5],
  ];
}

function aplicar(m: Matriz, x: number, y: number): [number, number] {
  return [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
}

function bboxDoCTM(ctm: Matriz): { x: number; y: number; w: number; h: number } {
  // PDF.js convencao: imagens sao renderizadas em unit square [0,1]x[0,1].
  // Aplica CTM nos 4 cantos e pega bounding box.
  const cantos: [number, number][] = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ];
  const xs: number[] = [];
  const ys: number[] = [];
  for (const [cx, cy] of cantos) {
    const [px, py] = aplicar(ctm, cx, cy);
    xs.push(px);
    ys.push(py);
  }
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

interface PdfjsObjs {
  has?: (name: string) => boolean;
  get: (name: string) => unknown;
}

async function bboxesDeImagensNaPagina(
  page: { getOperatorList: () => Promise<{ fnArray: number[]; argsArray: unknown[][] }>;
          objs: PdfjsObjs;
          commonObjs: PdfjsObjs },
): Promise<BboxPDF[]> {
  const ops = await page.getOperatorList();
  const fns = ops.fnArray;
  const args = ops.argsArray;

  const stack: Matriz[] = [];
  let ctm: Matriz = [1, 0, 0, 1, 0, 0];
  const resultado: BboxPDF[] = [];

  const tentarPegarImg = (name: string): { width: number; height: number } | null => {
    try {
      const obj = page.objs.has?.(name) ? page.objs.get(name) : null;
      if (obj && typeof obj === 'object' && 'width' in obj && 'height' in obj) {
        return { width: Number((obj as { width: number }).width), height: Number((obj as { height: number }).height) };
      }
    } catch {
      // ignora
    }
    try {
      const obj = page.commonObjs.has?.(name) ? page.commonObjs.get(name) : null;
      if (obj && typeof obj === 'object' && 'width' in obj && 'height' in obj) {
        return { width: Number((obj as { width: number }).width), height: Number((obj as { height: number }).height) };
      }
    } catch {
      // ignora
    }
    return null;
  };

  for (let i = 0; i < fns.length; i++) {
    const op = fns[i];
    const a = args[i];
    if (op === OPS.save) {
      stack.push([...ctm] as Matriz);
    } else if (op === OPS.restore) {
      ctm = stack.pop() ?? ([1, 0, 0, 1, 0, 0] as Matriz);
    } else if (op === OPS.transform && a && a.length >= 6) {
      const t = a as unknown as Matriz;
      ctm = multiplicar(ctm, t);
    } else if (op === OPS.paintImageXObject && a && a.length >= 1) {
      const name = a[0] as string;
      const img = tentarPegarImg(name);
      const bbox = bboxDoCTM(ctm);
      resultado.push({
        ...bbox,
        width: img?.width ?? 0,
        height: img?.height ?? 0,
      });
    } else if (op === OPS.paintInlineImageXObject && a && a.length >= 1) {
      const inline = a[0] as { width?: number; height?: number };
      const bbox = bboxDoCTM(ctm);
      resultado.push({
        ...bbox,
        width: inline.width ?? 0,
        height: inline.height ?? 0,
      });
    }
  }
  return resultado;
}

async function listarImagensSemConteudo(
  doc: Awaited<ReturnType<typeof getDocumentProxy>>,
  pageNum: number,
): Promise<{ width: number; height: number; pixels: number }[]> {
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

function detectarBackground(
  imgsPorPagina: { width: number; height: number }[][],
): { width: number; height: number } | null {
  const total = imgsPorPagina.length;
  if (total === 0) return null;
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
  if (melhor.count / total < FRACAO_PARA_BACKGROUND) return null;
  return { width: melhor.w, height: melhor.h };
}

interface RectanglePx {
  left: number;
  top: number;
  width: number;
  height: number;
}

function bboxParaRectanglePagina(
  bbox: BboxPDF,
  pageBox: [number, number, number, number],
): RectanglePx {
  // page.view = [x0, y0, x1, y1] em pontos (PDF user space)
  // Origem PDF: canto INFERIOR esquerdo, Y cresce pra cima
  // Origem canvas: canto SUPERIOR esquerdo, Y cresce pra baixo
  const pageHeight = pageBox[3] - pageBox[1];
  const xPdf = bbox.x - pageBox[0];
  const yPdfTopo = pageHeight - (bbox.y - pageBox[1]) - bbox.h;
  const left = Math.max(0, Math.round(xPdf * ESCALA_RENDER) - MARGEM_RECT_PX);
  const top = Math.max(0, Math.round(yPdfTopo * ESCALA_RENDER) - MARGEM_RECT_PX);
  const width = Math.round(bbox.w * ESCALA_RENDER) + 2 * MARGEM_RECT_PX;
  const height = Math.round(bbox.h * ESCALA_RENDER) + 2 * MARGEM_RECT_PX;
  return { left, top, width, height };
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

    // Detecta background pelo extractImages (so dimensoes).
    const dimsPorPagina: { width: number; height: number }[][] = [];
    for (let i = 1; i <= numPaginas; i++) {
      dimsPorPagina.push(await listarImagensSemConteudo(doc, i));
    }
    const background = detectarBackground(dimsPorPagina);

    // Pra cada pagina: descobre bboxes via operatorList. Filtra
    // imagens grandes nao-background.
    interface PaginaTrabalho {
      pagina: number;
      bboxes: BboxPDF[];
      pageBox: [number, number, number, number];
    }
    const trabalhos: PaginaTrabalho[] = [];
    for (let i = 1; i <= numPaginas; i++) {
      const page = await doc.getPage(i);
      const pageBox = page.view as [number, number, number, number];
      const bboxes = await bboxesDeImagensNaPagina(page);
      const grandes = bboxes.filter((b) => {
        const px = b.width * b.height;
        if (px < LIMIAR_PIXELS) return false;
        if (background && b.width === background.width && b.height === background.height) return false;
        return true;
      });
      if (grandes.length > 0) {
        trabalhos.push({ pagina: i, bboxes: grandes, pageBox });
      }
    }

    // Roda OCR. Pra cada pagina-com-trabalho: renderiza, OCR-a cada bbox.
    const textos = textosVetorial.slice();
    if (trabalhos.length > 0) {
      worker = await createWorker(LANG, 1, { logger: () => {} });
      for (const { pagina, bboxes, pageBox } of trabalhos) {
        const png = await renderPageAsImage(doc, pagina, {
          scale: ESCALA_RENDER,
          canvasImport,
        });
        const buf = Buffer.from(png);
        const partes: string[] = [];
        for (const bbox of bboxes) {
          const rect = bboxParaRectanglePagina(bbox, pageBox);
          if (rect.width <= 0 || rect.height <= 0) continue;
          try {
            const { data: ocrData } = await worker.recognize(buf, { rectangle: rect });
            const t = ocrData.text.trim();
            if (t.length >= LIMIAR_OCR_MIN_CHARS) partes.push(t);
          } catch {
            // se rectangle der erro, ignora aquela bbox
          }
        }
        if (partes.length > 0) {
          const idx = pagina - 1;
          textos[idx] =
            textos[idx]!.length > 0
              ? `${textos[idx]}\n${partes.join('\n')}`
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
