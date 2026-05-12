/**
 * Extrator pdfjs-dist: PDF.js da Mozilla, usado direto.
 *
 * Diferente do pdf-parse, aqui controlamos a leitura. Para cada pagina,
 * pegamos os items com posicao (transform[4]=x, transform[5]=y) e
 * ordenamos por y (linha) depois x (coluna), pra preservar ordem visual
 * mesmo em layout multi-coluna (achado robusto do lab Python).
 */

import { readFileSync } from 'node:fs';

import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';

import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'pdfjs-dist';

interface ItemTexto {
  str: string;
  transform: number[];
  hasEOL?: boolean;
}

function juntarItens(items: ItemTexto[]): string {
  // Ordena por y (decrescente — em PDF o y cresce de baixo pra cima),
  // depois por x (crescente). Tolerancia de 2pt agrupa items na mesma linha.
  const TOLERANCIA = 2;
  const ordenado = [...items].sort((a, b) => {
    const dy = b.transform[5]! - a.transform[5]!;
    if (Math.abs(dy) > TOLERANCIA) return dy;
    return a.transform[4]! - b.transform[4]!;
  });

  let texto = '';
  let lastY: number | null = null;
  for (const it of ordenado) {
    const y = it.transform[5]!;
    if (lastY !== null && Math.abs(y - lastY) > TOLERANCIA) {
      texto += '\n';
    } else if (texto && !texto.endsWith(' ') && !texto.endsWith('\n')) {
      texto += ' ';
    }
    texto += it.str;
    lastY = y;
  }
  return texto;
}

export async function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  const inicio = performance.now();
  try {
    const buffer = readFileSync(caminhoPdf);
    const loadingTask = getDocument({
      data: new Uint8Array(buffer),
      useSystemFonts: false,
      disableFontFace: true,
      verbosity: 0,
    });
    const doc = await loadingTask.promise;
    const paginas: string[] = [];
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      const content = await page.getTextContent();
      paginas.push(juntarItens(content.items as ItemTexto[]));
      page.cleanup();
    }
    await doc.cleanup();
    await doc.destroy();
    const texto = paginas.join('\n\n');
    return {
      extrator: NOME_EXTRATOR,
      texto,
      paginas,
      numPaginas: doc.numPages,
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
  }
}
