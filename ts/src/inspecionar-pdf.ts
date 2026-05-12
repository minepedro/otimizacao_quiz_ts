/**
 * Raio-x de PDF: gera ficha tecnica de cada PDF pra ajudar a entender
 * por que extratores se saem bem ou mal nele.
 *
 * Uso mupdf (lib mais completa). Pra cada PDF, coleta:
 *   - tamanho do arquivo, num paginas, dimensoes medias
 *   - caracteres extraidos (total e por pagina: media, min, max)
 *   - paginas "pobres em texto" (< limiar) — proxy de "pagina de imagem"
 *   - quantidade de imagens (se mupdf reportar via structured text)
 *   - se tem gabarito, razao chars_extraidos/chars_gabarito
 *     -> estimativa do quanto de conteudo esta em imagem (1 - razao)
 *
 * Saida: resultados/raio-x/<nome>.json e log no console.
 *
 * Uso:
 *   npm run inspecionar
 *   npm run inspecionar -- --apenas "Aula 04"
 */

import { readdirSync, readFileSync, writeFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { join, parse } from 'node:path';

import * as mupdf from 'mupdf';

import { PROJETO_RAIZ } from './utils/env.js';

const PDFS_DIR = join(PROJETO_RAIZ, 'dados', 'pdfs');
const GABARITOS_DIR = join(PROJETO_RAIZ, 'dados', 'gabaritos');
const RAIO_X_DIR = join(PROJETO_RAIZ, 'resultados', 'raio-x');

const LIMIAR_POUCO_TEXTO = 100; // chars: abaixo disso, pagina eh "pobre em texto"

interface RaioX {
  arquivo: string;
  tamanhoBytes: number;
  tamanhoMB: number;
  numPaginas: number;
  dimensoesMedias: { largura: number; altura: number };
  caracteresExtraidos: number;
  caracteresPorPagina: { media: number; min: number; max: number };
  paginasComPoucoTexto: number[]; // 1-indexed
  numImagensReportadas: number;
  comGabarito: {
    caracteresGabarito: number;
    razaoExtraidoSobreGabarito: number;
    estimativaConteudoEmImagem: number; // 1 - razao, capado em [0, 1]
  } | null;
}

interface BlocoStructuredText {
  type?: string;
  lines?: unknown[];
}

function parseArgs(): { filtro: string | null } {
  const args = process.argv.slice(2);
  let filtro: string | null = null;
  const idx = args.indexOf('--apenas');
  if (idx !== -1 && args[idx + 1]) filtro = args[idx + 1]!;
  return { filtro };
}

function gerarRaioX(caminhoPdf: string, nome: string): RaioX {
  const buffer = readFileSync(caminhoPdf);
  const stats = statSync(caminhoPdf);
  const doc = mupdf.Document.openDocument(
    new Uint8Array(buffer),
    'application/pdf',
  );
  const numPaginas = doc.countPages();

  const charsPorPag: number[] = [];
  let larguraTotal = 0;
  let alturaTotal = 0;
  let numImagens = 0;
  let textoTotal = '';

  for (let i = 0; i < numPaginas; i++) {
    const page = doc.loadPage(i);
    const bounds = page.getBounds();
    larguraTotal += bounds[2] - bounds[0];
    alturaTotal += bounds[3] - bounds[1];

    // Texto puro
    const stext = page.toStructuredText('preserve-whitespace');
    const t = stext.asText();
    charsPorPag.push(t.length);
    textoTotal += t + '\n\n';
    stext.destroy();

    // Imagens: usa preserve-images e parseia JSON pra contar blocos type=image
    try {
      const stextImg = page.toStructuredText('preserve-images');
      const json = JSON.parse(stextImg.asJSON()) as { blocks?: BlocoStructuredText[] };
      if (json.blocks) {
        for (const b of json.blocks) {
          if (b.type === 'image') numImagens += 1;
        }
      }
      stextImg.destroy();
    } catch {
      // ignora se asJSON nao funcionar nessa pagina
    }

    page.destroy();
  }
  doc.destroy();

  const media = (arr: number[]): number =>
    arr.length === 0 ? 0 : Math.round(arr.reduce((s, x) => s + x, 0) / arr.length);

  const paginasPouco: number[] = [];
  charsPorPag.forEach((n, idx) => {
    if (n < LIMIAR_POUCO_TEXTO) paginasPouco.push(idx + 1);
  });

  let comGabarito: RaioX['comGabarito'] = null;
  const caminhoGabarito = join(GABARITOS_DIR, `${nome}.txt`);
  if (existsSync(caminhoGabarito)) {
    const gab = readFileSync(caminhoGabarito, 'utf-8');
    const charsGab = gab.length;
    const razao = charsGab === 0 ? 0 : textoTotal.length / charsGab;
    const estimativa = Math.max(0, Math.min(1, 1 - razao));
    comGabarito = {
      caracteresGabarito: charsGab,
      razaoExtraidoSobreGabarito: Number(razao.toFixed(4)),
      estimativaConteudoEmImagem: Number(estimativa.toFixed(4)),
    };
  }

  return {
    arquivo: `${nome}.pdf`,
    tamanhoBytes: stats.size,
    tamanhoMB: Number((stats.size / 1024 / 1024).toFixed(2)),
    numPaginas,
    dimensoesMedias: {
      largura: Math.round(larguraTotal / numPaginas),
      altura: Math.round(alturaTotal / numPaginas),
    },
    caracteresExtraidos: textoTotal.length,
    caracteresPorPagina: {
      media: media(charsPorPag),
      min: charsPorPag.length === 0 ? 0 : Math.min(...charsPorPag),
      max: charsPorPag.length === 0 ? 0 : Math.max(...charsPorPag),
    },
    paginasComPoucoTexto: paginasPouco,
    numImagensReportadas: numImagens,
    comGabarito,
  };
}

function main(): number {
  const { filtro } = parseArgs();

  if (!existsSync(RAIO_X_DIR)) mkdirSync(RAIO_X_DIR, { recursive: true });

  let pdfs = readdirSync(PDFS_DIR)
    .filter((f) => f.toLowerCase().endsWith('.pdf'))
    .sort();
  if (filtro) {
    pdfs = pdfs.filter((f) => f.toLowerCase().includes(filtro.toLowerCase()));
  }

  if (pdfs.length === 0) {
    console.log(`Nenhum PDF em ${PDFS_DIR} (filtro=${JSON.stringify(filtro)})`);
    return 1;
  }

  console.log(`Raio-x de ${pdfs.length} PDFs.\n`);

  for (const pdf of pdfs) {
    const nome = parse(pdf).name;
    const caminhoPdf = join(PDFS_DIR, pdf);
    const r = gerarRaioX(caminhoPdf, nome);

    const destino = join(RAIO_X_DIR, `${nome}.json`);
    writeFileSync(destino, JSON.stringify(r, null, 2), 'utf-8');

    console.log(`[${nome}]`);
    console.log(`  ${r.numPaginas} pg, ${r.tamanhoMB} MB, ${r.caracteresExtraidos} chars extraidos`);
    console.log(
      `  chars/pg: media=${r.caracteresPorPagina.media} min=${r.caracteresPorPagina.min} max=${r.caracteresPorPagina.max}`,
    );
    console.log(
      `  paginas com < ${LIMIAR_POUCO_TEXTO} chars: ${r.paginasComPoucoTexto.length}/${r.numPaginas}` +
        (r.paginasComPoucoTexto.length > 0
          ? ` (${r.paginasComPoucoTexto.slice(0, 10).join(', ')}${r.paginasComPoucoTexto.length > 10 ? '...' : ''})`
          : ''),
    );
    console.log(`  imagens reportadas: ${r.numImagensReportadas}`);
    if (r.comGabarito) {
      console.log(
        `  vs gabarito: extraido/gabarito=${r.comGabarito.razaoExtraidoSobreGabarito} ` +
          `-> ~${(r.comGabarito.estimativaConteudoEmImagem * 100).toFixed(1)}% do conteudo em imagem`,
      );
    } else {
      console.log(`  vs gabarito: sem gabarito`);
    }
    console.log(`  -> ${destino}`);
    console.log('');
  }

  return 0;
}

process.exit(main());
