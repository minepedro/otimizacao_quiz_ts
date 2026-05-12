/**
 * Benchmark dos extratores locais contra o gabarito (Claude API).
 *
 * Pra cada PDF em dados/pdfs/ que tenha um gabarito em dados/gabaritos/,
 * roda todos os extratores listados, calcula 4 metricas de similaridade,
 * registra tempo. Escreve resultados em:
 *
 *   - resultados/benchmark.csv  (uma linha por par PDF x extrator)
 *   - resultados/benchmark.json (agregados por extrator)
 *
 * Uso:
 *   npm run benchmark
 *   npm run benchmark -- --apenas "Aula 04"
 *   npm run benchmark -- --apenas-extrator mupdf
 *
 * MERGE: quando rodado com --apenas ou --apenas-extrator, mescla com o
 * CSV existente — linhas pra pares (PDF, extrator) que NAO foram
 * filtrados sao preservadas. Sem filtro, sobrescreve tudo.
 */

import { readdirSync, readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, parse } from 'node:path';

import * as pdfParseExtrator from './extratores/pdf-parse.js';
import * as pdfjsExtrator from './extratores/pdfjs.js';
import * as unpdfExtrator from './extratores/unpdf.js';
import * as pdf2jsonExtrator from './extratores/pdf2json.js';
import * as mupdfExtrator from './extratores/mupdf.js';
import * as tesseractExtrator from './extratores/tesseract.js';
import * as tesseractUnpdfExtrator from './extratores/tesseract-unpdf.js';
import * as hibridoExtrator from './extratores/hibrido.js';
import * as hibrido2Extrator from './extratores/hibrido2.js';
import * as hibridoUnpdfExtrator from './extratores/hibrido-unpdf.js';
import * as hibridoUnpdfImgExtrator from './extratores/hibrido-unpdf-img.js';
import * as hibridoUnpdfImgHashExtrator from './extratores/hibrido-unpdf-img-hash.js';
import * as hibridoUnpdfImgHashCleanExtrator from './extratores/hibrido-unpdf-img-hash-clean.js';
import * as hibridoCleanBest from './extratores/hibrido-clean-best.js';
import * as hibridoCleanConf from './extratores/hibrido-clean-conf.js';
import * as hibridoCleanPrep from './extratores/hibrido-clean-prep.js';
import * as hibridoCleanMulti from './extratores/hibrido-clean-multi.js';
import * as hibridoCleanConfMulti from './extratores/hibrido-clean-conf-multi.js';
import * as hibridoCleanPrepConf from './extratores/hibrido-clean-prep-conf.js';
import * as hibridoCleanConfBest from './extratores/hibrido-clean-conf-best.js';
import * as doclingExtrator from './extratores/docling.js';
import * as mineruExtrator from './extratores/mineru.js';
import * as hibridoUnpdfBboxExtrator from './extratores/hibrido-unpdf-bbox.js';

import { PROJETO_RAIZ } from './utils/env.js';
import { calcularMetricas } from './utils/metricas.js';
import type { ResultadoExtracao } from './utils/tipos.js';

const PDFS_DIR = join(PROJETO_RAIZ, 'dados', 'pdfs');
const GABARITOS_DIR = join(PROJETO_RAIZ, 'dados', 'gabaritos');
const RESULTADOS_DIR = join(PROJETO_RAIZ, 'resultados');
const CSV_PATH = join(RESULTADOS_DIR, 'benchmark.csv');
const JSON_PATH = join(RESULTADOS_DIR, 'benchmark.json');

interface Extrator {
  nome: string;
  extrair: (caminho: string) => Promise<ResultadoExtracao>;
}

const EXTRATORES: Extrator[] = [
  { nome: pdfParseExtrator.NOME_EXTRATOR, extrair: pdfParseExtrator.extrair },
  { nome: pdfjsExtrator.NOME_EXTRATOR, extrair: pdfjsExtrator.extrair },
  { nome: unpdfExtrator.NOME_EXTRATOR, extrair: unpdfExtrator.extrair },
  { nome: pdf2jsonExtrator.NOME_EXTRATOR, extrair: pdf2jsonExtrator.extrair },
  { nome: mupdfExtrator.NOME_EXTRATOR, extrair: mupdfExtrator.extrair },
  { nome: tesseractExtrator.NOME_EXTRATOR, extrair: tesseractExtrator.extrair },
  { nome: tesseractUnpdfExtrator.NOME_EXTRATOR, extrair: tesseractUnpdfExtrator.extrair },
  { nome: hibridoExtrator.NOME_EXTRATOR, extrair: hibridoExtrator.extrair },
  { nome: hibrido2Extrator.NOME_EXTRATOR, extrair: hibrido2Extrator.extrair },
  { nome: hibridoUnpdfExtrator.NOME_EXTRATOR, extrair: hibridoUnpdfExtrator.extrair },
  { nome: hibridoUnpdfImgExtrator.NOME_EXTRATOR, extrair: hibridoUnpdfImgExtrator.extrair },
  { nome: hibridoUnpdfImgHashExtrator.NOME_EXTRATOR, extrair: hibridoUnpdfImgHashExtrator.extrair },
  { nome: hibridoUnpdfImgHashCleanExtrator.NOME_EXTRATOR, extrair: hibridoUnpdfImgHashCleanExtrator.extrair },
  { nome: hibridoCleanBest.NOME_EXTRATOR, extrair: hibridoCleanBest.extrair },
  { nome: hibridoCleanConf.NOME_EXTRATOR, extrair: hibridoCleanConf.extrair },
  { nome: hibridoCleanPrep.NOME_EXTRATOR, extrair: hibridoCleanPrep.extrair },
  { nome: hibridoCleanMulti.NOME_EXTRATOR, extrair: hibridoCleanMulti.extrair },
  { nome: hibridoCleanConfMulti.NOME_EXTRATOR, extrair: hibridoCleanConfMulti.extrair },
  { nome: hibridoCleanPrepConf.NOME_EXTRATOR, extrair: hibridoCleanPrepConf.extrair },
  { nome: hibridoCleanConfBest.NOME_EXTRATOR, extrair: hibridoCleanConfBest.extrair },
  { nome: doclingExtrator.NOME_EXTRATOR, extrair: doclingExtrator.extrair },
  { nome: mineruExtrator.NOME_EXTRATOR, extrair: mineruExtrator.extrair },
  { nome: hibridoUnpdfBboxExtrator.NOME_EXTRATOR, extrair: hibridoUnpdfBboxExtrator.extrair },
];

interface LinhaCsv {
  pdf: string;
  extrator: string;
  numPaginas: number;
  tempoSeg: number;
  numCaracteres: number;
  simLevenshtein: number;
  coberturaPalavras: number;
  simBigramas: number;
  tamanhoRelativo: number;
  qualidadeExcesso: number;
  erro: string;
}

interface AgregadoExtrator {
  numPdfs: number;
  numErros: number;
  tempoMedio: number;
  simLevenshteinMedia: number;
  coberturaPalavrasMedia: number;
  simBigramasMedia: number;
  tamanhoRelativoMedio: number;
  qualidadeExcessoMedia: number;
}

function parseArgs(): { filtroPdf: string | null; filtroExtrator: string | null } {
  const args = process.argv.slice(2);
  let filtroPdf: string | null = null;
  let filtroExtrator: string | null = null;
  const idxP = args.indexOf('--apenas');
  if (idxP !== -1 && args[idxP + 1]) filtroPdf = args[idxP + 1]!;
  const idxE = args.indexOf('--apenas-extrator');
  if (idxE !== -1 && args[idxE + 1]) filtroExtrator = args[idxE + 1]!;
  return { filtroPdf, filtroExtrator };
}

function escaparCsv(v: string): string {
  if (/[",\n]/.test(v)) return `"${v.replace(/"/g, '""')}"`;
  return v;
}

function linhaParaCsv(l: LinhaCsv): string {
  return [
    escaparCsv(l.pdf),
    escaparCsv(l.extrator),
    String(l.numPaginas),
    l.tempoSeg.toFixed(3),
    String(l.numCaracteres),
    l.simLevenshtein.toFixed(4),
    l.coberturaPalavras.toFixed(4),
    l.simBigramas.toFixed(4),
    l.tamanhoRelativo.toFixed(4),
    l.qualidadeExcesso.toFixed(4),
    escaparCsv(l.erro),
  ].join(',');
}

const CSV_HEADER =
  'pdf,extrator,num_paginas,tempo_seg,num_caracteres,sim_levenshtein,cobertura_palavras,sim_bigramas,tamanho_relativo,qualidade_excesso,erro';

function parsearLinhaCsv(linha: string): string[] {
  const cols: string[] = [];
  let atual = '';
  let dentroAspas = false;
  for (let i = 0; i < linha.length; i++) {
    const c = linha[i]!;
    if (dentroAspas) {
      if (c === '"' && linha[i + 1] === '"') {
        atual += '"';
        i++;
      } else if (c === '"') {
        dentroAspas = false;
      } else {
        atual += c;
      }
    } else if (c === '"') {
      dentroAspas = true;
    } else if (c === ',') {
      cols.push(atual);
      atual = '';
    } else {
      atual += c;
    }
  }
  cols.push(atual);
  return cols;
}

function lerCsvExistente(path: string): LinhaCsv[] {
  if (!existsSync(path)) return [];
  const conteudo = readFileSync(path, 'utf-8');
  const linhas = conteudo.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (linhas.length <= 1) return [];
  const dados: LinhaCsv[] = [];
  for (let i = 1; i < linhas.length; i++) {
    const cols = parsearLinhaCsv(linhas[i]!);
    if (cols.length < 10) continue;
    // Compatibilidade: CSV antigo tem 10 colunas (sem qualidade_excesso),
    // novo tem 11. Linhas antigas ficam com qualidade_excesso = 0 ate
    // serem reprocessadas.
    const ehNovoFormato = cols.length >= 11;
    dados.push({
      pdf: cols[0]!,
      extrator: cols[1]!,
      numPaginas: Number(cols[2]),
      tempoSeg: Number(cols[3]),
      numCaracteres: Number(cols[4]),
      simLevenshtein: Number(cols[5]),
      coberturaPalavras: Number(cols[6]),
      simBigramas: Number(cols[7]),
      tamanhoRelativo: Number(cols[8]),
      qualidadeExcesso: ehNovoFormato ? Number(cols[9]) : 0,
      erro: ehNovoFormato ? cols[10] ?? '' : cols[9] ?? '',
    });
  }
  return dados;
}

function mesclarLinhas(antigas: LinhaCsv[], novas: LinhaCsv[]): LinhaCsv[] {
  // Pares (pdf, extrator) das linhas novas têm prioridade. Antigas
  // que não foram refeitas são preservadas. Resultado ordenado por
  // pdf depois extrator.
  const novasKeys = new Set(novas.map((l) => `${l.pdf}|${l.extrator}`));
  const mantidas = antigas.filter((l) => !novasKeys.has(`${l.pdf}|${l.extrator}`));
  const todas = [...mantidas, ...novas];
  todas.sort(
    (a, b) => a.pdf.localeCompare(b.pdf) || a.extrator.localeCompare(b.extrator),
  );
  return todas;
}

function calcularAgregados(linhas: LinhaCsv[]): Record<string, AgregadoExtrator> {
  const agg: Record<string, AgregadoExtrator> = {};
  for (const ex of EXTRATORES) {
    const doExtrator = linhas.filter((l) => l.extrator === ex.nome);
    const semErro = doExtrator.filter((l) => l.erro === '');
    const n = semErro.length;
    const media = (campo: keyof LinhaCsv): number => {
      if (n === 0) return 0;
      let s = 0;
      for (const l of semErro) s += Number(l[campo]);
      return Number((s / n).toFixed(4));
    };
    agg[ex.nome] = {
      numPdfs: doExtrator.length,
      numErros: doExtrator.length - n,
      tempoMedio: media('tempoSeg'),
      simLevenshteinMedia: media('simLevenshtein'),
      coberturaPalavrasMedia: media('coberturaPalavras'),
      simBigramasMedia: media('simBigramas'),
      tamanhoRelativoMedio: media('tamanhoRelativo'),
      qualidadeExcessoMedia: media('qualidadeExcesso'),
    };
  }
  return agg;
}

async function main(): Promise<number> {
  const { filtroPdf, filtroExtrator } = parseArgs();

  if (!existsSync(RESULTADOS_DIR)) mkdirSync(RESULTADOS_DIR, { recursive: true });

  let pdfs = readdirSync(PDFS_DIR)
    .filter((f) => f.toLowerCase().endsWith('.pdf'))
    .sort();
  if (filtroPdf) {
    pdfs = pdfs.filter((f) => f.toLowerCase().includes(filtroPdf.toLowerCase()));
  }

  const extratoresAtivos = filtroExtrator
    ? EXTRATORES.filter((e) =>
        e.nome.toLowerCase().includes(filtroExtrator.toLowerCase()),
      )
    : EXTRATORES;

  if (pdfs.length === 0) {
    console.log(`Nenhum PDF em ${PDFS_DIR} (filtro=${JSON.stringify(filtroPdf)})`);
    return 1;
  }
  if (extratoresAtivos.length === 0) {
    console.log(`Nenhum extrator casa com filtro=${JSON.stringify(filtroExtrator)}`);
    return 1;
  }

  console.log(
    `Benchmark: ${pdfs.length} PDFs x ${extratoresAtivos.length} extratores = ` +
      `${pdfs.length * extratoresAtivos.length} execucoes\n`,
  );

  const linhas: LinhaCsv[] = [];

  for (const pdf of pdfs) {
    const nome = parse(pdf).name;
    const caminhoPdf = join(PDFS_DIR, pdf);
    const caminhoGabarito = join(GABARITOS_DIR, `${nome}.txt`);

    if (!existsSync(caminhoGabarito)) {
      console.log(`[SKIP PDF] ${nome} — sem gabarito em ${caminhoGabarito}`);
      continue;
    }

    const gabarito = readFileSync(caminhoGabarito, 'utf-8');
    console.log(`[PDF] ${nome} (${gabarito.length} chars no gabarito)`);

    for (const ex of extratoresAtivos) {
      process.stdout.write(`  - ${ex.nome.padEnd(12)} ... `);
      const r = await ex.extrair(caminhoPdf);
      const m = calcularMetricas(r.texto, gabarito);
      const linha: LinhaCsv = {
        pdf: nome,
        extrator: ex.nome,
        numPaginas: r.numPaginas,
        tempoSeg: r.tempoSegundos,
        numCaracteres: r.numCaracteres,
        simLevenshtein: m.similaridadeLevenshtein,
        coberturaPalavras: m.coberturaPalavras,
        simBigramas: m.similaridadeBigramas,
        tamanhoRelativo: m.tamanhoRelativo,
        qualidadeExcesso: m.qualidadeExcesso,
        erro: r.erro ?? '',
      };
      linhas.push(linha);
      if (r.erro) {
        console.log(`ERRO em ${r.tempoSegundos.toFixed(2)}s: ${r.erro.slice(0, 80)}`);
      } else {
        console.log(
          `${r.tempoSegundos.toFixed(2)}s, ${r.numCaracteres} chars, ` +
            `lev=${m.similaridadeLevenshtein.toFixed(3)} ` +
            `cob=${m.coberturaPalavras.toFixed(3)} ` +
            `bi=${m.similaridadeBigramas.toFixed(3)} ` +
            `tam=${m.tamanhoRelativo.toFixed(2)} ` +
            `qexc=${m.qualidadeExcesso.toFixed(3)}`,
        );
      }
    }
  }

  if (linhas.length === 0) {
    console.log('\nNenhuma execucao concluida — sem PDFs com gabarito.');
    return 1;
  }

  // Merge: se houve filtro, preserva linhas antigas pra pares (PDF, extrator)
  // que NAO foram refeitos nessa execucao. Sem filtro, sobrescreve tudo.
  const houveFiltro = filtroPdf !== null || filtroExtrator !== null;
  let todasLinhas: LinhaCsv[];
  if (houveFiltro) {
    const antigas = lerCsvExistente(CSV_PATH);
    todasLinhas = mesclarLinhas(antigas, linhas);
    const mantidas = todasLinhas.length - linhas.length;
    console.log(
      `\nMerge: ${mantidas} linha(s) preservada(s) do CSV anterior, ` +
        `${linhas.length} nova(s)/atualizada(s).`,
    );
  } else {
    todasLinhas = linhas;
  }

  const csv = [CSV_HEADER, ...todasLinhas.map(linhaParaCsv)].join('\n');
  writeFileSync(CSV_PATH, csv + '\n', 'utf-8');

  const agregados = calcularAgregados(todasLinhas);
  writeFileSync(JSON_PATH, JSON.stringify({ porExtrator: agregados }, null, 2), 'utf-8');

  console.log(`\nCSV:  ${CSV_PATH}`);
  console.log(`JSON: ${JSON_PATH}`);
  console.log('\nResumo (medias dos PDFs sem erro):');
  for (const [nome, a] of Object.entries(agregados)) {
    console.log(
      `  ${nome.padEnd(30)} ` +
        `lev=${a.simLevenshteinMedia.toFixed(3)} ` +
        `cob=${a.coberturaPalavrasMedia.toFixed(3)} ` +
        `bi=${a.simBigramasMedia.toFixed(3)} ` +
        `tam=${a.tamanhoRelativoMedio.toFixed(2)} ` +
        `qexc=${a.qualidadeExcessoMedia.toFixed(3)} ` +
        `tempo=${a.tempoMedio.toFixed(2)}s ` +
        `erros=${a.numErros}/${a.numPdfs}`,
    );
  }

  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
