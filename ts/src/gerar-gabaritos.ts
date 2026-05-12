/**
 * Gera gabaritos (gold standard) chamando extrator Claude pra cada PDF.
 *
 * Idempotente: pula PDF se o .txt do gabarito ja existe, OU se metadata
 * registra recusa anterior do modelo. Use `--forcar` pra ignorar ambos.
 *
 * Uso:
 *   npm run gerar-gabaritos
 *   npm run gerar-gabaritos -- --forcar
 *   npm run gerar-gabaritos -- --apenas "Aula 04"
 */

import { readdirSync, readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, parse } from 'node:path';

import { extrairComChunking } from './extratores/claude.js';
import { PROJETO_RAIZ } from './utils/env.js';

const PDFS_DIR = join(PROJETO_RAIZ, 'dados', 'pdfs');
const GABARITOS_DIR = join(PROJETO_RAIZ, 'dados', 'gabaritos');
const RESULTADOS_DIR = join(PROJETO_RAIZ, 'resultados');
const METADATA_PATH = join(RESULTADOS_DIR, 'gabaritos_metadata.json');

interface MetadataEntry {
  numPaginas?: number;
  numCaracteres?: number;
  tempoSegundos?: number;
  tokensInput?: number;
  tokensOutput?: number;
  custoUsd?: number;
  numChunks?: number;
  chunks?: unknown[];
  erro?: string;
  recusaDetectada?: boolean;
}

type Metadata = Record<string, MetadataEntry>;

function carregarMetadata(): Metadata {
  if (!existsSync(METADATA_PATH)) return {};
  return JSON.parse(readFileSync(METADATA_PATH, 'utf-8')) as Metadata;
}

function salvarMetadata(m: Metadata): void {
  writeFileSync(METADATA_PATH, JSON.stringify(m, null, 2), 'utf-8');
}

function parseArgs(): { forcar: boolean; filtro: string | null } {
  const args = process.argv.slice(2);
  const forcar = args.includes('--forcar');
  let filtro: string | null = null;
  const idx = args.indexOf('--apenas');
  if (idx !== -1 && args[idx + 1]) filtro = args[idx + 1]!;
  return { forcar, filtro };
}

async function main(): Promise<number> {
  const { forcar, filtro } = parseArgs();

  if (!existsSync(GABARITOS_DIR)) mkdirSync(GABARITOS_DIR, { recursive: true });
  if (!existsSync(RESULTADOS_DIR)) mkdirSync(RESULTADOS_DIR, { recursive: true });

  const metadata = carregarMetadata();

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

  console.log(`Encontrados ${pdfs.length} PDFs para processar.\n`);
  let custoTotal = 0;

  for (const pdf of pdfs) {
    const nome = parse(pdf).name;
    const caminhoPdf = join(PDFS_DIR, pdf);
    const destinoTxt = join(GABARITOS_DIR, `${nome}.txt`);

    if (existsSync(destinoTxt) && !forcar) {
      console.log(`[SKIP] ${nome} — gabarito ja existe`);
      continue;
    }

    const entradaAnterior = metadata[nome];
    if (!forcar && entradaAnterior?.recusaDetectada) {
      console.log(
        `[SKIP] ${nome} — recusa do modelo registrada anteriormente ` +
          '(use --forcar pra tentar de novo)',
      );
      continue;
    }

    console.log(`[RUN ] ${nome}`);
    const resultado = await extrairComChunking(caminhoPdf);

    if (resultado.erro) {
      console.log(`        erro: ${resultado.erro}`);
      metadata[nome] = {
        erro: resultado.erro,
        tempoSegundos: Number(resultado.tempoSegundos.toFixed(2)),
        numPaginas: resultado.numPaginas,
        tokensInput: resultado.tokensInput,
        tokensOutput: resultado.tokensOutput,
        custoUsd: resultado.custoUsd,
        recusaDetectada: resultado.recusaDetectada,
      };
      salvarMetadata(metadata);
      continue;
    }

    writeFileSync(destinoTxt, resultado.texto, 'utf-8');
    custoTotal += resultado.custoUsd;
    metadata[nome] = {
      numPaginas: resultado.numPaginas,
      numCaracteres: resultado.numCaracteres,
      tempoSegundos: Number(resultado.tempoSegundos.toFixed(2)),
      tokensInput: resultado.tokensInput,
      tokensOutput: resultado.tokensOutput,
      custoUsd: resultado.custoUsd,
      numChunks: resultado.numChunks,
      chunks: resultado.chunks,
    };

    const extra =
      resultado.numChunks > 1 ? `, ${resultado.numChunks} chunks` : '';
    console.log(
      `        ok: ${resultado.numCaracteres} chars, ${resultado.tokensInput} in / ${resultado.tokensOutput} out, $${resultado.custoUsd}${extra}`,
    );

    salvarMetadata(metadata);
  }

  console.log(`\nCusto desta execucao: $${custoTotal.toFixed(4)} USD`);
  console.log(`Metadata: ${METADATA_PATH}`);
  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
