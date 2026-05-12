/**
 * Extrator MinerU (OpenDataLab): converte PDF pra markdown estruturado
 * via pipeline de modelos ML (layout, OCR, table, formula).
 *
 * Wrapper sobre o CLI `mineru` instalado na .venv do projeto.
 *
 * Pre-requisitos importantes (descobertos na dor):
 *
 *   1. Python 3.10-3.12 no Windows (NAO funciona com 3.13 — dependencia
 *      `ray` nao suporta Python 3.13 + Windows. Sintoma: trava sem
 *      output. Veja CLAUDE.md.)
 *
 *   2. PyTorch com CUDA pra usar GPU (CPU funciona mas eh dolorosamente
 *      lento — 5+ min/PDF). Com RTX 5070 + cu128: ~40s/PDF.
 *
 *   3. Modelos pre-baixados em ~/.cache/huggingface (~3GB).
 *      Comando: `mineru-models-download -s huggingface -m pipeline`
 *
 * Estrutura de output do MinerU:
 *   <output_dir>/<nome_base>/auto/<nome_base>.md  ← arquivo principal
 *   <output_dir>/<nome_base>/auto/...             ← jsons + pdfs auxiliares
 *   <output_dir>/<nome_base>/auto/images/         ← imagens extraidas
 *
 * Backend `pipeline` (escolhido) eh o mais simples e CPU-friendly.
 * Outras opcoes: vlm-auto-engine (alta precisao, GPU forte), hibrido.
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { join, parse } from 'node:path';

import { PROJETO_RAIZ } from '../utils/env.js';
import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'mineru';

const MINERU_BIN = join(PROJETO_RAIZ, '.venv', 'Scripts', 'mineru.exe');
const TMP_DIR = join(PROJETO_RAIZ, '.mineru-tmp');

interface ResultadoSpawn {
  code: number;
  stderr: string;
}

function spawnPromise(cmd: string, args: string[]): Promise<ResultadoSpawn> {
  return new Promise((resolve) => {
    const proc = spawn(cmd, args, { windowsHide: true });
    let stderr = '';
    proc.stderr.on('data', (d: Buffer) => {
      stderr += d.toString();
    });
    proc.on('close', (code) => resolve({ code: code ?? -1, stderr }));
    proc.on('error', (e) => resolve({ code: -1, stderr: String(e) }));
  });
}

export async function extrair(caminhoPdf: string): Promise<ResultadoExtracao> {
  const inicio = performance.now();
  try {
    if (!existsSync(MINERU_BIN)) {
      return {
        extrator: NOME_EXTRATOR,
        texto: '',
        paginas: [],
        numPaginas: 0,
        numCaracteres: 0,
        tempoSegundos: (performance.now() - inicio) / 1000,
        erro: `mineru.exe nao encontrado em ${MINERU_BIN}. Rode 'pip install mineru[all]' na .venv (Python 3.10-3.12).`,
      };
    }

    if (existsSync(TMP_DIR)) rmSync(TMP_DIR, { recursive: true, force: true });
    mkdirSync(TMP_DIR, { recursive: true });

    const { code, stderr } = await spawnPromise(MINERU_BIN, [
      '-p', caminhoPdf,
      '-o', TMP_DIR,
      '-b', 'pipeline',
      '-l', 'latin',
      '-d', 'cuda', // usa GPU se disponivel, cai pra CPU se nao tiver
    ]);

    if (code !== 0) {
      return {
        extrator: NOME_EXTRATOR,
        texto: '',
        paginas: [],
        numPaginas: 0,
        numCaracteres: 0,
        tempoSegundos: (performance.now() - inicio) / 1000,
        erro: `mineru exit ${code}: ${stderr.slice(-300)}`,
      };
    }

    // Estrutura: <TMP_DIR>/<nome_base>/auto/<nome_base>.md
    const nomeBase = parse(caminhoPdf).name;
    const arquivoMd = join(TMP_DIR, nomeBase, 'auto', `${nomeBase}.md`);
    if (!existsSync(arquivoMd)) {
      return {
        extrator: NOME_EXTRATOR,
        texto: '',
        paginas: [],
        numPaginas: 0,
        numCaracteres: 0,
        tempoSegundos: (performance.now() - inicio) / 1000,
        erro: `arquivo nao gerado: ${arquivoMd}`,
      };
    }

    const texto = readFileSync(arquivoMd, 'utf-8');
    return {
      extrator: NOME_EXTRATOR,
      texto,
      paginas: [], // MinerU nao separa por pagina no markdown
      numPaginas: 0,
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
