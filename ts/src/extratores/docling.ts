/**
 * Extrator Docling (IBM): converte PDF pra markdown estruturado.
 *
 * Wrapper sobre o CLI `docling` instalado na .venv do projeto. Spawn
 * subprocess Python, captura output (.md) e devolve no contrato
 * ResultadoExtracao.
 *
 * Pre-requisito: ter rodado `pip install docling` na .venv (1x setup).
 *
 * Trade-offs vs extratores TS puros:
 *   - MUITO mais lento (~30-90s/PDF no CPU vs ~0.1-15s dos outros)
 *   - Output em markdown estruturado (titulos, listas) — pode ter mais
 *     "qualidade percebida" pelo usuario final do tutor-ai
 *   - Suporte nativo a layout, tabelas, formulas
 *   - Licenca MIT (compativel com tutor-ai)
 *
 * Limitacoes conhecidas:
 *   - Falha em alguns PDFs com encoding nao-UTF8 (ex: Manual de Pesquisa)
 *   - Embute imagens base64 inline por padrao — usamos `--image-export-mode
 *     placeholder` pra evitar inflar o output em MBs
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { join, parse } from 'node:path';

import { PROJETO_RAIZ } from '../utils/env.js';
import type { ResultadoExtracao } from '../utils/tipos.js';

export const NOME_EXTRATOR = 'docling';

const DOCLING_BIN = join(PROJETO_RAIZ, '.venv', 'Scripts', 'docling.exe');
const TMP_DIR = join(PROJETO_RAIZ, '.docling-tmp');

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
    if (!existsSync(DOCLING_BIN)) {
      return {
        extrator: NOME_EXTRATOR,
        texto: '',
        paginas: [],
        numPaginas: 0,
        numCaracteres: 0,
        tempoSegundos: (performance.now() - inicio) / 1000,
        erro: `docling.exe nao encontrado em ${DOCLING_BIN}. Rode 'pip install docling' na .venv.`,
      };
    }

    if (existsSync(TMP_DIR)) rmSync(TMP_DIR, { recursive: true, force: true });
    mkdirSync(TMP_DIR, { recursive: true });

    // Usa pypdfium2 (Google PDFium) em vez do `docling-parse` default —
    // o default tem 2 bugs conhecidos no nosso conjunto:
    //   - "Inconsistent number of pages" em alguns PDFs (EPM 309 STP, 5S)
    //   - "arquivo nao gerado" em Manual (encoding interno)
    // pypdfium2 contorna ambos.
    const { code, stderr } = await spawnPromise(DOCLING_BIN, [
      caminhoPdf,
      '--to', 'md',
      '--image-export-mode', 'placeholder',
      '--pdf-backend', 'pypdfium2',
      '--output', TMP_DIR,
    ]);

    if (code !== 0) {
      return {
        extrator: NOME_EXTRATOR,
        texto: '',
        paginas: [],
        numPaginas: 0,
        numCaracteres: 0,
        tempoSegundos: (performance.now() - inicio) / 1000,
        erro: `docling exit ${code}: ${stderr.slice(0, 300)}`,
      };
    }

    const nomeBase = parse(caminhoPdf).name;
    const arquivoMd = join(TMP_DIR, `${nomeBase}.md`);
    if (!existsSync(arquivoMd)) {
      // procura por outro nome (docling pode renomear)
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
      paginas: [], // docling nao separa por pagina em markdown
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
