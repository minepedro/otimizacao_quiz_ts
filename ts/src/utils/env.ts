/**
 * Carrega `ANTHROPIC_API_KEY` do `.env` na raiz do REPO.
 *
 * Estrutura do repo (apos reorganizacao dual TS+Python):
 *   otimizacao_quiz_ts/         <- raiz do repo (.env vive aqui)
 *   ├── ts/                     <- pasta deste codigo
 *   │   └── src/utils/env.ts    <- voce esta aqui
 *   ├── py/                     <- equivalente em Python
 *   ├── dados/                  <- compartilhado entre TS e Python
 *   └── resultados/             <- compartilhado
 *
 * Por isso o RAIZ sobe 3 niveis (em vez de 2 como antes da reorganizacao).
 * `PROJETO_RAIZ` aponta pra raiz do REPO, nao da pasta `ts/`.
 *
 * Procura ANTHROPIC_API_KEY nesta ordem:
 * 1. `process.env.ANTHROPIC_API_KEY` (já configurada na shell)
 * 2. `.env` na raiz do REPO (gitignored)
 *
 * Devolve `null` se não encontrar — caller decide como lidar.
 */

import { config } from 'dotenv';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const RAIZ = resolve(__dirname, '../../..'); // ts/src/utils -> ts/src -> ts/ -> raiz_repo
const ENV_PATH = resolve(RAIZ, '.env');

config({ path: ENV_PATH });

export function carregarChaveAnthropic(): string | null {
  const chave = process.env.ANTHROPIC_API_KEY;
  if (!chave || !chave.trim()) return null;
  return chave.trim();
}

export const PROJETO_RAIZ = RAIZ;
