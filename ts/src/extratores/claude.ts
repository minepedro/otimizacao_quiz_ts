/**
 * Extrator de PDF usando Claude (gold standard do mini-laboratorio).
 *
 * Porte do `lab/extratores/extrator_claude.py` do lab Python. Mesma
 * heuristica de chunking recursivo e detector de recusa por copyright.
 *
 * Modelo: claude-sonnet-4-6.
 * Chunking: se a API der erro OU se a resposta truncar (`stop_reason === 'max_tokens'`),
 * divide o PDF ao meio por pagina e tenta cada metade.
 * Detector de recusa: muito input + pouco output + fim normal = provavel
 * recusa por copyright; nao divide, propaga.
 */

import Anthropic from '@anthropic-ai/sdk';
import { readFileSync } from 'node:fs';
import { PDFDocument } from 'pdf-lib';

import { carregarChaveAnthropic } from '../utils/env.js';
import type { ChunkInfo, ResultadoClaude } from '../utils/tipos.js';

const MODELO = 'claude-sonnet-4-6';

// Precos oficiais Sonnet 4.6 (USD por milhao de tokens)
const INPUT_COST_PER_MTOKEN = 3.0;
const OUTPUT_COST_PER_MTOKEN = 15.0;

const MAX_TOKENS_RESPOSTA = 64_000;

// Heuristica de deteccao de recusa silenciosa.
const RECUSA_INPUT_MIN = 10_000;
const RECUSA_OUTPUT_MAX = 500;
export const PREFIXO_ERRO_RECUSA = 'RECUSA:';

// Rate limit: erro 429 da API. Diferente de recusa — vale a pena re-tentar
// com espera. Chunkar NAO resolve (so piora, gera mais chamadas).
export const PREFIXO_ERRO_RATE_LIMIT = 'RATE_LIMIT:';
const MAX_RETRIES_SDK = 6; // SDK Anthropic ja faz retry; aumentamos do default 2
const MAX_RETRIES_MANUAIS = 3; // se SDK desistiu, tentamos mais N vezes com sleep
const SLEEP_BASE_SEG = 30; // 1a tentativa manual: 30s, depois 60s, depois 120s

const PROMPT_TRANSCRICAO_LITERAL = `\
Transcreva fielmente todo o conteudo textual deste PDF, em ordem natural \
de leitura. Regras:

1. NAO resuma, NAO interprete, NAO comente — apenas transcreva.
2. Preserve a ordem visual: titulos antes de corpo; em layout multi-coluna, \
coluna da esquerda antes da direita.
3. Pule cabecalhos/rodapes repetitivos (numeros de pagina, nome do arquivo \
no rodape, etc).
4. Separe paginas com duas quebras de linha (\\n\\n).
5. Nao use markdown; produza texto puro.
6. NAO adicione preambulo, titulo, "Aqui esta a transcricao:" nem comentarios. \
Comece direto com o primeiro conteudo do PDF e termine no ultimo.
`;

interface RetornoChamada {
  texto: string;
  tokensInput: number;
  tokensOutput: number;
  custoUsd: number;
  stopReason: string | null;
  erro: string | null;
}

function ehErroRateLimit(msg: string): boolean {
  return msg.includes('429') || msg.toLowerCase().includes('rate_limit');
}

async function dormir(seg: number): Promise<void> {
  return new Promise((r) => setTimeout(r, seg * 1000));
}

async function chamarClaudeUmaVez(
  client: Anthropic,
  pdfBytes: Uint8Array,
): Promise<RetornoChamada> {
  try {
    const pdfBase64 = Buffer.from(pdfBytes).toString('base64');

    // Streaming necessario pra max_tokens alto.
    const stream = client.messages.stream({
      model: MODELO,
      max_tokens: MAX_TOKENS_RESPOSTA,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'document',
              source: {
                type: 'base64',
                media_type: 'application/pdf',
                data: pdfBase64,
              },
            },
            { type: 'text', text: PROMPT_TRANSCRICAO_LITERAL },
          ],
        },
      ],
    });

    // Drena o stream.
    for await (const _ of stream) {
      // necessario consumir pra completar
    }
    const final = await stream.finalMessage();

    const primeiroBloco = final.content[0];
    const texto = primeiroBloco?.type === 'text' ? primeiroBloco.text : '';
    const tokensInput = final.usage.input_tokens;
    const tokensOutput = final.usage.output_tokens;
    const custoUsd = Number(
      (
        (tokensInput * INPUT_COST_PER_MTOKEN +
          tokensOutput * OUTPUT_COST_PER_MTOKEN) /
        1_000_000
      ).toFixed(4),
    );

    // Detector de recusa.
    const ehRecusa =
      final.stop_reason === 'end_turn' &&
      tokensInput >= RECUSA_INPUT_MIN &&
      tokensOutput < RECUSA_OUTPUT_MAX;

    let erro: string | null = null;
    if (ehRecusa) {
      const amostra = texto.slice(0, 200).replace(/\n/g, ' ');
      erro =
        `${PREFIXO_ERRO_RECUSA} output_tokens=${tokensOutput} pra ` +
        `input_tokens=${tokensInput}. Provavel recusa do modelo ` +
        `(copyright/policy). Resposta: ${JSON.stringify(amostra)}`;
    }

    return {
      texto,
      tokensInput,
      tokensOutput,
      custoUsd,
      stopReason: final.stop_reason,
      erro,
    };
  } catch (e) {
    const msg = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    const erro = ehErroRateLimit(msg) ? `${PREFIXO_ERRO_RATE_LIMIT} ${msg}` : msg;
    return {
      texto: '',
      tokensInput: 0,
      tokensOutput: 0,
      custoUsd: 0,
      stopReason: null,
      erro,
    };
  }
}

async function chamarClaudeComPdfBytes(
  pdfBytes: Uint8Array,
  chave: string,
  verbose: boolean = false,
): Promise<RetornoChamada> {
  // SDK ja faz retries proprios (default 2, aumentamos pra MAX_RETRIES_SDK).
  // Em caso de 429 que escape, fazemos retries manuais com sleep escalonado.
  const client = new Anthropic({ apiKey: chave, maxRetries: MAX_RETRIES_SDK });

  for (let i = 0; i <= MAX_RETRIES_MANUAIS; i++) {
    const r = await chamarClaudeUmaVez(client, pdfBytes);
    if (!r.erro || !r.erro.startsWith(PREFIXO_ERRO_RATE_LIMIT)) return r;
    if (i === MAX_RETRIES_MANUAIS) return r; // ultima tentativa, propaga
    const sleep = SLEEP_BASE_SEG * Math.pow(2, i); // 30s, 60s, 120s
    if (verbose) {
      console.log(`        rate limit: aguardando ${sleep}s antes de re-tentar (${i + 1}/${MAX_RETRIES_MANUAIS})...`);
    }
    await dormir(sleep);
  }
  // Nao deveria chegar aqui (loop sempre retorna), mas TS exige
  return {
    texto: '',
    tokensInput: 0,
    tokensOutput: 0,
    custoUsd: 0,
    stopReason: null,
    erro: `${PREFIXO_ERRO_RATE_LIMIT} retries esgotados`,
  };
}

async function fatiarPdf(
  bytes: Uint8Array,
  start: number,
  end: number,
): Promise<Uint8Array> {
  const original = await PDFDocument.load(bytes);
  const sub = await PDFDocument.create();
  const indices = Array.from({ length: end - start }, (_, i) => start + i);
  const pages = await sub.copyPages(original, indices);
  for (const p of pages) sub.addPage(p);
  return await sub.save();
}

async function contarPaginas(bytes: Uint8Array): Promise<number> {
  try {
    const doc = await PDFDocument.load(bytes);
    return doc.getPageCount();
  } catch {
    return 0;
  }
}

interface OpcoesChunking {
  maxProfundidade?: number;
  verbose?: boolean;
}

/**
 * Extrator Claude com chunking recursivo.
 *
 * Tenta o PDF inteiro primeiro. Se a chamada falhar (exceto recusa) ou
 * a resposta truncar, divide ao meio e chama recursivamente. Para de
 * descer quando chunk tem 1 pagina ou bate `maxProfundidade`.
 */
export async function extrairComChunking(
  caminhoPdf: string,
  opcoes: OpcoesChunking = {},
): Promise<ResultadoClaude> {
  const { maxProfundidade = 6, verbose = true } = opcoes;
  const inicio = performance.now();
  const nomeExtrator = 'claude_sonnet_4_6_chunked';

  const chave = carregarChaveAnthropic();
  if (!chave) {
    return {
      extrator: nomeExtrator,
      texto: '',
      paginas: [],
      numPaginas: 0,
      numCaracteres: 0,
      tempoSegundos: (performance.now() - inicio) / 1000,
      erro: 'ANTHROPIC_API_KEY nao configurada (nem no env, nem no .env)',
      tokensInput: 0,
      tokensOutput: 0,
      custoUsd: 0,
      numChunks: 0,
      chunks: [],
      recusaDetectada: false,
    };
  }

  const pdfBytes = readFileSync(caminhoPdf);
  const numPaginas = await contarPaginas(pdfBytes);

  const stats = {
    tokensInput: 0,
    tokensOutput: 0,
    custoUsd: 0,
    numChunks: 0,
    chunks: [] as ChunkInfo[],
    recusaDetectada: false,
  };

  async function recurse(
    start: number,
    end: number,
    depth: number,
  ): Promise<string> {
    const n = end - start;
    const label = `pg ${start + 1}-${end}`;
    const prefixo = '  '.repeat(depth);
    if (verbose) {
      console.log(`${prefixo}[chunk ${label}] ${n} pg, depth=${depth}`);
    }

    const sliceBytes = await fatiarPdf(pdfBytes, start, end);
    const r = await chamarClaudeComPdfBytes(sliceBytes, chave!, verbose);

    stats.tokensInput += r.tokensInput;
    stats.tokensOutput += r.tokensOutput;
    stats.custoUsd = Number((stats.custoUsd + r.custoUsd).toFixed(4));
    stats.numChunks += 1;
    const registro: ChunkInfo = {
      paginas: label,
      depth,
      stopReason: r.stopReason,
      erro: r.erro,
      foiDividido: false,
    };
    stats.chunks.push(registro);

    // Rate limit NAO se resolve dividindo (so gera mais chamadas e mais erros).
    // chamarClaudeComPdfBytes ja fez retries com sleep antes de propagar.
    if (r.erro && r.erro.startsWith(PREFIXO_ERRO_RATE_LIMIT)) {
      if (verbose) console.log(`${prefixo}  -> rate limit persistiu, NAO dividindo`);
      return r.texto;
    }

    // Recusa NAO se resolve dividindo.
    const ehRecusa = !!r.erro && r.erro.startsWith(PREFIXO_ERRO_RECUSA);
    if (ehRecusa) {
      if (verbose) console.log(`${prefixo}  -> recusa detectada, NAO dividindo`);
      stats.recusaDetectada = true;
      return r.texto;
    }

    const precisaDividir = !!r.erro || r.stopReason === 'max_tokens';

    if (precisaDividir && n > 1 && depth < maxProfundidade) {
      registro.foiDividido = true;
      if (verbose) {
        const motivo = r.erro ? 'erro' : 'max_tokens';
        console.log(`${prefixo}  -> dividindo (${motivo})`);
      }
      const meio = start + Math.floor(n / 2);
      const esq = await recurse(start, meio, depth + 1);
      const dir = await recurse(meio, end, depth + 1);
      return esq + '\n\n' + dir;
    }

    if (precisaDividir) {
      if (verbose) {
        console.log(`${prefixo}  -> chunk indivisivel, retornando como esta`);
      }
      return r.texto;
    }

    return r.texto;
  }

  if (verbose) console.log(`Extracao recursiva: ${caminhoPdf} (${numPaginas} pg)`);
  const texto = await recurse(0, numPaginas, 0);

  let erroFinal: string | null = null;
  if (stats.recusaDetectada) {
    erroFinal =
      `${PREFIXO_ERRO_RECUSA} modelo recusou transcrever este PDF ` +
      '(provavel copyright/policy). Veja chunks[].erro pra detalhes.';
  }

  return {
    extrator: nomeExtrator,
    texto,
    paginas: texto ? texto.split('\n\n') : [],
    numPaginas,
    numCaracteres: texto.length,
    tempoSegundos: (performance.now() - inicio) / 1000,
    erro: erroFinal,
    tokensInput: stats.tokensInput,
    tokensOutput: stats.tokensOutput,
    custoUsd: stats.custoUsd,
    numChunks: stats.numChunks,
    chunks: stats.chunks,
    recusaDetectada: stats.recusaDetectada,
  };
}
