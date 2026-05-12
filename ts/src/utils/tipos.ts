/**
 * Tipos compartilhados entre extratores e scripts CLI.
 *
 * Convenção: campos comuns no `ResultadoExtracao`. Extratores específicos
 * (Claude, com tokens/custo) estendem com campos extras.
 */

export interface ResultadoExtracao {
  extrator: string;
  texto: string;
  paginas: string[];
  numPaginas: number;
  numCaracteres: number;
  tempoSegundos: number;
  erro: string | null;
}

export interface ChunkInfo {
  paginas: string;
  depth: number;
  stopReason: string | null;
  erro: string | null;
  foiDividido: boolean;
}

export interface ResultadoClaude extends ResultadoExtracao {
  tokensInput: number;
  tokensOutput: number;
  custoUsd: number;
  numChunks: number;
  chunks: ChunkInfo[];
  recusaDetectada: boolean;
}
