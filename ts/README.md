# Lab TS — extratores de PDF em TypeScript

Subpasta TS do repo `otimizacao_quiz_ts`. Implementa **15+ extratores**
de PDF com 5 métricas de comparação contra gabarito Claude API.

> Veja o [README raiz](../README.md) pra entender a arquitetura dual
> TS+Python do repo.

## Stack

- **Node 20+** (testado em 24.x)
- **TypeScript strict** rodado direto via `tsx` (sem build step)
- **ESM** (`"type": "module"` no package.json)
- Imports usam extensão `.js` mesmo o arquivo sendo `.ts` (NodeNext)

## Quickstart

```powershell
npm install              # instala dependências
npm run typecheck        # valida tipos (strict mode)
npm run gerar-gabaritos  # !! GASTA DINHEIRO !! - chama Claude API
npm run benchmark        # roda os 15 extratores contra os gabaritos (grátis, ~10min)
npm run inspecionar      # raio-x dos PDFs (resultados/raio-x/<nome>.json)
```

Filtros úteis:

```powershell
npm run benchmark -- --apenas "Aula 04"          # filtra PDFs por substring
npm run benchmark -- --apenas-extrator mupdf     # filtra extratores
```

## Extratores implementados

| Categoria | Extrator | Stack |
|---|---|---|
| Texto puro JS | `pdf-parse`, `pdfjs-dist`, `unpdf`, `pdf2json` | Apache/MIT |
| Texto puro WASM | `mupdf` | AGPL ⚠️ |
| OCR puro | `tesseract`, `tesseract-unpdf` | MIT/Apache |
| Híbrido (texto + OCR seletivo) | `hibrido`, `hibrido2`, `hibrido-unpdf`, `hibrido-unpdf-img`, `hibrido-unpdf-img-hash`, `hibrido-unpdf-img-hash-clean`, `hibrido-clean-best`, `hibrido-clean-conf`, `hibrido-clean-prep`, `hibrido-clean-multi`, `hibrido-clean-conf-multi`, `hibrido-clean-prep-conf`, `hibrido-clean-conf-best`, `hibrido-unpdf-bbox` | MIT |
| Subprocess Python | `docling`, `mineru` | precisam venv |
| Gabarito (gold standard) | `claude` | Anthropic API ($) |

**Vencedor MIT-only**: `hibrido-clean-conf-best` — cobertura 0.884,
Levenshtein 0.823, 10s/PDF, **0 erros nos 9 PDFs**.

## Arquitetura building blocks

A família `hibrido-clean-*` usa uma base configurável em
[src/extratores/_hibrido-img-base.ts](src/extratores/_hibrido-img-base.ts)
que aceita 4 building blocks ligáveis:

```ts
interface OpcoesHibrido {
  tessdataBest?: boolean;     // modelo LSTM melhor
  confidenceMin?: number;     // filtro confidence (0..100)
  preProcessar?: boolean;     // binarização antes do OCR
  escalasOcr?: number[];      // ex: [1, 2] roda em 2 escalas
}
```

Cada `hibrido-clean-<bloco>.ts` é um wrapper de ~10 linhas chamando
`extrairComOpcoes(caminho, { ...bloco... }, nome)`. Combinações são
triviais: criar `hibrido-clean-best-conf.ts` com `{ tessdataBest: true,
confidenceMin: 60 }`.

## Métricas

5 métricas comparando texto extraído contra gabarito, todas após
normalização (lowercase, sem acento, espaços colapsados):

- **Cobertura** — % das palavras únicas do gabarito presentes no extraído
- **Levenshtein** — similaridade caractere a caractere (1 - dist/max_len)
- **Bigramas** — Jaccard sobre pares de caracteres
- **Tamanho relativo** — len(extraido) / len(gabarito)
- **qexc (qualidade do excesso)** — % das palavras-extras que parecem pt-BR

Definições em [src/utils/metricas.ts](src/utils/metricas.ts).

## Saída do benchmark

`../resultados/benchmark.csv` — uma linha por par PDF×extrator.
`../resultados/benchmark.json` — agregados por extrator.

Ambos são compartilhados com o lab Python (mesma pasta `resultados/`).

## Estrutura

```
ts/
├── src/
│   ├── extratores/    (claude.ts, pdf-parse.ts, pdfjs.ts, unpdf.ts,
│   │                   pdf2json.ts, mupdf.ts, tesseract.ts,
│   │                   tesseract-unpdf.ts, hibrido*.ts, docling.ts,
│   │                   mineru.ts, _hibrido-img-base.ts)
│   ├── utils/         (env.ts, tipos.ts, metricas.ts)
│   ├── gerar-gabaritos.ts
│   ├── benchmark.ts
│   └── inspecionar-pdf.ts
├── package.json
├── tsconfig.json
└── README.md          ← este arquivo
```

## Cuidados

1. **`npm run gerar-gabaritos` custa dinheiro.** Chama Claude API com PDFs
   inteiros. Custo de referência: Aula 04 (31 pg) ≈ $0.20 USD.
2. **PDFs em `../dados/pdfs/` são privados.** Alguns têm copyright (livros).
3. **`.env` (na raiz do repo) tem a chave Anthropic.** Está no `.gitignore`.
4. Sempre rode `npm run typecheck` antes de dar tarefa por concluída.

## Detalhes técnicos não-óbvios

### Detector de recusa por copyright

O Claude às vezes recusa transcrever PDFs com copyright. A heurística
está em [src/extratores/claude.ts](src/extratores/claude.ts): muito
input + pouquíssimo output + `stop_reason === 'end_turn'` = provável
recusa silenciosa. Erro vem prefixado com `RECUSA:`. Chunking não tenta
dividir nesses casos.

### Chunking recursivo

PDFs grandes podem estourar `max_tokens`. Quando `stop_reason === 'max_tokens'`,
o extrator divide o PDF ao meio por página e tenta cada metade
recursivamente, até `maxProfundidade` (padrão 6) ou chunk de 1 página.

### Pegadinha do tesseract.js + langPath

Tesseract cachea `por.traineddata` na cwd. Em chamadas seguintes ignora
`langPath` se o cache existir. Pra usar `tessdata_best`, sempre passar
`cachePath` separado.

### Pegadinha do unpdf + canvas no Node

`renderPageAsImage` exige (1) `canvasImport: () => import('@napi-rs/canvas')`
explícito, (2) doc criado com `CanvasFactory` via
`createIsomorphicCanvasFactory(canvasImport)`, (3) reutilizar o
`PDFDocumentProxy` entre chamadas (passar `data` cru no loop dá
`DataCloneError` na segunda iteração).
