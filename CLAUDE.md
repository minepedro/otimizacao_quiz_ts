# CLAUDE.md — instruções pro Claude Code neste projeto

> Este arquivo é lido automaticamente pelo Claude Code toda vez que ele abre
> esse diretório. Serve pra alinhar contexto, convenções e cuidados.

## O que é este projeto

Laboratório público de extratores de PDF, alimentando o app `tutor-ai`
(Electron + TS) do Pedro. Mede vários extratores contra um "gabarito"
(gold standard) gerado pela Claude API.

**Arquitetura dual** (a partir de 2026-05-12):

```
otimizacao_quiz_ts/      ← raiz do repo (.env, .gitignore, README aqui)
├── ts/                  ← lab TS (15+ extratores) — MADURO
│   ├── src/             ← código TS
│   ├── package.json
│   ├── tsconfig.json
│   └── README.md        ← especifico do TS lab
├── py/                  ← lab Python (modelos pesados, GPU) — EM CONSTRUCAO
│   ├── src/             ← código Python
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── README.md        ← especifico do Python lab
├── dados/               ← compartilhado (manifesto.json, pdfs, gabaritos)
├── resultados/          ← compartilhado (benchmark.csv, raio-x)
├── README.md            ← README raiz (descreve arquitetura dual)
└── CLAUDE.md            ← este arquivo
```

**Importante pro Claude Code:**
- Pra trabalhar no lab TS, `cd ts` antes de rodar `npm` ou ler arquivos `src/`.
- Pra trabalhar no lab Python, `cd py` antes de ativar venv ou rodar scripts.
- `dados/` e `resultados/` ficam na **raiz** do repo (compartilhados).
- `.env` (chave Anthropic) está na **raiz** do repo.

## Estado atual (Sprint 1 do plano de migração)

- **Lab TS:** 15+ extratores, vencedor `hibrido-clean-conf-best`
  (cob 0.884, lev 0.823, 10s/PDF, 0/9 erros).
- **Lab Python:** esqueleto + schemas Pydantic (`BlocoExtraido`,
  `GabaritoV2`). Implementação propriamente dita começa no Sprint 3.
- **Repo:** `git init` feito, será publicado no GitHub no fim do Sprint 1.

Veja [README raiz](README.md) pro status completo dos sprints e
[ts/README.md](ts/README.md) / [py/README.md](py/README.md) pros
detalhes específicos.

**Vencedor TS atual (concorrente do futuro Python):** `hibrido-clean-conf-best`
(MIT, cobertura média **0.884** nos 9 PDFs, Levenshtein **0.823**, bigramas
**0.812** — melhor em 3 das 5 métricas). Tempo médio 10.7s/PDF.

## O conjunto de PDFs

A "fonte da verdade" sobre os PDFs do lab é [dados/manifesto.json](dados/manifesto.json),
que lista cada arquivo em `dados/pdfs/` com tipo, area, se tem imagens, se
tem copyright e notas relevantes (ex: achados sobre extracao). **Antes de
analisar resultados, leia esse arquivo** — ele explica o conjunto sem
precisar abrir cada PDF.

### Fluxo pra adicionar PDF novo

1. Copia o PDF pra `dados/pdfs/<Nome do PDF>.pdf`. **Nome do arquivo é a
   chave** — manter estável (renomear depois quebra associação com gabarito).
2. Adiciona uma entrada em `dados/manifesto.json` com tipo, area e notas.
3. (Opcional) `npm run gerar-gabaritos -- --apenas "<substring>"` — gera
   o gabarito (custa $).
4. (Opcional) `npm run inspecionar -- --apenas "<substring>"` — gera o
   raio-x do PDF.
5. (Opcional) `npm run benchmark -- --apenas "<substring>"` — roda os
   extratores no PDF novo.

## Cuidados importantes

1. **`npm run gerar-gabaritos` custa dinheiro.** Chama Claude API com PDFs
   inteiros. Nunca rode sem o Pedro pedir explicitamente. Custo de
   referência: Aula 04 (31 pg) ≈ $0.20 USD.
2. **PDFs em `dados/pdfs/` são privados.** Alguns têm copyright (livros).
   Não inclua trechos em commits, issues, nem em respostas pro usuário sem
   ele pedir.
3. **`.env` tem a chave da Anthropic.** Está no `.gitignore`. Nunca leia
   ela em voz alta nem cole em lugar nenhum.
4. **`.claude/` está no `.gitignore`.** Settings locais não vão pro
   repositório — não tem problema escrever ali.

## Stack e convenções

- **Node 20+** (testado em 24.x), **TypeScript strict**.
- Roda direto via **`tsx`** — não tem build step. Não crie `dist/`.
- **ESM** (`"type": "module"` no package.json). Imports de arquivos `.ts`
  locais usam extensão **`.js`** por causa do NodeNext (ex:
  `import { foo } from './utils/env.js'` — note o `.js` mesmo o arquivo
  sendo `.ts`).
- **Identificadores em português** no código do projeto: `extrator`,
  `gabarito`, `chunks`, `recusaDetectada`, `carregarChaveAnthropic`. Mantenha
  o padrão. Comentários e mensagens de log também em português.
- **Sem acentos em strings de log e mensagens** (legado do lab Python pra
  evitar problemas no console Windows). Veja os logs existentes — segue o
  estilo "Extracao recursiva", "ja existe", etc.
- **Scripts CLI são idempotentes:** `--apenas <substring>` filtra, `--forcar`
  ignora cache/skip. Mantenha esse padrão em scripts novos.
- Tipos compartilhados ficam em `src/utils/tipos.ts`. Extrator novo deve
  retornar `ResultadoExtracao` ou estender ele.
- Carregamento da chave Anthropic: sempre via
  `carregarChaveAnthropic()` de `src/utils/env.ts`, nunca lendo `process.env`
  direto.

## Estrutura

```
otimizacao_quiz_ts/
├── src/
│   ├── extratores/      (claude.ts, pdf-parse.ts, pdfjs.ts, unpdf.ts, pdf2json.ts, mupdf.ts, tesseract.ts, tesseract-unpdf.ts, hibrido.ts, hibrido2.ts, hibrido-unpdf.ts)
│   ├── utils/           (env.ts, tipos.ts, metricas.ts)
│   ├── gerar-gabaritos.ts
│   ├── benchmark.ts
│   └── inspecionar-pdf.ts
├── dados/
│   ├── manifesto.json   (metadados dos PDFs — fonte da verdade)
│   ├── pdfs/            (PDFs de entrada, gitignored)
│   └── gabaritos/       (.txt gerados pelo Claude, gitignored)
├── resultados/
│   ├── benchmark.csv    (uma linha por par PDF×extrator)
│   ├── benchmark.json   (agregados por extrator)
│   ├── raio-x/          (.json por PDF — saída do inspecionar)
│   └── gabaritos_metadata.json
├── .env                 (chave Anthropic, gitignored)
└── CLAUDE.md            (este arquivo)
```

## Extratores implementados

| Extrator | Lib | Categoria | Notas |
|---|---|---|---|
| `claude` | `@anthropic-ai/sdk` | Gabarito (pago) | Gold standard. Não conta no benchmark. |
| `pdf-parse` | `pdf-parse` | Texto puro JS | Mais simples, baseline. |
| `pdfjs-dist` | `pdfjs-dist` | Texto puro JS | PDF.js direto. Usa posições x/y pra ordenar (suporte a colunas). |
| `unpdf` | `unpdf` | Texto puro JS | Wrapper moderno do PDF.js, otimizado pra serverless. |
| `pdf2json` | `pdf2json` | Texto puro JS | Parser baseado em eventos, retorna JSON estruturado. |
| `mupdf` | `mupdf` | Texto puro WASM | Port do MuPDF (motor do PyMuPDF). **Licença AGPL-3.0** — avaliar antes de embarcar no tutor-ai. |
| `tesseract` | `tesseract.js` + `mupdf` | OCR puro | Renderiza cada página em PNG (mupdf, 2x scale) e roda OCR (tesseract.js, por+eng). Mantido pra comparação de render. **Licença AGPL via mupdf.** |
| `tesseract-unpdf` | `tesseract.js` + `unpdf` + `@napi-rs/canvas` | OCR puro 100% MIT | Igual ao `tesseract`, mas renderiza com `unpdf.renderPageAsImage` (que usa pdf.js + napi-rs/canvas). ~10% mais lento que mupdf-render. Métricas equivalentes. **Esta é a opção recomendada pro tutor-ai** — sem dependência AGPL. Cuidado técnico: pdf.js no Node "transfere" o buffer entre chamadas; sempre passar o `PDFDocumentProxy` pro `renderPageAsImage`, não o `data` cru, e criar o doc com `CanvasFactory` explícita via `createIsomorphicCanvasFactory`. |
| `hibrido` | `mupdf` + `tesseract.js` | Texto + OCR seletivo (v1) | mupdf pra texto; OCR só nas páginas com `< 100` chars extraídos. **Limitação:** páginas com screenshot grande + texto em volta (Aula 2) não acionam OCR. Mantido pra comparação histórica. |
| `hibrido2` | `mupdf` + `tesseract.js` | Texto + OCR seletivo (v2, mupdf) | Dispara OCR quando `area_imagem >= 30%` da página **E** `chars_extraidos < 500`. Quando dispara, **substitui** o texto da página pelo OCR (não soma — evita duplicação). Recupera os ~22% perdidos da Aula 2 e fica equivalente ao tesseract puro com ~3× menos tempo. **Vencedor do benchmark mas AGPL via mupdf.** |
| `hibrido-unpdf` | `unpdf` + `extractImages` + `tesseract.js` | Texto + OCR seletivo MIT-only | Versão 100% MIT do hibrido2. Como `unpdf` não dá bbox de blocos de imagem na página, usa heurística baseada em tamanho de imagem (mais frequente = background) + chars vetoriais. Bom mas superado por `img` e `img-hash`. |
| `hibrido-unpdf-img` | `unpdf` + `extractImages` + `tesseract.js` + `@napi-rs/canvas` | Texto + OCR direto na imagem | Em vez de renderizar a página, pega os bytes da imagem direto via `extractImages`, converte raw RGB → PNG via canvas, OCR na imagem. **Soma com texto vetorial** (não substitui). Mais rápido que renderizar a página. Bug conhecido: detecção de background por tamanho falha quando todas as páginas têm imagens do mesmo tamanho mas conteúdo diferente (ex: PDF scan). |
| `hibrido-unpdf-img-hash` | igual + `crypto` | Versão refinada do `img` | (1) Detecta background por **SHA-256 dos bytes** (não por tamanho) — resolve o caso scan onde cada página tem imagem única do mesmo tamanho. (2) **Cache de OCR por hash** — não OCR-a a mesma imagem 2× (útil pra screenshots reusados). Bom mas superado por `img-hash-clean`. |
| `hibrido-unpdf-img-hash-clean` | igual + dedup textual | **Vencedor atual.** | Adiciona ao `img-hash` uma etapa de **deduplicação OCR vs vetorial**: pra cada frase do OCR, se >= 60% das palavras já estão no texto vetorial daquela página, descarta a frase (provável duplicação). Cobertura 0.895 (igual ao img-hash) + Levenshtein 0.785 (melhor) + bigramas 0.734 (melhor) + tam 1.09 (menos inflado). Custa ~0.3s extra de processamento. **Recomendação atual pro tutor-ai.** |
| `hibrido-clean-best` / `-conf` / `-prep` / `-multi` | `_hibrido-img-base.ts` (configurável) | Building blocks isolados | Cada um liga 1 dos 4 building blocks: **best** usa modelo LSTM tessdata_best; **conf** filtra palavras OCR com confidence < 60; **prep** binariza imagem antes do OCR; **multi** roda OCR em 2 escalas e mescla. São wrappers de 5-10 linhas chamando `extrairComOpcoes(caminho, options, nome)` da base. |
| `hibrido-unpdf-bbox` | `unpdf` + `pdfjs-dist` (`getOperatorList`/`OPS`) + `tesseract.js` | Texto + OCR em região da página | Renderiza página inteira, descobre posição das imagens via `getOperatorList` rastreando CTM (current transformation matrix), passa `rectangle` pro tesseract. Boa cobertura mas tende a inflar tamanho (não dedupliica imagens repetidas na mesma página). |

Cada extrator em `ts/src/extratores/<nome>.ts` exporta `NOME_EXTRATOR` e
`extrair(caminhoPdf): Promise<ResultadoExtracao>`. Pra adicionar um
extrator novo: criar arquivo seguindo esse contrato e adicionar no array
`EXTRATORES` em [ts/src/benchmark.ts](ts/src/benchmark.ts).

**Nota sobre caminhos no resto deste arquivo:** após a reorganização dual
TS+Python (Sprint 1 do plano de migração), todos os caminhos `src/...`
referenciados abaixo agora estão em `ts/src/...`. Comandos `npm run *`
devem ser executados de dentro de `ts/`.

### Arquitetura building blocks (família clean-*)

Pra experimentar refinamentos sem duplicar código, a partir de 2026-05-12
existe [src/extratores/_hibrido-img-base.ts](src/extratores/_hibrido-img-base.ts)
— uma **base configurável** que aceita 4 opções (building blocks ligáveis):

```ts
interface OpcoesHibrido {
  tessdataBest?: boolean;     // modelo LSTM melhor
  confidenceMin?: number;     // filtro confidence (0..100)
  preProcessar?: boolean;     // binarização antes do OCR
  escalasOcr?: number[];      // ex: [1, 2] roda em 2 escalas
}
```

Cada `hibrido-clean-<bloco>.ts` é um wrapper de ~10 linhas chamando
`extrairComOpcoes(caminho, { ...bloco... }, nome)`. Pra testar
**combinações**, basta criar `hibrido-clean-best-conf.ts` que passa
`{ tessdataBest: true, confidenceMin: 60 }`. Sem copiar lógica.

O `_` no nome do arquivo indica que é base (não-extrator). Não tem
`NOME_EXTRATOR`/`extrair` exportados como esperado pelo benchmark.

## Métricas do benchmark

Definidas em [src/utils/metricas.ts](src/utils/metricas.ts). Comparam o
texto extraído contra o gabarito, todas após normalização (lowercase, sem
acento, espaços colapsados):

- `sim_levenshtein`: `1 - dist/max(len)`. 1.0 = idêntico. Punitivo com qualquer diferença.
- `cobertura_palavras`: % das palavras únicas do gabarito que aparecem no extraído. Ignora ordem e duplicatas.
- `sim_bigramas`: Jaccard sobre bigramas de caractere. Robusto a reordenação.
- `tamanho_relativo`: `len(extraido) / len(gabarito)`. 1.0 = mesmo tamanho; <1 = truncou; >1 = ruído extra.

## Saída do benchmark

`resultados/benchmark.csv` — uma linha por par PDF×extrator. Colunas:
`pdf, extrator, num_paginas, tempo_seg, num_caracteres, sim_levenshtein, cobertura_palavras, sim_bigramas, tamanho_relativo, erro`.

`resultados/benchmark.json` — agregados (médias por extrator entre os PDFs sem erro).

## Comandos úteis

```powershell
npm install              # instala dependências
npm run typecheck        # roda tsc --noEmit, valida tipos sem gerar arquivos
npm run gerar-gabaritos  # !! GASTA DINHEIRO !! - chama Claude API
npm run benchmark        # roda os 7 extratores locais contra os gabaritos (gratis, ~1min)
npm run inspecionar      # raio-x dos PDFs (resultados/raio-x/<nome>.json)
```

Filtros úteis (já implementados em `gerar-gabaritos.ts`, padrão pra novos
scripts):

```powershell
npm run gerar-gabaritos -- --apenas "Aula 04"   # só o que casa com "Aula 04"
npm run gerar-gabaritos -- --forcar             # ignora idempotência

npm run benchmark -- --apenas "Aula 04"         # mesmo filtro no benchmark
npm run benchmark -- --apenas-extrator mupdf    # roda só um extrator
```

**Merge do benchmark.csv:** quando rodado com `--apenas` ou `--apenas-extrator`,
o `benchmark.ts` PRESERVA as linhas existentes do CSV pra pares (PDF, extrator)
que NÃO foram refeitos nessa execução. Sem filtro, sobrescreve tudo. Permite
testar/refazer só um extrator novo sem perder o resultado dos outros (e
sem precisar re-rodar o benchmark inteiro de 15 min).

## Antes de finalizar qualquer mudança em código

Sempre rode `npm run typecheck` antes de dar tarefa por concluída. O
projeto é strict, e o tipo `ResultadoExtracao` é o contrato comum entre
extratores — quebrar ele quebra o benchmark futuro.

## Detalhes técnicos que não dá pra deduzir do código

### Detector de recusa por copyright

O Claude às vezes recusa transcrever PDFs com copyright (livros). A
heurística está em [src/extratores/claude.ts](src/extratores/claude.ts):
muito input + pouquíssimo output + `stop_reason === 'end_turn'` = provável
recusa silenciosa. Quando isso acontece:

- O erro vem prefixado com `RECUSA:` (constante `PREFIXO_ERRO_RECUSA`).
- O chunking **não tenta dividir** — recusa não se resolve fatiando.
- O metadata em `resultados/gabaritos_metadata.json` registra
  `recusaDetectada: true`, e o `gerar-gabaritos.ts` pula esse PDF nas
  próximas execuções (até o `--forcar`).

Mantenha esse comportamento em qualquer refatoração do extrator Claude.

### Chunking recursivo

PDFs grandes podem estourar `max_tokens` na resposta. Quando o
`stop_reason === 'max_tokens'`, o extrator divide o PDF ao meio por página
e tenta cada metade recursivamente, até `maxProfundidade` (padrão 6) ou
até chunk de 1 página. Texto resultante é concatenado com `\n\n`.

### Preços hardcoded

Os preços do Claude Sonnet 4.6 estão no topo de
[src/extratores/claude.ts](src/extratores/claude.ts) como constantes. Se
trocar de modelo, atualize as duas constantes (`INPUT_COST_PER_MTOKEN` e
`OUTPUT_COST_PER_MTOKEN`) e o nome em `MODELO`.

## Onde paramos (snapshot — atualizar quando voltar)

**Última sessão:** benchmark completo nos 4 PDFs do conjunto. Vencedor pro
tutor-ai é `hibrido-unpdf` (100% MIT). Cobertura ~0.99 nos 3 PDFs
"normais" (slides com ou sem screenshot). Cobertura **0.18** no PDF
recém-adicionado `Aula+04+%28resolu%C3%A7%C3%A3o%29.pdf` — caso limite.

### A descoberta da última sessão

PDFs **manuscritos com fórmulas matemáticas** (resoluções de exercícios
de probabilidade no caso) são **fronteira do tesseract.js**. Não é falha
da heurística do `hibrido-unpdf` — ela detectou corretamente que precisa
OCR. O motor (tesseract.js) é o gargalo: não foi treinado pra
- símbolos matemáticos (∩, ∪, ², =ind=)
- diagramas de árvore
- subscritos/sobrescritos
- manuscrito

Verificação: o `tesseract` puro deu o mesmo resultado (0.194) — sintoma
de limite do motor, não da heurística.

### Decisão tomada (2026-05-12)

**Como tratar PDFs manuscritos/com fórmulas no tutor-ai:** escolha
**manual** do usuário, não automática.

- **Padrão (grátis):** `hibrido-unpdf` (vencedor MIT do lab).
- **Botão "extração premium" (pago):** `claude.ts` — Claude API (~$0.05/PDF).

Pedro descartou (por enquanto) Mathpix, Google Vision e detector
automático de qualidade. Razão: usuário sabe melhor que algoritmo se
vale o custo extra; evita falsos positivos; dá controle e transparência.

**Implicação pro lab:** não precisa criar detector de qualidade,
extrator cascata, nem integrar OCRs cloud. O lab já tem tudo: hibrido-unpdf
é o caminho grátis, claude.ts é o caminho premium. Trabalho de UI/cobrança
fica no tutor-ai.

### Ideia futura (não implementar ainda)

Combinar **enunciado + resolução** num único request pra Claude API
quando o usuário tem o par. Resolveria casos onde a resolução
isolada fica confusa (referencia o enunciado). Claude API aceita
múltiplos PDFs no mesmo request. Veria evolução do lab com:
"par de PDFs" no manifesto + extrator `claude-contextualizado.ts`.

### O que NÃO precisa fazer ao retomar

- Não precisa re-rodar gerar-gabaritos (todos os 4 PDFs já têm gabarito,
  custo total acumulado: $0.25).
- Não precisa re-instalar nada (`@napi-rs/canvas`, `mupdf`,
  `tesseract.js`, `unpdf`, `fastest-levenshtein` — tudo no
  package-lock).
- Não precisa re-implementar nada — todos os 9 extratores funcionam.

### O que está em [resultados/](resultados/)

- `benchmark.csv` / `benchmark.json` — última rodada nos 4 PDFs.
- `raio-x/<pdf>.json` — ficha técnica de cada PDF.
- `gabaritos_metadata.json` — custo e tempo dos gabaritos gerados.
