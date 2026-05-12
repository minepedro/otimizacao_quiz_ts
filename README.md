# otimizacao_quiz_ts — laboratório de extração de PDF

Lab público pra encontrar o **melhor extrator de PDF** alimentando o
app `tutor-ai` (gerador de quiz a partir de material de estudo).

Estratégia: testar **muitos extratores** contra o mesmo conjunto de PDFs,
medir cobertura/Levenshtein/bigramas/qualidade-do-excesso/tamanho-relativo
contra um gabarito gerado pela Claude API.

## Arquitetura dual

O repo abriga **dois laboratórios paralelos**:

```
otimizacao_quiz_ts/
├── ts/        ← lab TS — 15+ extratores, 5 métricas, vencedor: hibrido-clean-conf-best
├── py/        ← lab Python (em construção) — modelos pesados, GPU, melhor qualidade
├── dados/     ← PDFs + gabaritos (compartilhado)
└── resultados/← CSVs de benchmark + raio-x dos PDFs (compartilhado)
```

A pasta **`ts/`** já está madura e contém o melhor extrator MIT-only
descoberto até agora: `hibrido-clean-conf-best`. Roda em qualquer máquina
(até notebook fraco), 10s/PDF, cobertura média 0.884 contra o gabarito v1.

A pasta **`py/`** é a próxima fase: pipeline com modelos pesados
(PyMuPDF, PaddleOCR, layout detection, GPU CUDA) pra alcançar o teto
técnico de qualidade — meta cobertura ≥ 0.93 em 5s/PDF na RTX 5070.

## Status do projeto

| Fase | Status | Resultado |
|---|---|---|
| TS lab | ✅ Concluído | 15 extratores, vencedor `hibrido-clean-conf-best` |
| Comparação Docling/MinerU | ✅ Concluído | TS vence em 7/9 PDFs |
| Gabarito v2 (markdown+JSON) | ⏳ Sprint 2 | — |
| Pipeline Python MVP | ⏳ Sprint 3 | — |
| Layout detection + reading order | ⏳ Sprint 4 | — |
| Comparação final 4 stacks | ⏳ Sprint 5 | — |
| Hyperparameter search | ⏳ Sprint 6 | — |
| API HTTP pro tutor-ai | ⏳ Sprint 7 (opcional) | — |

## Quickstart

### Lab TS

```powershell
cd ts
npm install
npm run typecheck
npm run benchmark              # roda os 15 extratores nos 9 PDFs
npm run inspecionar             # raio-x dos PDFs
```

Detalhes em [ts/README.md](ts/README.md).

### Lab Python

```powershell
cd py
py -3.12 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
# (próximos passos ainda em construção — Sprint 3+)
```

Detalhes em [py/README.md](py/README.md).

## Métricas usadas

5 métricas, todas comparando texto extraído vs gabarito (Claude API):

- **Cobertura** — % das palavras únicas do gabarito que aparecem no extraído
- **Levenshtein** — `1 - dist_edicao / max(len)`, medida caractere a caractere
- **Bigramas** — Jaccard sobre pares de caracteres, robusto a reordenação
- **Tamanho relativo** — `len(extraido) / len(gabarito)`, 1.0 = perfeito
- **qualidade do excesso (qexc)** — % das palavras "extras" que parecem palavras válidas em pt-BR (não ruído OCR)

Definições e implementação: [ts/src/utils/metricas.ts](ts/src/utils/metricas.ts).

## Conjunto de PDFs

Listado em [dados/manifesto.json](dados/manifesto.json) — 9 PDFs cobrindo:

- Slides acadêmicos (Aula 04 Redes, Aula 03 Probabilidade, Aula 2 Economia)
- Slides com imagens/screenshots (EPM 309 STP, PokaYoke, 5S)
- Manuscrito matemático (Aula 04 Resolução — caso limite)
- Apostilas (Manual Pesquisa, Realismo)

Os PDFs em si não vão pro git (privados/copyright). O manifesto descreve cada um.

## Decisões de produto importantes

- **Pipeline TS** roda no client (`tutor-ai` é Electron) — distribuição zero-config pro usuário final.
- **Pipeline Python** vai virar API HTTP servida do servidor — pra casos de qualidade premium.
- **Claude API** continua como "extração premium" manual quando o usuário escolhe (PDFs muito difíceis).

## Licença

MIT — código + métricas + scripts. PDFs e gabaritos não são distribuídos.

## Roadmap completo

Veja o plano detalhado por sprint em [.claude/plans/](.claude/plans/) (gitignored — pra desenvolvedores).
