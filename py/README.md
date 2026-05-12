# Lab Python — pipeline de extração de PDF "best-in-class"

Subpasta Python do repo `otimizacao_quiz_ts`. **Em construção** —
objetivo é alcançar o teto técnico de qualidade usando modelos pesados
(PyMuPDF, PaddleOCR, layout detection, GPU CUDA).

> Veja o [README raiz](../README.md) pra entender a arquitetura dual
> TS+Python do repo.

## Status

⏳ **Sprint 1 (atual):** setup do esqueleto.
⏳ Sprint 2: gabarito v2 (markdown + JSON estruturado).
⏳ Sprint 3: pipeline MVP com PyMuPDF + PaddleOCR.
⏳ Sprint 4: layout detection + reading order.
⏳ Sprint 5: comparação final 4 stacks (TS, Python, Docling, MinerU).
⏳ Sprint 6: hyperparameter search (CPU + GPU).
⏳ Sprint 7 (opcional): API HTTP pro tutor-ai.

## Stack alvo

- **Python 3.10-3.12** (Windows não suporta 3.13 — dependência `ray` do MinerU)
- **PyMuPDF (fitz)** — extração base de texto vetorial + bbox + imagens
- **PaddleOCR** — OCR (rápido, multi-idioma incluindo PT, suporta GPU)
- **Pillow + opencv** — pré-processamento de imagem
- **Pydantic** — validação de schemas (`BlocoExtraido`)
- **Anthropic SDK** — geração de gabarito v2
- **PyTorch (CUDA 12.8 pra Blackwell/RTX 50)** — backbone ML

Lista completa em [requirements.txt](requirements.txt).

## Setup

```powershell
cd py
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/pip install -r requirements.txt

# Pra GPU (RTX 50/40/30 com CUDA 12.8):
.venv/Scripts/pip install --upgrade --force-reinstall \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verificar GPU:

```powershell
.venv/Scripts/python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Arquitetura planejada

```
PDF
 ↓
[1] PyMuPDF: texto vetorial + lista de imagens + bbox + fontes
 ↓
[2] Hash-based image dedup (port da heurística TS de background detection)
 ↓
[3] OCR seletivo via PaddleOCR (apenas imagens não-background grandes)
 ↓
[4] Layout detection (DocLayout-YOLO) → classificação por região
 ↓
[5] Reading order (XY-cut ou modelo)
 ↓
[6] Header/footer detection (frases repetidas entre páginas)
 ↓
[7] Output: BlocoExtraido[] + markdown estruturado + texto puro
```

Ver [src/schemas.py](src/schemas.py) pro schema do `BlocoExtraido`
(equivalente Python do tipo TS, enriquecido com `bbox`, `parent_id`,
`block_id`, `ocr_confidence` pra rastreabilidade RAG).

## Métricas

Mesmas 5 do lab TS (port direto de
[../ts/src/utils/metricas.ts](../ts/src/utils/metricas.ts)):

- Cobertura, Levenshtein, Bigramas, Tamanho relativo, qexc

Comparação cross-stack (TS vs Python) usa mesmo CSV
([../resultados/benchmark.csv](../resultados/benchmark.csv)).

## Decisões importantes

- **Não embarca direto no Electron** — Python+modelos pesados é incompatível.
  Pipeline vai virar **API HTTP** servida do servidor (Sprint 7).
- **Escolha de Python sobre TS:** ecossistema ML (PaddleOCR, Surya,
  transformers, torch), suporte GPU first-class, mesmos modelos que MinerU/Docling.
- **Tradeoff aceito:** complexidade de empacotamento vs qualidade técnica máxima.

## Pegadinhas conhecidas (ordem de descoberta)

### 1. Python 3.13 + Windows não funciona pro MinerU

Dependência `ray` do MinerU não suporta 3.13 no Windows. Sintoma:
trava sem output. Use 3.10, 3.11 ou 3.12.

### 2. PyTorch CPU vs CUDA — pip não troca automaticamente

Se MinerU/Docling instalou `torch` CPU como dep, depois `pip install
torch --index-url cu128` não substitui (vê que já existe). Forçar:

```powershell
.venv/Scripts/pip install --upgrade --force-reinstall \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. MinerU usa ModelScope (servidor chinês) por padrão

Pode travar no download. Solução: rodar `mineru-models-download -s
huggingface -m pipeline` antes do primeiro uso (baixa do HuggingFace).

### 4. Docling com `docling-parse` quebra em alguns PDFs

"Inconsistent number of pages" em alguns PDFs. Solução: usar backend
`pypdfium2` (`docling --pdf-backend pypdfium2 ...`).

## Estrutura

```
py/
├── src/
│   ├── extractors/    (em construção: pymupdf_base.py, paddleocr_engine.py, layout_detector.py)
│   ├── pipeline/      (em construção: orchestrator.py, block_classifier.py, header_footer_detector.py)
│   ├── schemas.py     ← Pydantic models (BlocoExtraido, GabaritoV2, ResultadoExtracao)
│   ├── metrics.py     (em construção)
│   ├── benchmark.py   (em construção)
│   ├── gabarito_v2.py (em construção)
│   └── hyperparam_search.py (em construção)
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md          ← este arquivo
```
