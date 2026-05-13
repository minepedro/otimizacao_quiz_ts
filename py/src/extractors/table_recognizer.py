"""
Reconhecedor de tabelas — converte imagem de tabela em markdown pipe.

Usa Microsoft Table Transformer (huggingface) pra detectar estrutura
de células (linhas + colunas + cabecalhos) + RapidOCR pra ler texto
de cada celula.

Modelo: microsoft/table-transformer-structure-recognition (~115MB)
Tempo: ~2-3s por tabela em CPU.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from transformers import TableTransformerForObjectDetection, AutoImageProcessor

# Singletons (lazy load)
_MODEL: "TableTransformerForObjectDetection | None" = None
_PROCESSOR: "AutoImageProcessor | None" = None
_LOCK = threading.Lock()

# Classes do modelo (mapeamento padrao)
CLASS_TABLE = 0
CLASS_COLUMN = 1
CLASS_ROW = 2
CLASS_COLUMN_HEADER = 3
CLASS_PROJECTED_ROW_HEADER = 4
CLASS_SPANNING_CELL = 5

# Confidence minimo pra aceitar deteccao de linha/coluna
DET_THRESHOLD = 0.5


def _get_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from transformers import TableTransformerForObjectDetection, AutoImageProcessor
                _PROCESSOR = AutoImageProcessor.from_pretrained(
                    "microsoft/table-transformer-structure-recognition"
                )
                _MODEL = TableTransformerForObjectDetection.from_pretrained(
                    "microsoft/table-transformer-structure-recognition"
                )
                _MODEL.eval()
    return _MODEL, _PROCESSOR


def _ocr_recorte(img_bgr: np.ndarray) -> str:
    """Le texto de um recorte usando RapidOCR. Devolve string concatenada."""
    if img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
        return ""
    from rapidocr_onnxruntime import RapidOCR
    # Reusa singleton se possivel — assumimos importado em outro lugar
    ocr = RapidOCR()
    try:
        resultado, _ = ocr(img_bgr)
    except Exception:
        return ""
    if not resultado:
        return ""
    linhas = []
    for item in resultado:
        try:
            score = float(item[2])
            if score < 0.5:
                continue
            t = (item[1] or "").strip()
            if t:
                linhas.append(t)
        except (IndexError, TypeError, ValueError):
            continue
    return " ".join(linhas)


def _detectar_estrutura(img_pil) -> dict:
    """Roda Table Transformer e devolve {rows: [...], cols: [...]}."""
    import torch
    from PIL import Image

    model, processor = _get_model()

    inputs = processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # post-process
    target_sizes = torch.tensor([img_pil.size[::-1]])  # (h, w)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=DET_THRESHOLD
    )[0]

    rows = []
    cols = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        cls = int(label.item())
        bbox = [float(b) for b in box.tolist()]
        if cls in (CLASS_ROW, CLASS_PROJECTED_ROW_HEADER):
            rows.append({"bbox": bbox, "score": float(score), "is_header": cls == CLASS_PROJECTED_ROW_HEADER})
        elif cls in (CLASS_COLUMN, CLASS_COLUMN_HEADER):
            cols.append({"bbox": bbox, "score": float(score), "is_header": cls == CLASS_COLUMN_HEADER})

    # Ordena rows por Y (top-down), cols por X (left-right)
    rows.sort(key=lambda r: r["bbox"][1])
    cols.sort(key=lambda c: c["bbox"][0])

    return {"rows": rows, "cols": cols}


def reconhecer_tabela(img_bgr: np.ndarray) -> str:
    """Recebe imagem da tabela em BGR (numpy), devolve markdown pipe table.

    Se a deteccao de estrutura falhar, devolve string vazia (caller
    pode fazer fallback pra OCR cru).
    """
    try:
        from PIL import Image
        import cv2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
    except Exception:
        return ""

    try:
        estrutura = _detectar_estrutura(img_pil)
    except Exception:
        return ""

    rows = estrutura["rows"]
    cols = estrutura["cols"]

    if not rows or not cols:
        return ""

    # Pra cada (linha, coluna): recorta intersecao + OCR
    grid = []
    for ri, row in enumerate(rows):
        linha_celulas = []
        for ci, col in enumerate(cols):
            # Bbox da celula = intersect(row, col)
            x1 = max(row["bbox"][0], col["bbox"][0])
            y1 = max(row["bbox"][1], col["bbox"][1])
            x2 = min(row["bbox"][2], col["bbox"][2])
            y2 = min(row["bbox"][3], col["bbox"][3])
            if x2 <= x1 or y2 <= y1:
                linha_celulas.append("")
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_bgr.shape[1], x2); y2 = min(img_bgr.shape[0], y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                linha_celulas.append("")
                continue
            recorte_celula = img_bgr[y1:y2, x1:x2].copy()
            texto = _ocr_recorte(recorte_celula).strip().replace("|", "\\|")
            linha_celulas.append(texto)
        grid.append(linha_celulas)

    # Gera markdown pipe table
    if not grid or not grid[0]:
        return ""
    n_cols = len(grid[0])
    linhas_md = []
    for i, linha in enumerate(grid):
        # Garante n_cols celulas (pad com vazio)
        celulas = list(linha) + [""] * (n_cols - len(linha))
        celulas = celulas[:n_cols]
        linhas_md.append("| " + " | ".join(celulas) + " |")
        if i == 0:
            linhas_md.append("|" + " --- |" * n_cols)

    return "\n".join(linhas_md)
