"""
Wrapper do DocLayout-YOLO pra deteccao de layout em paginas de PDF.

Mesmo modelo usado pelo MinerU (DocLayout-YOLO-DocStructBench).
Detecta 10 tipos de regiao por pagina:

  0  title              -> mapeia pra TipoBloco "title"
  1  plain text         -> mapeia pra "paragraph"
  2  abandon            -> mapeia pra "header"/"footer" (cabecalho/rodape)
  3  figure             -> mapeia pra "image" (CANDIDATA pra OCR)
  4  figure_caption     -> "caption"
  5  table              -> "table"
  6  table_caption      -> "caption"
  7  table_footnote     -> "footnote" (mapeia pra "caption" no nosso schema)
  8  isolate_formula    -> "formula"
  9  formula_caption    -> "caption"

Modelo eh baixado automaticamente do HuggingFace (~25MB) na primeira chamada
e cacheado em ~/.cache/huggingface/hub/.

Inferencia: ~1-2s por pagina em CPU. GPU acelera ~5x se torch detectar CUDA.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from doclayout_yolo import YOLOv10

# Modelo HuggingFace
HF_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
HF_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
IMG_SIZE = 1024  # Tamanho usado pelo modelo (deve casar)
CONF_MIN = 0.20  # Confidence minima pra aceitar deteccao

LayoutClass = Literal[
    "title",
    "plain_text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
]

# IDs do modelo -> nomes legiveis (espelha r.names que o YOLO devolve)
CLASSE_POR_ID: dict[int, LayoutClass] = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}


@dataclass
class RegiaoLayout:
    """Uma regiao detectada pelo modelo de layout em UMA pagina."""

    classe: LayoutClass
    confidence: float
    # bbox em pixels da imagem renderizada [x1, y1, x2, y2]
    bbox: tuple[float, float, float, float]
    # Largura/altura da imagem original (pra normalizar coords se precisar)
    img_width: int
    img_height: int


_MODEL: "YOLOv10 | None" = None


def _get_model() -> "YOLOv10":
    """Singleton do modelo YOLO. Baixa do HF na primeira chamada."""
    global _MODEL
    if _MODEL is None:
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
        _MODEL = YOLOv10(path)
    return _MODEL


def detectar(img_bgr: np.ndarray, conf_min: float = CONF_MIN) -> list[RegiaoLayout]:
    """Detecta regioes de layout em UMA imagem (renderizacao de UMA pagina).

    Args:
        img_bgr: imagem BGR np.ndarray (de PyMuPDF + OpenCV)
        conf_min: confidence minima pra aceitar (0..1)

    Returns:
        Lista de regioes ordenadas por (top, left) pra reading order
        natural top-down, left-right.
    """
    model = _get_model()
    h, w = img_bgr.shape[:2]
    results = model.predict(img_bgr, imgsz=IMG_SIZE, conf=conf_min, device="cpu", verbose=False)
    if not results:
        return []
    r = results[0]
    regioes: list[RegiaoLayout] = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        classe = CLASSE_POR_ID.get(cls_id)
        if classe is None:
            continue
        regioes.append(
            RegiaoLayout(
                classe=classe,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                img_width=w,
                img_height=h,
            )
        )
    # Reading order natural: top-down, left-right
    regioes.sort(key=lambda r: (round(r.bbox[1] / 50) * 50, r.bbox[0]))
    return regioes


def warmup() -> float:
    """Forca download do modelo. Retorna tempo gasto (segundos)."""
    t0 = time.perf_counter()
    _get_model()
    return time.perf_counter() - t0


# Mapeamento das 10 classes do DocLayout-YOLO pros 15 TipoBloco do nosso schema
def classe_para_tipo_bloco(classe: LayoutClass) -> str:
    """Mapeia classe do DocLayout pra TipoBloco do BlocoExtraido."""
    if classe == "title":
        return "title"
    if classe == "plain_text":
        return "paragraph"
    if classe == "abandon":
        # abandon = header/footer/numero de pagina. Sem mais info pra
        # diferenciar — vai como header (downstream pode olhar y do bbox).
        return "header"
    if classe == "figure":
        return "image"
    if classe in ("figure_caption", "table_caption", "table_footnote", "formula_caption"):
        return "caption"
    if classe == "table":
        return "table"
    if classe == "isolate_formula":
        return "formula"
    return "paragraph"  # fallback
