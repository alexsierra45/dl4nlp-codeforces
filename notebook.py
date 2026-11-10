# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     gpuType: T4
#     provenance: []
#   jupytext:
#     notebook_metadata_filter: accelerator,colab
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="88ef5c74"
# # Predicción de etiquetas algorítmicas y dificultad de problemas de Codeforces
# ## Ajuste fino multitarea de ModernBERT
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
# https://colab.research.google.com/github/alexsierra45/dl4nlp-codeforces/blob/main/Sierra_Alcala_Alex.ipynb)
#
# **Instrucciones de ejecución**
#
# 1. Abrir el notebook en Google Colab y seleccionar *Entorno de ejecución → Cambiar tipo de
#    entorno de ejecución → GPU T4*.
# 2. (Opcional) Añadir el token de Hugging Face como secreto de Colab con el nombre `HF_TOKEN`
#    (icono de la llave de la barra lateral). Sin token el notebook funciona igual, pero no sube nada al Hub.
# 3. Ejecutar *Entorno de ejecución → Ejecutar todo*.
#
# Dos *flags* de la celda de configuración controlan la ejecución:
#
# | Flag | Efecto |
# |---|---|
# | `RUN_TRAINING` | `True`: entrena el modelo. `False`: lo descarga ya ajustado desde el Hub. |
# | `PUSH_TO_HUB` | Sube el modelo y el dataset procesado al Hub (requiere token). |
#
# El modelo se sube al Hub al final de cada época, de modo que una desconexión de Colab no obliga a
# repetir todo el entrenamiento.
#
# Tiempo aproximado en una T4: el ajuste fino tarda unos 35 minutos
# (35,5 min en la ejecución de este notebook) y la extracción de *embeddings* del *linear probe*,
# algo más de 2 minutos; el resto de celdas añade unos pocos minutos.

# %% [markdown] id="8ed21e88"
# # 1. Datos del alumno y título del proyecto
#
# - **Alumno:** Alex Sierra Alcalá
# - **Asignatura:** Deep Learning for NLP — Máster Universitario en Inteligencia Artificial
# - **Proyecto:** *Predicción de etiquetas algorítmicas y dificultad de problemas de Codeforces
#   mediante ajuste fino multitarea de ModernBERT*
# - **Modelo ajustado en el Hub:** [AlexSA45/modernbert-codeforces-tags-rating](
#   https://huggingface.co/AlexSA45/modernbert-codeforces-tags-rating)

# %% [markdown] id="0c91f7a8"
# # 2. Introducción
#
# ## 2.1. La tarea
#
# [Codeforces](https://codeforces.com/) es una de las plataformas de programación competitiva más
# usadas. Cada problema lleva asociadas dos piezas de metadatos:
#
# - **Etiquetas algorítmicas** (`dp`, `greedy`, `graphs`, `math`...): las técnicas con las que se
#   puede resolver. Un problema suele tener varias, así que es un problema de **clasificación
#   multi-etiqueta**.
# - **Rating de dificultad** (de 800 a 3500): estima la habilidad necesaria para resolverlo. Es un
#   problema de **regresión**.
#
# En este proyecto se predicen **ambas cosas a la vez** a partir únicamente del enunciado en inglés.
# Un sistema así sirve para recomendar problemas con los que entrenar una técnica concreta, estimar
# la dificultad de un problema nuevo o etiquetar colecciones de problemas sin metadatos.
#
# La tarea es difícil porque el enunciado describe *qué* hay que calcular, no *cómo*. Además, los
# números de las restricciones (por ejemplo, $n \le 2 \cdot 10^5$ frente a $n \le 20$) son pistas
# decisivas sobre la complejidad esperada de la solución.
#
# ## 2.2. Por qué ModernBERT
#
# Se usa [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
# (Warner et al., 2024), un *encoder* bidireccional al estilo BERT con un diseño moderno. Según su
# *model card*:
#
# - Está preentrenado con **2 billones de tokens de texto en inglés y código**, un dominio cercano
#   al de los enunciados de programación competitiva.
# - Tiene un **contexto nativo de 8192 tokens**, frente a los 512 de BERT. Muchos enunciados de
#   Codeforces superan los 512 tokens.
# - Tiene 22 capas y **149 millones de parámetros**.
#
# Un *encoder* encaja mejor que un LLM generativo en esta tarea: la salida es un vector de
# probabilidades y un número, no texto.
#
# Con unos 150M de parámetros, el modelo, sus gradientes y los estados del optimizador caben en los
# 16 GB de una T4. Técnicas como LoRA o QLoRA están
# pensadas para modelos de miles de millones de parámetros que no caben en memoria; aquí no son
# necesarias, y el ajuste fino completo permite adaptar todas las capas a un dominio muy técnico.
#
# ## 2.3. Por qué multitarea
#
# Etiquetas y dificultad están relacionadas: algunas técnicas aparecen sobre todo en problemas
# difíciles y otras en problemas fáciles. Compartir el *encoder*
# entre las dos tareas permite aprovechar esa relación, y se obtiene un único modelo que resuelve
# ambas con una sola pasada.

# %% [markdown] id="07c6d6b8"
# ## 2.4. Preparación del entorno
#
# Instalamos las librerías necesarias. Se fijan las versiones del ecosistema Hugging Face con las
# que se ha probado el notebook.
#
# - `transformers`: modelos, tokenizadores y la clase `Trainer`.
# - `datasets`: contenedor `Dataset` usado por el `Trainer`.
# - `accelerate`: abstracción del hardware usada internamente por el `Trainer`.
# - `huggingface_hub`: lectura del dataset y subida del modelo al Hub.
# - `scikit-learn`: baselines clásicos y métricas.

# %% id="c26fc00a"
# %pip install -q transformers==5.19.0 datasets==5.1.0 accelerate==1.15.0
# %pip install -q huggingface_hub==1.33.0 scikit-learn pyarrow seaborn

# %% [markdown] id="8eb369b6"
# Importaciones, versiones y semillas. `transformers.set_seed` fija la semilla de `random`, `numpy`
# y `torch` para que los resultados sean reproducibles.

# %% id="53fe2641" outputId="ef6bdccb-625e-4de1-f51d-8deb67324c0e" colab={"base_uri": "https://localhost:8080/"}
import collections
import hashlib
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import Dataset
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, login
from IPython.display import display
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import ModelOutput

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
datasets.disable_progress_bars()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

for lib in [torch, transformers, datasets, sklearn, np, pd]:
    print(f"{lib.__name__:>14}: {lib.__version__}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    print("No se ha detectado GPU")

# %% [markdown] id="fd4c1913"
# ## 2.5. Configuración
#
# Todos los hiperparámetros y *flags* están en esta celda.

# %% id="c78cb696" outputId="b102d7cf-aa12-4577-d7b3-4fd3252bc3e5" colab={"base_uri": "https://localhost:8080/"}
CONFIG = {
    # Flags de ejecución
    "RUN_TRAINING": True,
    "PUSH_TO_HUB": True,
    # Hub
    "HUB_MODEL_ID": "AlexSA45/modernbert-codeforces-tags-rating",
    "HUB_DATASET_ID": "AlexSA45/codeforces-tags-rating-processed",
    # Datos
    "DATASET_ID": "open-r1/codeforces",
    "MIN_TAG_COUNT": 100,     # mínimo de apariciones en train para conservar una etiqueta
    "TOP_K_TAGS": 20,         # máximo de etiquetas a conservar
    # Modelo
    "BASE_MODEL": "answerdotai/ModernBERT-base",
    "MAX_LEN": 1024,
    "DROPOUT": 0.1,
    "LAMBDA_RATING": 0.5,
    "POS_WEIGHT_CLIP": 10.0,
    # Entrenamiento
    "LR": 5e-5,
    "EPOCHS": 5,
    "BATCH_SIZE": 8,
    "GRAD_ACCUM": 2,
    "WARMUP_RATIO": 0.1,
    "WEIGHT_DECAY": 0.01,
    "EARLY_STOPPING_PATIENCE": 2,
    "SEED": 42,
}

MAX_STEPS = -1

SEED = CONFIG["SEED"]
set_seed(SEED)
print(json.dumps(CONFIG, indent=1))

# %% [markdown] id="81c70d0b"
# ## 2.7. Autenticación en Hugging Face Hub

# %% id="090325f8" outputId="1aa48315-ad5c-4454-ae82-8e17254fa6ef" colab={"base_uri": "https://localhost:8080/"}
HF_TOKEN = os.environ.get("HF_TOKEN")
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    pass

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Sesión iniciada en el Hub.")
else:
    CONFIG["PUSH_TO_HUB"] = False
    print("Sin token: no se subirá nada al Hub (PUSH_TO_HUB=False).")
PUSH_TO_HUB = CONFIG["PUSH_TO_HUB"]

# %% [markdown] id="45c5dff3"
# # 3. Dataset
#
# ## 3.1. Fuente
#
# Se usa [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces), publicado por
# Hugging Face dentro del proyecto Open-R1. Contiene algo más de 10 000 problemas únicos de
# Codeforces, desde 2010 hasta principios de 2025, con el enunciado dividido en campos, sus
# etiquetas y su rating.
#
# **Campos usados y descartados.**
#
# | Campo | Uso | Motivo |
# |---|---|---|
# | `title`, `description`, `input_format`, `output_format`, `interaction_format`, `note` | Entrada | Enunciado del problema |
# | `tags` | Objetivo 1 | Etiquetas algorítmicas |
# | `rating` | Objetivo 2 | Dificultad |
# | `id`, `aliases`, `contest_id`, `contest_start`, `index` | Auxiliar | Deduplicación y división temporal |
# | `editorial` | **Descartado** | Explica la solución: sería una fuga de información |
# | `examples` | Descartado | Casos numéricos con poca señal y muchos tokens |
# | `official_tests`, `generated_checker` | Descartado | Tests y validadores, sin relación con la tarea |
#
# **Lectura eficiente.** Los ficheros parquet del dataset pesan unos 2,7 GB, casi todo por la
# columna `official_tests`. Como parquet guarda los datos por columnas, leemos solo las que
# necesitamos con `HfFileSystem` y `pyarrow` y descargamos apenas unos MB. Así, además, el campo
# `editorial` ni siquiera llega a descargarse.

# %% id="25db7800" outputId="404e1463-24f0-4a0e-f6d2-8a329edb7318" colab={"base_uri": "https://localhost:8080/", "height": 458}
COLUMNS = [
    "id", "aliases", "contest_id", "contest_name", "contest_start", "index", "title",
    "description", "input_format", "output_format", "interaction_format", "note", "rating",
    "tags",
]

fs = HfFileSystem()
parquet_files = sorted(fs.glob(f"datasets/{CONFIG['DATASET_ID']}/data/*.parquet"))
frames = []
t0 = time.time()
for path in parquet_files:
    with fs.open(path, "rb", block_size=2**20) as fh:
        frames.append(pq.read_table(fh, columns=COLUMNS).to_pandas())
raw = pd.concat(frames, ignore_index=True)
print(f"{len(parquet_files)} ficheros, {len(raw)} problemas leídos en {time.time() - t0:.0f} s")
raw.head(3)

# %% [markdown] id="433e1399"
# ## 3.2. Construcción del texto de entrada
#
# Unimos todos los *splits* originales y
# construimos un único texto por problema, en este orden:
#
# ```
# Title: ...
# Input: ...
# Output: ...
# Interaction: ...   (solo problemas interactivos)
# Statement: ...
# Note: ...
# ```
#
# Las restricciones (por ejemplo, $1 \le n \le 2 \cdot 10^5$) están casi siempre en la sección de
# entrada y son la pista más informativa sobre la complejidad de la solución. Al ponerlas al
# principio, si el texto supera `MAX_LEN` tokens se trunca la narrativa del final y no ellas.
#
# **Limpieza de LaTeX.** Codeforces escribe las fórmulas entre `$$$`. Sustituimos los comandos más
# frecuentes por su equivalente legible (`\le` → `<=`, `10^{5}` → `10^5`, `\texttt{x}` → `x`...)
# y **conservamos siempre los números**, porque forman parte de la señal.

# %% id="10b97cd7" outputId="90ed22c1-b7a3-4462-fe58-8b014b438f28" colab={"base_uri": "https://localhost:8080/"}
LATEX_RULES = [
    (r"\$\$\$", ""),
    (r"\^\{\\text\{[^{}]*\}\}", ""),                      # marcas de nota al pie: ^{\text{∗}}
    (r"\\(?:leq|le)(?![a-zA-Z])", "<="),
    (r"\\(?:geq|ge)(?![a-zA-Z])", ">="),
    (r"\\(?:neq|ne)(?![a-zA-Z])", "!="),
    (r"\\(?:cdot|times)(?![a-zA-Z])", "*"),
    (r"\\(?:ldots|dots|cdots)(?![a-zA-Z])", "..."),
    (r"\\(?:texttt|textbf|textit|text|mathit|mathrm|operatorname)\{([^{}]*)\}", r"\1"),
    (r"\^\{([^{}]*)\}", r"^\1"),                          # 10^{5} -> 10^5
    (r"\\([a-zA-Z]+)", r"\1"),                            # resto: \gcd -> gcd, \max -> max
    (r"\s+", " "),                                        # colapsar espacios
]
LATEX_RULES = [(re.compile(p), r) for p, r in LATEX_RULES]


def clean_latex(text):
    """Limpia el LaTeX de Codeforces conservando números y operadores."""
    if not isinstance(text, str):
        return ""
    for pattern, repl in LATEX_RULES:
        text = pattern.sub(repl, text)
    return text.strip()


def build_text(row):
    """Concatena los campos del enunciado en el orden descrito arriba."""
    parts = [
        ("Title", row["title"]),
        ("Input", row["input_format"]),
        ("Output", row["output_format"]),
        ("Interaction", row["interaction_format"]),
        ("Statement", row["description"]),
        ("Note", row["note"]),
    ]
    lines = []
    for name, value in parts:
        value = clean_latex(value)
        if value:
            lines.append(f"{name}: {value}")
    return "\n".join(lines)


raw["text"] = raw.apply(build_text, axis=1)

example = raw[raw["input_format"].fillna("").str.contains(r"\\le")].iloc[0]
print("ANTES (input_format):\n", example["input_format"][:400])
print("\nDESPUÉS:\n", clean_latex(example["input_format"])[:400])
print("\nTEXTO COMPLETO (primeros 700 caracteres):\n", example["text"][:700])

# %% [markdown] id="408787e2"
# ## 3.3. Limpieza y deduplicación
#
# - Se eliminan los problemas **sin enunciado**.
# - **Duplicados por alias:** los problemas compartidos entre la Div. 1 y la Div. 2 se registran en
#   `aliases`. Se agrupan con una clave común (el menor identificador del grupo) y se conserva la
#   aparición más antigua.
# - **Duplicados por texto:** se calcula un *hash* del texto normalizado (minúsculas y sin
#   espacios) y también se conserva solo la aparición más antigua.
#
# Si un mismo enunciado apareciera en train y en test, el modelo podría memorizarlo y la
# evaluación sería demasiado optimista.

# %% id="2eb1041b" outputId="84613bc4-590d-4372-8d5f-44f161809bc1" colab={"base_uri": "https://localhost:8080/"}
df = raw[raw["description"].fillna("").str.strip() != ""].copy()
n_no_desc = len(raw) - len(df)
df = df.sort_values(["contest_start", "contest_id", "index"]).reset_index(drop=True)

all_ids = set(df["id"])


def canonical_id(row):
    aliases = row["aliases"] if row["aliases"] is not None else []
    return min([row["id"]] + [a for a in aliases if a in all_ids])


df["canonical_id"] = df.apply(canonical_id, axis=1)
n_before = len(df)
df = df.drop_duplicates("canonical_id", keep="first")
n_alias_dups = n_before - len(df)

df["text_hash"] = df["text"].str.lower().str.replace(r"\s+", "", regex=True).map(
    lambda s: hashlib.md5(s.encode()).hexdigest())
n_before = len(df)
df = df.drop_duplicates("text_hash", keep="first").reset_index(drop=True)
n_text_dups = n_before - len(df)

# Las meta-etiquetas que empiezan por '*' (p. ej. '*special') no son técnicas algorítmicas
df["tags_clean"] = df["tags"].map(
    lambda ts: sorted({t for t in (ts if ts is not None else []) if not t.startswith("*")}))
df["date"] = pd.to_datetime(df["contest_start"], unit="s")

print(f"Problemas sin enunciado eliminados: {n_no_desc}")
print(f"Duplicados por alias eliminados:    {n_alias_dups}")
print(f"Duplicados por texto eliminados:    {n_text_dups}")
print(f"Problemas restantes:                {len(df)}")

# %% [markdown] id="07d31ea1"
# ## 3.4. División temporal
#
# La división es por fecha del concurso, nunca aleatoria: el 80 % más antiguo para train, el
# 10 % siguiente para validación y el 10 % más reciente para test. Así se entrena con problemas
# antiguos y se evalúa con problemas nuevos, igual que ocurriría al usar el modelo en un concurso
# futuro. Una división aleatoria mezclaría problemas del mismo concurso, que comparten autores y
# estilo, entre train y test.

# %% id="a56e4bac" outputId="58f21450-2001-4f37-f1e2-8c62cfa11813" colab={"base_uri": "https://localhost:8080/"}
dates = df["date"].sort_values().reset_index(drop=True)
val_cut = dates.iloc[int(0.8 * len(df))]
test_cut = dates.iloc[int(0.9 * len(df))]
df["split"] = np.where(df["date"] < val_cut, "train",
                       np.where(df["date"] < test_cut, "val", "test"))

assert (df.groupby("contest_id")["split"].nunique() == 1).all()
print("Corte val :", val_cut)
print("Corte test:", test_cut)

# %% [markdown] id="f7bda6ed"
# ## 3.5. Vocabulario de etiquetas y normalización del rating
#
# Todo lo que se aprende de los datos se calcula solo con train:
#
# - **Vocabulario:** etiquetas con al menos `MIN_TAG_COUNT` apariciones en train, hasta un máximo
#   de `TOP_K_TAGS`. Las etiquetas muy raras (`2-sat`, `schedules`...) tienen demasiado pocos
#   ejemplos para aprenderlas. Se descartan los problemas que se quedan sin ninguna etiqueta del
#   vocabulario.
# - **Rating:** se normaliza con *z-score* usando la media y la desviación típica de train. Los
#   problemas sin rating se conservan con `rating_mask = 0`, de modo
#   que no cuentan ni en la pérdida ni en las métricas de rating.

# %% id="7d2decf2" outputId="56d461a6-9dbd-41ab-b4c9-83bc159c31c4" colab={"base_uri": "https://localhost:8080/"}
train_tag_counts = collections.Counter(
    t for ts in df.loc[df["split"] == "train", "tags_clean"] for t in ts)
TAGS = [t for t, c in train_tag_counts.most_common()
        if c >= CONFIG["MIN_TAG_COUNT"]][:CONFIG["TOP_K_TAGS"]]
TAG2IDX = {t: i for i, t in enumerate(TAGS)}
N_TAGS = len(TAGS)
print(f"{N_TAGS} etiquetas conservadas:", TAGS)

df["tags_kept"] = df["tags_clean"].map(lambda ts: [t for t in ts if t in TAG2IDX])
no_tags = df["tags_kept"].map(len) == 0
print("Problemas sin ninguna etiqueta conservada:",
      no_tags.groupby(df["split"]).sum().to_dict())
df = df[~no_tags].reset_index(drop=True)


def multi_hot(tags):
    y = np.zeros(N_TAGS, dtype=np.float32)
    y[[TAG2IDX[t] for t in tags]] = 1.0
    return y


df["y_tags"] = df["tags_kept"].map(multi_hot)
df["rating_mask"] = df["rating"].notna().astype(np.float32)
train_ratings = df.loc[(df["split"] == "train") & df["rating"].notna(), "rating"]
RATING_MEAN, RATING_STD = float(train_ratings.mean()), float(train_ratings.std())
df["rating_z"] = ((df["rating"] - RATING_MEAN) / RATING_STD).fillna(0.0).astype(np.float32)
print(f"Rating en train: media={RATING_MEAN:.1f}, desviación={RATING_STD:.1f}")

SPLITS = ["train", "val", "test"]
data = {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}

# %% [markdown] id="cb8a5999"
# ## 3.6. Análisis exploratorio
#
# ### Tamaño y rango de fechas de cada split

# %% id="d076b8d1" outputId="d977a340-4173-4e4d-86a8-6c8ef01b1454" colab={"base_uri": "https://localhost:8080/", "height": 143}
summary = pd.DataFrame({
    s: {
        "problemas": len(d),
        "concursos": d["contest_id"].nunique(),
        "desde": d["date"].min().date(),
        "hasta": d["date"].max().date(),
        "con rating": int(d["rating_mask"].sum()),
        "rating medio": round(d["rating"].mean(), 1),
        "etiquetas/problema": round(d["tags_kept"].map(len).mean(), 2),
    } for s, d in data.items()
}).T
summary

# %% [markdown] id="5773194e"
# Definimos un estilo común para las figuras.

# %% id="9a65f4a6"
COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
MODEL_COLORS = {"ModernBERT ajustado": "#2a78d6", "TF-IDF + LR": "#eb6834",
                "Linear probe": "#1baf7a"}
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e5e3", "grid.linewidth": 0.8,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.titleweight": "bold", "axes.axisbelow": True,
})

# %% [markdown] id="f81aa94e"
# ### Frecuencia de las etiquetas

# %% id="3c713527" outputId="671ea9b6-1f8e-41d1-8c0f-8e8b914fca24" colab={"base_uri": "https://localhost:8080/", "height": 1000}
tag_freq = pd.DataFrame({s: d["tags_kept"].explode().value_counts() for s, d in data.items()})
tag_freq = tag_freq.reindex(TAGS).fillna(0).astype(int)
for s in SPLITS:
    tag_freq[f"% {s}"] = (100 * tag_freq[s] / len(data[s])).round(1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(TAGS[::-1], tag_freq["train"][::-1], color=COLORS["train"], height=0.7)
ax.set_xlabel("Problemas de train con la etiqueta")
ax.set_title("Frecuencia de cada etiqueta en train")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()
tag_freq

# %% [markdown] id="d0edc2ca"
# ### Coste del vocabulario
#
# Quedarnos con las `TOP_K_TAGS` etiquetas más frecuentes simplifica el problema, pero tiene un
# coste: las etiquetas descartadas desaparecen y algunos problemas se quedan sin ninguna. Medimos
# ese coste en train y cuánto cubren las K etiquetas más frecuentes para distintos valores de K.

# %% id="591bb741" outputId="b1a8ba97-c09b-4d7e-fa01-ffdd827ceb76" colab={"base_uri": "https://localhost:8080/", "height": 626}
from matplotlib.ticker import PercentFormatter

total_occ = sum(train_tag_counts.values())
lost = {t: c for t, c in train_tag_counts.most_common() if t not in TAG2IDX}
n_train_before = int((dates < val_cut).sum())
n_train_dropped = n_train_before - len(data["train"])
keeps_all = (data["train"]["tags_clean"].map(len) == data["train"]["tags_kept"].map(len)).mean()

print(f"Etiquetas distintas en train: {len(train_tag_counts)}; conservadas: {N_TAGS}")
print(f"Apariciones de etiquetas perdidas: {sum(lost.values())} de {total_occ} "
      f"({100 * sum(lost.values()) / total_occ:.1f}%)")
print(f"Problemas de train descartados (sin ninguna etiqueta conservada, incluidos los que "
      f"no tenían etiquetas): {n_train_dropped} de {n_train_before} "
      f"({100 * n_train_dropped / n_train_before:.1f}%)")
print(f"Problemas de train conservados con todas sus etiquetas: {100 * keeps_all:.1f}%")

counts_sorted = np.array([c for _, c in train_tag_counts.most_common()])
coverage = np.cumsum(counts_sorted) / counts_sorted.sum()
ks = np.arange(1, len(coverage) + 1)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(ks, coverage, color=COLORS["train"], linewidth=2, marker="o", markersize=4)
ax.axvline(N_TAGS, color="#0b0b0b", linestyle="--", linewidth=1.5)
ax.annotate(f"K = {N_TAGS}: {100 * coverage[N_TAGS - 1]:.1f}%",
            (N_TAGS, coverage[N_TAGS - 1]), xytext=(8, -18), textcoords="offset points")
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set(title="Apariciones de etiquetas cubiertas por las K más frecuentes (train)",
       xlabel="K (nº de etiquetas conservadas)", ylabel="Apariciones cubiertas")
plt.tight_layout()
plt.show()

pd.Series(lost, name="apariciones en train").to_frame().T

# %% [markdown] id="c4f83a4d"
# ### Coocurrencia de etiquetas
#
# Medimos qué etiquetas aparecen juntas en train con el *lift*:
# $\text{lift}(A, B) = P(A, B) \,/\, (P(A)\,P(B))$. Un *lift* de 1 indica que aparecen juntas lo
# esperable por azar; mayor que 1, que tienden a ir juntas, y menor que 1, que se evitan. Si el
# modelo aprende estas asociaciones, puede predecir pares de etiquetas aunque el texto solo dé
# pistas de una de ellas.

# %% id="e9a66455" outputId="57fca679-cf01-41fd-f2ea-302af75f00df" colab={"base_uri": "https://localhost:8080/", "height": 1000}
Y_tr = np.stack(data["train"]["y_tags"])
co = Y_tr.T @ Y_tr
p = Y_tr.mean(axis=0)
lift = (co / len(Y_tr)) / np.outer(p, p)
log_lift = np.log2(np.clip(lift, 2 ** -2, None))
np.fill_diagonal(log_lift, np.nan)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(log_lift, xticklabels=TAGS, yticklabels=TAGS, cmap="RdBu_r", center=0,
            vmin=-2, vmax=2, linewidths=0.5, linecolor="white",
            cbar_kws={"label": "log2(lift)  (0 = independientes)"}, ax=ax)
ax.set_title("Coocurrencia de etiquetas en train (rojo: juntas; azul: se evitan)")
plt.tight_layout()
plt.show()

pairs = [(TAGS[i], TAGS[j], int(co[i, j]), lift[i, j])
         for i in range(N_TAGS) for j in range(i + 1, N_TAGS) if co[i, j] >= 30]
top_pairs = pd.DataFrame(pairs, columns=["etiqueta A", "etiqueta B", "problemas", "lift"])
top_pairs.sort_values("lift", ascending=False).head(10).round(2)

# %% [markdown] id="50c422b7"
# ### Número de etiquetas por problema y distribución del rating

# %% id="22235320" outputId="2d65df25-26bf-4806-cb87-f69237f72e39" colab={"base_uri": "https://localhost:8080/", "height": 471}
from matplotlib.ticker import PercentFormatter

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

max_tags = max(int(d["tags_kept"].map(len).max()) for d in data.values())
counts = np.arange(0, max_tags + 1)
width = 0.8 / len(SPLITS)
for i, s in enumerate(SPLITS):
    n_tags = data[s]["tags_kept"].map(len)
    props = n_tags.value_counts(normalize=True).reindex(counts, fill_value=0)
    offset = (i - (len(SPLITS) - 1) / 2) * width
    axes[0].bar(counts + offset, props.values, width=width, color=COLORS[s],
                label=f"{s} (n={len(n_tags):,})")
axes[0].set_xticks(counts)
axes[0].set(title="Etiquetas por problema", xlabel="Nº de etiquetas", ylabel="% de problemas")

for s in SPLITS:
    r = np.sort(data[s]["rating"].dropna().to_numpy())
    axes[1].step(r, np.arange(1, len(r) + 1) / len(r), where="post",
                 linewidth=2, color=COLORS[s], label=f"{s} (mediana {np.median(r):.0f})")
axes[1].set(title="Rating acumulado", xlabel="Rating", ylabel="% con rating ≤ x",
            xlim=(800, 3500))

for ax in axes:
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)

plt.tight_layout()
plt.show()

# %% [markdown] id="088f2ce3"
# ### Longitud en tokens
#
# Tokenizamos los textos completos con el tokenizador de ModernBERT para decidir
# `MAX_LEN` con datos y medir cuánto texto se pierde al truncar.

# %% id="0f6ee928" outputId="38268d52-c4e1-4904-81ac-bac4026ff1d2" colab={"base_uri": "https://localhost:8080/", "height": 498}
tokenizer = AutoTokenizer.from_pretrained(CONFIG["BASE_MODEL"])
MAX_LEN = CONFIG["MAX_LEN"]
for s in SPLITS:
    data[s]["n_tokens"] = [len(ids) for ids in tokenizer(list(data[s]["text"]))["input_ids"]]

all_lengths = pd.concat([data[s]["n_tokens"] for s in SPLITS])
pct = all_lengths.quantile([0.5, 0.9, 0.95, 0.99]).astype(int)
print("Percentiles de longitud (tokens):", pct.to_dict())
for limit in [512, 1024]:
    print(f"Textos con más de {limit} tokens: {100 * (all_lengths > limit).mean():.1f}%")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(all_lengths.clip(upper=3000), bins=60, color=COLORS["train"])
ax.axvline(512, color="#52514e", linestyle=":", linewidth=1.5)
ax.axvline(MAX_LEN, color="#0b0b0b", linestyle="--", linewidth=1.5)
ax.text(MAX_LEN, ax.get_ylim()[1] * 0.92, f" MAX_LEN = {MAX_LEN}", fontsize=9)
ax.text(512, ax.get_ylim()[1] * 0.80, " 512 (BERT)", fontsize=9, color="#52514e")
ax.set(title="Longitud de los enunciados en tokens (recortada a 3000)",
       xlabel="Tokens", ylabel="Problemas")
plt.tight_layout()
plt.show()


# %% [markdown] id="7fa28b3c"
# ## 3.7. Tokenización y datasets para el `Trainer`
#
# Tokenizamos con `truncation=True` y `max_length=MAX_LEN`, **sin padding**: el relleno se añade en
# cada lote, solo hasta la longitud del texto más largo del lote. Guardamos también la
# longitud (`length`) para agrupar textos de longitud parecida en el mismo lote.

# %% id="b10dfb1f" outputId="e3d0c4a8-5466-4c2f-a18c-5be1cd8dea2a" colab={"base_uri": "https://localhost:8080/"}
def to_hf_dataset(frame):
    """Tokeniza un split y lo convierte en un `Dataset` con las etiquetas de ambas tareas."""
    enc = tokenizer(list(frame["text"]), truncation=True, max_length=MAX_LEN)
    return Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels_tags": np.stack(frame["y_tags"]).tolist(),
        "labels_rating": frame["rating_z"].tolist(),
        "rating_mask": frame["rating_mask"].tolist(),
        "length": [len(ids) for ids in enc["input_ids"]],
    })


hf = {s: to_hf_dataset(data[s]) for s in SPLITS}
print(hf["train"])

# %% [markdown] id="a27b9aaf"
# Subimos al Hub el dataset ya procesado (texto limpio, etiquetas conservadas, rating y split), para
# poder reutilizarlo sin repetir el preprocesamiento.

# %% id="fa1a6a3a" outputId="cf0e8b99-599d-4d86-e2ea-0758b38abce1" colab={"base_uri": "https://localhost:8080/", "height": 557, "referenced_widgets": ["e8c54b7b0bb34dc1808e4b1d1b386922", "44bbae338708489b9ed8664ec339064e", "4ddf350d2ca04d3ca40fc0f03a3b11d7", "8165fb8bbaec4195af05df0d0e7c2da1", "c3a52b0cd4b84c98a54746315db2d098", "ccc0325587db4a47aabcd0e6287a22b9", "31df795f13814334a905afe39ddfbeb7", "30bd9ba6aa90467096276ae0c48cf983", "cb0c6adb619547cc9becba450b63052c", "74e48104fc7a4ede9cc35e71621e9c57", "bed01b9373194890ac98ef4342dc5d58", "a10dea8d2f13491891a0ad9c61edc40e", "08db622fed964fa19a1cf1d4134c30f5", "f7bfbfb4fc854e04b13808c79b0ecf42", "b65984b63a9840a6a2b0a709e555a6c8", "cced50b6cca14bd39137bbfe1c8bde5b", "f3f424041fbf42808500a009ffa31be7", "9d2358cb6146405d92aab89e3f98054b", "b848bc2ff4a34ff2976dc6956541efea", "7cfd868dbbee45f696cad66675e6eacf", "10fc968755b64035a9472f587ec4f8bc", "76a6adaf67284fbabcf189a7ac45d3f2", "eb29ad09939d46ac8afccbd1e3b70f11", "4e9b3c8dd93146e2aaecd01964b9b480", "900ea12cc401445fb49b70375a9fc735", "af31c8b2bc2d435eb68076ee37572146", "3aed186b32a54af38a97e0f73ae09aee", "a6144810e5bb409499518fa1905c76ba", "98308299417f46a9bb2b70fe4fce6471", "b2824f74b3a54f0090510ba084e053a0", "507e09a7735040d3aaf66f08c172f332", "92b5b57b1602408da2b3a3085ded0def", "4df7058a86c84ebcb55f07a7fd2a3c3a", "f6106fd46ae54ce9a25e85c58f0ef24b", "b0aebd8164514329b5df31710125c17a", "f0f6b544238d433a95a4271eb7c69eb7", "8c6529bb34f24e4d96af343b02e11948", "fe6a0926fcef498884b302d0cd894d97", "525c6e5272ad4857897e55308bc18ce0", "dfcc071fa99d44bd9719ea8e4ec58848", "14eb383599504c13a3fc1fde8252abfb", "ad0ccfe331674d5d9ffbcac8a8a82d07", "e914ab17e33f47fd9b6f1494cdb90faf", "018a78c6af6947ae8c4750ed726c465d", "2fa5ac9e56154ef09fdcf1f0d8a2b8fe", "b8b45e6377b4410e961b3f9c8ed681f3", "73b67d9ca88b40d39722087be87e347d", "78e31d7fecef490d9a1086e5c9f0a2d2", "4586210a1e2f4bc8b3713c9583a839f1", "8276acb70f7f406f88c406f1a7aaa7dc", "5d29c1016b2f4328bbaba9e7eea96445", "560ea8b31156436796e75d3dbdd87b78", "a2a92ccfcfaf4ced8ba08b7d6dea2f5f", "48476c706bf84ba585e63770a144d8b4", "bcc5d7d688554dff8a73adbd9d3ae8e5", "906e79001d4342f6ac5e1893127133da", "dbcba0ad5317462aaee294d30476d5f8", "b12f0d3ae7134186a9d14499e3d49e7e", "8cef1a1654bf472cb416726030cc4b13", "aa873d59330b4b27bb970e5e4c851d8a", "ff9aa0e0440d4876946e31b8463d9556", "c1eba5214dc74f9892e65f7e54196a33", "c56bd2715ecc4301b7818a42b88b79e3", "7d280fa7892f43988a27a5fdf08fa6dc", "ecd25403185a4c2092e7ee491898023e", "a70662c87d0942ce8b9eae6250b9d114", "5da2307500094237a588d5320119a439", "e9512f80a0f6469da391f199dbaf2a2c", "de9cc1b39dc24b549463ed4294987916", "7da3cb65075345dc9fb3f519a0914908", "c1be8191b0634c2ebfd79089d08b1a0a", "78106203e763450287bafbe329dd6074", "3b626e9f752849ff959ad66a29b55aef", "27f9e3c0550346b3ad8f8b3fa54e621d", "24486931268d43d99d272f446aecff33", "d0ed025b602b4df8a5d81ff307337836", "368db5b4615d48979acd25a990070dd3", "ed0ceca49ff04bd2b077d8898e1e056e", "dbd135c4fa044e2983b53bf3ce67ee96", "e7af1ff11b3a461bae6c03c5b1e69a3b", "afe407092b7841d7913212b466d79aab", "6382d498cb534373bbe2497a282ada33", "1a9c3c0146a64053832d812d2f1e561f", "caf34e9416a14962b214d2d6ac6b2d2b", "0c71a31c27b046edae7776ccbfeba735", "4e0d12233bfb4af1a76f8708c9ed4bac", "e21dbae624254a12b4970c9fc0fac4ab", "7b8800d4a5944f938396f81a1a60cb8c", "fa9aa4a548eb435d9340c5fd00c7420c", "a6add581fc0e476ebc479d46ed49f553", "ce156e7f14264d95adadd6b597832b46", "05872adb23fb427faefbdcaff55364cc", "be9354c0d7e548dabdd25ba99d3aeda9", "8735fb929d8e44b994fd2c75e289949e", "9c3a8720797549c492e17de188b39e48", "012557a5da484c34aecd81fe9d119763", "87fd86ae98e7431fa41353cafa29fce6", "eb40566234924375ab873f8bb93eac51", "592a12d119c14b33866dac1a3adf4bfd", "bf0a875049c149dd83d575d38e66ec41", "4e2ed850fdf944b39a6d8724ca0db754", "f8923626ae2946af90f072fe1c91ce71", "b337143bc6c148e2897b09c3506a844f", "733cea62057247049d4d9213ac36df87", "caf9d4df483c4d898bfad61adab07085", "9ba112cd687d4130a7c335d88f88c7f6", "c43c4eef1aee4de4abd604f7aaa32ca0", "f938521a10ce40daab0b1a6674b7807c", "6a449a76a85e46f8bd0a37272baefecd", "be8c9a83bfef4c1db418c776040ac9e4", "cb6d35170890419a9ca7270c39ad1c2c", "68af4bd07cdc4dcdacad790ec4b8df38", "efa8fe3711d14b3f984625083b8b949b", "20bb0e9c97324f0f84ce614970eeff65", "d168a12625c746718d321174540a68f4", "9174734d3e824be39970a57c87896468", "cc12423a64444649b4abffb68f1f675d", "eef76be8ec634e52b84abf7e975885af", "41faa1a4d03a43d1be386885f2677e06", "8f677fd6209e4643888104c9b6570737", "21570eb4b35841eab9ae332577c2be1a", "3e0ed0e086314828aa45dae36850a0d4", "396f74fccd954f84898ba430fd53cd11", "95e06b3a284747bc9c638fd446103171", "082b7e069da44460aaa48916d36e9740", "1de79beeda0b4532bf8d44a8bdb5394d", "12c665d0554448b9afd29c146d586358", "e6a793e8aebb4538b3659f0a9aab7ec7", "ed0543f4c0bb4d08bad36377950905c2", "a855dcd1f6e74679b2d67a95e702f99a", "f30cbf26bb064861a29ac5f565fd7105", "65890222e5fe4b0d9a7939ca4fbfa6fb"]}
if PUSH_TO_HUB:
    keep = ["id", "contest_id", "contest_name", "date", "title", "text", "tags_kept", "rating",
            "split"]
    processed = datasets.DatasetDict({
        s: Dataset.from_pandas(data[s][keep].astype({"date": str}), preserve_index=False)
        for s in SPLITS})
    processed.push_to_hub(CONFIG["HUB_DATASET_ID"])
    print("Dataset procesado subido a", CONFIG["HUB_DATASET_ID"])

# %% [markdown] id="28c25481"
# # 4. Implementación
#
# ## 4.1. Arquitectura del modelo
#
# No existe una clase estándar que resuelva clasificación
# multi-etiqueta y regresión a la vez, así que definimos una propia:
#
# ```
#                         ┌─► Dropout ─► Linear(768, n_tags) ─► logits de etiquetas
# texto ─► ModernBERT ─► mean pooling
#                         └─► Dropout ─► Linear(768, 1) ──────► rating normalizado
# ```
#
# - **Mean pooling** sobre `attention_mask`: promedia los vectores de todos los tokens reales del
#   texto.
#
# ## 4.2. Función de pérdida
#
# La pérdida combina una **entropía cruzada binaria ponderada** para las etiquetas y un **error
# cuadrático medio enmascarado** para el rating:
#
# $$
# \mathcal{L} = \mathrm{BCE}_{w}(z, y) + \lambda \cdot
# \frac{\sum_i m_i (\hat r_i - r_i)^2}{\max(\sum_i m_i, 1)}
# $$
#
# - $z$ son los *logits* de las etiquetas e $y$ el vector *multi-hot* real.
# - $w_k = \min(\text{neg}_k / \text{pos}_k,\ 10)$ es el peso de los positivos de la etiqueta
#   $k$, calculado en train. Compensa el desbalance: sin él, el modelo aprendería a no predecir
#   casi nunca las etiquetas raras. El recorte a 10 evita pesos extremos.
# - $\hat r_i$ y $r_i$ son el rating predicho y el real, ambos normalizados (*z-score*).
# - $m_i$ es `rating_mask`: los problemas sin rating no cuentan en la regresión.
# - $\lambda$ (`LAMBDA_RATING` = 0,5) equilibra las dos tareas.

# %% id="e9ec87d2" outputId="29e35462-9e4c-4c82-db65-53de676e65a4" colab={"base_uri": "https://localhost:8080/", "height": 167}
class CFMultiTaskConfig(PretrainedConfig):
    """Configuración del modelo multitarea."""

    model_type = "cf_multitask"

    def __init__(self, base_model_name="answerdotai/ModernBERT-base", encoder_config=None,
                 tag_names=None, lambda_rating=0.5, pos_weight=None, rating_mean=0.0,
                 rating_std=1.0, dropout=0.1, thresholds=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.encoder_config = encoder_config or {}
        self.tag_names = list(tag_names or [])
        self.n_tags = len(self.tag_names)
        self.lambda_rating = lambda_rating
        self.pos_weight = pos_weight
        self.rating_mean = rating_mean
        self.rating_std = rating_std
        self.dropout = dropout
        self.thresholds = thresholds


@dataclass
class CFMultiTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    tag_logits: Optional[torch.FloatTensor] = None
    rating_pred: Optional[torch.FloatTensor] = None


class CFMultiTaskModel(PreTrainedModel):
    """ModernBERT + mean pooling + dos cabezas (etiquetas y rating)."""

    config_class = CFMultiTaskConfig
    base_model_prefix = "cf_multitask"
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        enc_kwargs = dict(config.encoder_config)
        enc_config = AutoConfig.for_model(enc_kwargs.pop("model_type"), **enc_kwargs)

        self.encoder = AutoModel.from_config(enc_config, attn_implementation="sdpa")
        hidden = enc_config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.tags_head = nn.Linear(hidden, config.n_tags)
        self.rating_head = nn.Linear(hidden, 1)
        pos_weight = config.pos_weight or [1.0] * config.n_tags
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        self.post_init()

    def _init_weights(self, module):
        if module in (self.tags_head, self.rating_head):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.zero_()

    @classmethod
    def from_base(cls, base_model_name, **config_kwargs):
        """Crea el modelo para entrenar: encoder preentrenado + cabezas nuevas."""
        enc_config = AutoConfig.from_pretrained(base_model_name)
        config = CFMultiTaskConfig(base_model_name=base_model_name,
                                   encoder_config=enc_config.to_dict(), **config_kwargs)
        model = cls(config)
        model.encoder = AutoModel.from_pretrained(base_model_name, attn_implementation="sdpa",
                                                  dtype=torch.float32)
        return model

    def forward(self, input_ids=None, attention_mask=None, labels_tags=None,
                labels_rating=None, rating_mask=None, **kwargs):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        tag_logits = self.tags_head(pooled)
        rating_pred = self.rating_head(pooled).squeeze(-1)

        loss = None
        if labels_tags is not None:

            bce = F.binary_cross_entropy_with_logits(
                tag_logits.float(), labels_tags.float(), pos_weight=self.pos_weight)
            m = rating_mask.float()
            mse = (m * (rating_pred.float() - labels_rating.float()) ** 2).sum() / m.sum().clamp(
                min=1.0)
            loss = bce + self.config.lambda_rating * mse
        return CFMultiTaskOutput(loss=loss, tag_logits=tag_logits, rating_pred=rating_pred)


Y_train = np.stack(data["train"]["y_tags"])
pos = Y_train.sum(axis=0)
POS_WEIGHT = np.minimum((len(Y_train) - pos) / np.maximum(pos, 1),
                        CONFIG["POS_WEIGHT_CLIP"]).round(3).tolist()
pd.DataFrame({"positivos en train": pos.astype(int), "pos_weight": POS_WEIGHT}, index=TAGS).T

# %% [markdown] id="ce9ecad1"
# ## 4.3. *Collator* y métricas
#
# El *collator* forma cada lote: rellena `input_ids` y `attention_mask` hasta la longitud máxima
# del lote y apila las tres etiquetas en tensores `float32`.
#
# Las métricas se definen una sola vez y se usan para todos los modelos, para que la comparación sea justa:
#
# - **Etiquetas:** F1 micro (dominado por las etiquetas frecuentes), F1 macro (media por etiqueta,
#   sensible a las raras), F1 por muestra, mAP macro (calidad del orden de las probabilidades,
#   independiente del umbral) y *subset accuracy* (acertar exactamente el conjunto de etiquetas).
# - **Rating** (solo con `rating_mask = 1`): MAE, RMSE, correlación de
#   Spearman y % de predicciones con error absoluto ≤ 200.

# %% id="469c8c52"
class MultiTaskCollator:
    """Padding dinámico de los textos + tensores float32 con las etiquetas."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
             for f in features],
            padding=True, pad_to_multiple_of=8, return_tensors="pt")
        for key in ["labels_tags", "labels_rating", "rating_mask"]:
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.float32)
        return batch


collator = MultiTaskCollator(tokenizer)


def tag_metrics(y_true, scores, thresholds=0.5):
    """Métricas multi-etiqueta; `thresholds` puede ser un escalar o un vector por etiqueta."""
    y_pred = (scores >= np.asarray(thresholds)).astype(int)
    has_pos = y_true.sum(axis=0) > 0
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_samples": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "map_macro": average_precision_score(y_true[:, has_pos], scores[:, has_pos],
                                             average="macro"),
        "subset_acc": float((y_pred == y_true).all(axis=1).mean()),
    }


def rating_metrics(r_true, r_pred, mask):
    """Métricas de regresión en la escala original, solo donde hay rating."""
    m = mask.astype(bool)
    err = r_pred[m] - r_true[m]
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "spearman": float(spearmanr(r_true[m], r_pred[m]).correlation),
        "within_200": float((np.abs(err) <= 200).mean()),
    }


def compute_metrics(eval_pred):
    """Métricas de validación durante el entrenamiento (umbral fijo 0.5)."""
    tag_logits, rating_pred = eval_pred.predictions
    y_tags, y_rating, mask = eval_pred.label_ids
    scores = 1 / (1 + np.exp(-tag_logits))
    out = tag_metrics(y_tags, scores, 0.5)
    out.update(rating_metrics(y_rating * RATING_STD + RATING_MEAN,
                              rating_pred * RATING_STD + RATING_MEAN, mask))
    return out

# %% [markdown] id="ad06be6d"
# ## 4.4. Hiperparámetros
#
# | Parámetro | Valor | Justificación |
# |---|---|---|
# | `max_length` | 1024 | Cubre la mayoría de enunciados; el resto se trunca por el final |
# | `learning_rate` | 5e-5 | Valor habitual para el ajuste fino completo de *encoders* de este tamaño |
# | `per_device_train_batch_size` | 8 | Cabe en la T4 con 1024 tokens en fp16 |
# | `gradient_accumulation_steps` | 2 | Lote efectivo de 16 |
# | `num_train_epochs` | 5 | Con *early stopping* (paciencia 2) |
# | `warmup_steps` | 0.1 | 10 % de pasos de calentamiento y luego decaimiento lineal |
# | `weight_decay` | 0.01 | Regularización estándar de AdamW |
# | `lambda_rating` | 0.5 | Peso de la pérdida de rating |
# | `fp16` | True | Precisión mixta; la T4 no soporta bf16 |
# | `train_sampling_strategy` | `group_by_length` | Agrupa textos de longitud parecida: menos padding |
# | `metric_for_best_model` | `f1_macro` (val, umbral 0.5) | Se queda con la mejor época |
# | `hub_strategy` | `every_save` | Sube el modelo al Hub en cada época |

# %% id="1a0bbcd6"
training_args = TrainingArguments(
    output_dir="runs/main",
    num_train_epochs=CONFIG["EPOCHS"],
    max_steps=MAX_STEPS,
    learning_rate=CONFIG["LR"],
    per_device_train_batch_size=CONFIG["BATCH_SIZE"],
    per_device_eval_batch_size=2 * CONFIG["BATCH_SIZE"],
    gradient_accumulation_steps=CONFIG["GRAD_ACCUM"],
    warmup_steps=CONFIG["WARMUP_RATIO"],
    lr_scheduler_type="linear",
    weight_decay=CONFIG["WEIGHT_DECAY"],
    fp16=torch.cuda.is_available(),
    train_sampling_strategy="group_by_length",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    label_names=["labels_tags", "labels_rating", "rating_mask"],
    remove_unused_columns=False,
    logging_steps=50,
    report_to="none",
    seed=SEED,
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=CONFIG["HUB_MODEL_ID"] if PUSH_TO_HUB else None,
    hub_strategy="every_save",
)

# %% [markdown] id="1a9c13b6"
# ## 4.5. Ajuste fino
#
# Con `RUN_TRAINING = True` se crea el modelo a partir de ModernBERT-base y se entrena; con
# `False` se descarga el ya ajustado desde el Hub junto con el historial del entrenamiento. Al
# terminar, el `Trainer` recarga los pesos de la **mejor época** según el F1 macro de validación.

# %% id="a9ebef33" outputId="482ea70a-bf7f-4df0-f83b-a7d070eb4f56" colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["a88c4245f0e743fcb9c86b642a902809", "38ed905340ed4f4483e5fcdbc218ffbf", "796396fbf5394ae78364709fdef8e603", "ad8d07ecc8a044a9aa20df1318f3494a", "0b64334186644ef2a728ae52585c5217", "ba72c5113f3041f4bde41ae0f68d9254", "d553ebf831874c5e95bcad6fa355d25f", "3bf62857e4ea48b28d9e38ad46fbd219", "47da80a5287241b0af40d3729f70b550", "59eda09f3a194f8ab82b48210be79db9", "e2c56e12cb9442d19f6b3e3a6beecef5", "fc367d9bacc44acba25d51209bc8e84a", "e7a76c478879472fb6a227401107d8a6", "dc88e96e12dc42beae12494fae2fe009", "e3ade11f14f74603960891ff493dad7b", "6aeeca2d1afc46bd8f2a5d07e2b65e1b", "c41cb90358754c12a509f336d3a61724", "204d6bad6e9841f8ac5e4a56d43e154d", "4057d0fb2cc34beda4eb0931f1d14224", "d692fc3fdf9e42b2bfb352c4b06c4f2a", "ca9249e726f24a56b3400ba202a17600", "8d8a8e9d36444b4f8d9a2601af4482b1", "252cc9b0a957494397d7a7352ecd77ac", "742d461a0c164600873249c4345ba804", "bf4e79d3403f4d6fa12f42931d9e5002", "732d347357194c9397bdcdb538b56b57", "3baafd2dbb3941febe18b69eaa0cda8f", "2fd63fe2574e416184835dbb05128478", "7cabea39068143409c22c3cbc971273f", "669d734b5de5470da3d87fd840a61c45", "4ef7fe10cde94cd0a4b4cbaad373eabe", "ffc06e06b43043fab81d7429e19b0654", "6b6d37f94666489f9dad48b41ccd41ad", "c174136a0c5e4a7ab165f6930af756c8", "ed72c796e8094f81addba6af72ebf521", "ba7912e2b86c4581b50ebce56b0cbb34", "a928bd3719f04b19960bc7ed5c0f43b7", "0f532c5ddb554ad8bcd7a3ff60c0e05d", "e0f512cb9bd84d0bb136cc482827b769", "77ec9833de8f4c359536eef9b0376c27", "5bf184f699bd4463b8e34a686cd66ef1", "996cfc5447b94c27825abd9519055dcc", "4336a587d3834f24b399a6d1e51335ba", "d5b643e55abc4746b08cf14cdfcd6369", "f06f3ee6dcd54c8e827630f72b820b1c", "79ed79e8e8014b948785b3545ffcf331", "d9926938c1a74b26854a035719865b94", "b1e85e2949444171b3f6488878fb597f", "fa6167deedb34392a415b72348981e48", "770b8dbc29844d7f8554becb4aa35f51", "a5249629589b4c5483e2731a685cedca", "ce9d4dbfae194c35b7111173259ba805", "d530f034b5534ab0ae194210a4a1b914", "c7574b5266f34f5f8e55263515046adb", "5ca86d86e8be46dca2adf0f2f15a03fa", "a769a45451a1454885abc8b35fa387e6", "0a8df653071840ab92857da2608ad282", "29e8d29608ae4a4183dce2a19cac8071", "375b17a2eb77437bb36efc222be580b0", "35c3139261d54278a8e434622a828b33", "7ff0ef7c6a5a414980cc175b00cbd200", "8f28bd733fb142e7a8f334b683a109b3", "b1c565e4231240959f5f399daf204c84", "50870a3202844f169b11a816c98d914f", "ef9342cb90814d2ea2c6c135b3a60e73", "d3835adcd62749d598a93ffd466fba01", "8bd51a47f564456f96d56388909b7be7", "31dede8c2f4d4dd48494af95bd2a6fc8", "419d8e0480bd4bddb46fccedbde0e7de", "7bdaef21de94433aab1807c54bc08dd4", "d6ae308001254eb6b288c475771c6e50", "dbc11d6f0fa940009c0e2268d05a5577", "80a2078b0dd04f1e8961a15a7fd0955e", "29b17cb1ab754464af9edd9aec1db72b", "ef11f3a6a5b74d9989824c7d3c1db659", "6224acf38fe9474999f3d0ac929e0280", "b4f6043558e246c39d26107f262e7345", "e0a6d486f0db4e20a4d4ef9a74b2ce91"]}
if CONFIG["RUN_TRAINING"]:
    model = CFMultiTaskModel.from_base(
        CONFIG["BASE_MODEL"], tag_names=TAGS, lambda_rating=CONFIG["LAMBDA_RATING"],
        pos_weight=POS_WEIGHT, rating_mean=RATING_MEAN, rating_std=RATING_STD,
        dropout=CONFIG["DROPOUT"])
    trainer = Trainer(
        model=model, args=training_args, train_dataset=hf["train"], eval_dataset=hf["val"],
        data_collator=collator, processing_class=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(CONFIG["EARLY_STOPPING_PATIENCE"])],
    )
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    model = trainer.model
    log_history = trainer.state.log_history
    train_info = {"seconds": train_seconds,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    print(f"Entrenamiento completado en {train_seconds / 60:.1f} min en {train_info['gpu']}")
else:
    model = CFMultiTaskModel.from_pretrained(CONFIG["HUB_MODEL_ID"], dtype=torch.float32)
    log_path = hf_hub_download(CONFIG["HUB_MODEL_ID"], "training_log.json")
    with open(log_path) as fh:
        saved = json.load(fh)
    log_history, train_info = saved["log_history"], saved["train_info"]
    print(f"Modelo descargado de {CONFIG['HUB_MODEL_ID']} "
          f"(entrenado en {train_info['seconds'] / 60:.1f} min en {train_info['gpu']})")

model = model.to(DEVICE).eval()
print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M "
      "(todos entrenables)")

# %% [markdown] id="8c52e520"
# ### Curvas de entrenamiento

# %% id="505f5342" outputId="bdb28fe1-6657-4e9c-e706-dc6f58e3ed52" colab={"base_uri": "https://localhost:8080/", "height": 666}
logs = pd.DataFrame(log_history)
train_logs = logs.dropna(subset=["loss"]) if "loss" in logs else pd.DataFrame()
eval_logs = logs.dropna(subset=["eval_loss"]).set_index("epoch")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if len(train_logs):
    axes[0].plot(train_logs["epoch"], train_logs["loss"], color=COLORS["train"], linewidth=2,
                 label="train")
axes[0].plot(eval_logs.index, eval_logs["eval_loss"], color=COLORS["val"], linewidth=2,
             marker="o", markersize=8, label="val")
axes[0].set(title="Pérdida combinada", xlabel="Época", ylabel="Pérdida")
axes[0].legend(frameon=False)
metric_style = {"eval_f1_macro": ("F1 macro", "#2a78d6"), "eval_f1_micro": ("F1 micro", "#eb6834"),
                "eval_map_macro": ("mAP macro", "#1baf7a")}
for col, (label, color) in metric_style.items():
    axes[1].plot(eval_logs.index, eval_logs[col], color=color, linewidth=2, marker="o",
                 markersize=8, label=label)
axes[1].set(title="Métricas de etiquetas en validación (umbral 0.5)", xlabel="Época")
axes[1].legend(frameon=False)
plt.tight_layout()
plt.show()

eval_cols = ["eval_loss", "eval_f1_micro", "eval_f1_macro", "eval_map_macro", "eval_mae",
             "eval_spearman"]
eval_logs[eval_cols].round(4)

# %% [markdown]
# **Análisis del entrenamiento.**
#
# - El ajuste fino tardó **35,5 minutos** en una T4 y completó las 5 épocas: el *early stopping* no
#   llegó a activarse porque el F1 macro de validación mejoró en todas ellas.
# - La pérdida de train baja de forma continua, de 1,674 en los primeros pasos a 0,560 al final. La
#   de validación, en cambio, se estanca pronto: baja de 1,443 a 1,190 entre las épocas 1 y 2,
#   alcanza su mínimo en la época 4 (1,151) y vuelve a subir en la 5 (1,183). Es una señal de
#   **inicio de sobreajuste**: el modelo se sigue ajustando a train, pero ya no generaliza mejor.
# - El **mAP** de validación, que mide la calidad del orden de las probabilidades y no depende del
#   umbral, se estabiliza desde la época 3 (0,518 → 0,517 → 0,515). El F1 con umbral 0,5 sigue
#   subiendo (macro 0,494 → 0,505 → 0,514) porque las probabilidades se vuelven más extremas y
#   cruzan el umbral con más frecuencia; los umbrales por etiqueta de la sección 5 corrigen parte
#   de ese efecto.
# - El `Trainer` se queda con la **época 5**, la de mayor F1 macro en validación (0,514). En el
#   rating, el MAE de validación baja de 511 a 408 y la correlación de Spearman sube de 0,68 a 0,78.

# %% [markdown] id="623b1433"
# # 5. Resultados y discusión
#
# ## 5.1. Predicciones y ajuste de umbrales
#
# El modelo devuelve una probabilidad por etiqueta. Usar el mismo umbral (0.5) para todas no es
# óptimo: las etiquetas raras suelen necesitar umbrales distintos. Ajustamos un **umbral por
# etiqueta en validación** (rejilla de 0.05 a 0.95 con paso 0.05, maximizando el F1 de cada
# etiqueta) y lo aplicamos sin cambios a test.
#
# Para la inferencia ordenamos los textos por longitud: los lotes tienen menos relleno y la
# predicción es más rápida.

# %% id="1f708598" outputId="1f7afbfc-442c-46f3-bf76-ccba1139906c" colab={"base_uri": "https://localhost:8080/", "height": 118}
def sorted_batches(ds, batch_size):
    """Itera por lotes ordenados por longitud; devuelve también el orden para deshacerlo."""
    order = np.argsort(np.asarray(ds["length"]))
    loader = DataLoader(ds.select(order), batch_size=batch_size, collate_fn=collator)
    return order, loader


@torch.no_grad()
def predict(model, ds, batch_size=32):
    """Probabilidades de cada etiqueta y rating (escala original) en el orden de `ds`."""
    model.eval()
    order, loader = sorted_batches(ds, batch_size)
    probs, ratings = [], []
    for batch in loader:
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            out = model(input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE))
        probs.append(torch.sigmoid(out.tag_logits.float()).cpu().numpy())
        ratings.append(out.rating_pred.float().cpu().numpy())
    probs, ratings = np.concatenate(probs), np.concatenate(ratings)
    inverse = np.argsort(order)
    return probs[inverse], ratings[inverse] * RATING_STD + RATING_MEAN


THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def tune_thresholds(y_true, scores):
    """Umbral por etiqueta que maximiza su F1 en validación."""
    thresholds = []
    for k in range(y_true.shape[1]):
        f1s = [f1_score(y_true[:, k], scores[:, k] >= t, zero_division=0)
               for t in THRESHOLD_GRID]
        thresholds.append(float(THRESHOLD_GRID[int(np.argmax(f1s))]))
    return np.array(thresholds)


Y = {s: np.stack(data[s]["y_tags"]) for s in SPLITS}
R = {s: data[s]["rating"].fillna(0).to_numpy(dtype=float) for s in SPLITS}
M = {s: data[s]["rating_mask"].to_numpy() for s in SPLITS}

ft_val_probs, ft_val_rating = predict(model, hf["val"])
ft_test_probs, ft_test_rating = predict(model, hf["test"])
ft_thresholds = tune_thresholds(Y["val"], ft_val_probs)
pd.Series(ft_thresholds, index=TAGS, name="umbral").to_frame().T

# %% [markdown] id="79c62aaf"
# Guardamos los umbrales en la configuración del modelo y subimos la versión final al Hub, junto
# con el historial del entrenamiento.

# %% id="f757e8fa" outputId="673c234d-5c25-4226-df7d-80399a5d7d73" colab={"base_uri": "https://localhost:8080/", "height": 261, "referenced_widgets": ["c1f3fad10588461b91282192db1120fe", "24963bb0400746dea30a666ed4f6fdd4", "8a0cebe0b9844ca2b4e940d1fe9c0eb6", "93fb7301a67a4d6e92982753d632564b", "47f2d3fe9fe44bf49699b1a27ef86bd7", "3ae7325dfb5641abac683c1092d0226e", "22da797f28c2436bb1bcff8201c090b5", "6b4b0ed343f74fe593fbd19b77480b2c", "b0fe4a9e99a9402db57eeafcf14e7349", "55510ec0b28144fab34294baadc63217", "1daaebfcc8524bbb87f8250d8c227b91", "2479b9e4e6c84d3f8ffcc657a35f0002", "27a792f92e104649b8afc6c1a3bf0c5a", "f688ac425e9140caaa3a21f55583709f", "5d3d4e9e90044afcb4f3e8ac8cccb1f3", "055b3275b9a84bb0ba2865290dfcb8d6", "f0798273372d48c1b15f1b55c633667d", "ce121075c0ce4e399a7d89b75ce29f2b", "46ee39651ac0463d90c2c7fc0216d46e", "a696db27299d4e6bb1d91a75e7da498c", "203ddb42193047129518197a0dac5b75", "b30ea60c2ed94053b4a2d7860940c214", "18162e921a7448e9ad95a42e693ae1f2", "2f6d9fdf9ef147419027a548f4d405a0", "1e26ab3055e64d3d86940382ad363d62", "7afda391110041ada934e85b5efe9fbe", "0f4e91d693724dacba0481cc9088a744", "e3a533fa14c846a99bcf6f86e6e1c3f6", "3f5ba6fa28424c2fbf11e08ce74b7d2a", "d873eb26ab5341808cedf840a523f4fa", "9b593e762604403a88afe5eb4f36f0c6", "56e1acaac75e4e47a9e42207934a9e0d", "7de3145b9f04482581b6756c533f58a6", "4b66c4f534d84990aeb634295e3814d3", "532483ad60cd4725962d18b91b5d6dfd", "0eb4d06ae0724e64a3dfb47267daa2e4", "1126a556975d4e9baa961dd356e6b97a", "fa31284d1bc94ed3a0d9d8d72c0b5ca0", "ddd0240839054e90af2c41ee345fded4", "6eb38e6e2b6b4d9e982acd489a2fecb6", "4c7fde761f364e6eae0ca5be523d1d1d", "6286df2c1c264954aca3a3f960ad807b", "a05b606c90814f758e0fa8a50b4c8eb4", "1c2ad8ad9c5c4e4e9b84aa5b1e45b84d", "ce2cb46f29ea43c6a44cf9527373ecee", "65ef715bb2c7403fbe3b50ec7282b5ab", "84bae51fc9e042209ab99767d88c0421", "00e68fb4e73b42718a42b02c7ec991fe", "3eb0ffd0334848b88a4cf860ebf0f761", "fbcdb48304ac47019c895c3d7cbc4557", "900088f945a149b3ac0090121fb7e671", "2a90d9f33fca4bd588e2d517d135ba88", "922352f617f246ea8fabcadd58f7c0a7", "73f944f6763d47808b14a102b7073f7a", "8a59ef946a3b4579a8266cfd26306579", "37f274facc7a4cd09d41296df50381fc", "c185492fce0d4681b4aa51f6be379267", "6ada89a0f0314819aec98170f6a0705c", "90adfb0280624bb1800eec3255073f74", "96fa7ec086c94a8980f58f5085a4b949", "68e44dce51c247119af46842954551f0", "78b3c4419f294f178cc7793f248eac8c", "d529afaf76804b509a816d0f55983921", "b91ada9d806c42388622f91778e14be6", "8ead3f5ccb044df19b9906134a8de1fd", "a384de7f6ac9408d9befbb46800d9b4a"]}
model.config.thresholds = ft_thresholds.tolist()
if PUSH_TO_HUB and CONFIG["RUN_TRAINING"]:
    model.push_to_hub(CONFIG["HUB_MODEL_ID"], commit_message="Modelo final con umbrales")
    tokenizer.push_to_hub(CONFIG["HUB_MODEL_ID"])
    log_file = "training_log.json"
    with open(log_file, "w") as fh:
        json.dump({"log_history": log_history, "train_info": train_info}, fh)
    HfApi().upload_file(path_or_fileobj=log_file, path_in_repo=log_file,
                        repo_id=CONFIG["HUB_MODEL_ID"])
    print("Modelo, tokenizador, umbrales e historial subidos a", CONFIG["HUB_MODEL_ID"])

# %% [markdown] id="3ab1c6d5"
# ### Uso del modelo publicado
#
# Esta celda carga el modelo **desde el Hub** y predice las etiquetas y la dificultad de un
# enunciado de test. Es todo lo que necesita un usuario del modelo: el tokenizador, la clase
# `CFMultiTaskModel` y los umbrales guardados en la configuración.

# %% id="5c98ed7e" outputId="d8d5c5d9-946d-428a-fc17-5344a85072af" colab={"base_uri": "https://localhost:8080/", "height": 269, "referenced_widgets": ["8c476ff12d564ca0bc6de5a2a43a2ec9", "44786f5c479849738ccb8458728ca3c9", "621702afe5cc4004a01beec07179c520", "eb799574a2424af4950ac66aec286198", "88d5d669162c405f838ccbf1f96d5091", "dcae1681d0334f7d8571a072b3df768d", "201436ed3711461a9542922d51074ffe", "62a659cd449643b99189d0d3217bc6e5", "75e3e1c1749748679ea802e7dd3b5bff", "4cac035fb3484da59426602014e0c3a1", "ffb4c861f18b46dc98f5d98e75a6eb24", "856487b7b40643b6a8cbd95b3176b28e", "ccd97b86a04045f8a276cc3bc31c007d", "4ff20b7495534357859c57681b028bea", "8f740cdf8eb84398ab70a81713f8ee11", "a5a361d9ae4c4bd2b55969da8ec58769", "b6615c8705544a928173a91d3231e9fa", "7e1d0d4248d4476e8964a007370186e8", "805c1ee554c64c09a247b51885bdf20e", "f92a6f071e3d4f9db667a9afb450bcb8", "7f5be6a3ec8c43e5ada8d356375adbb6", "ce0d79ca8634443c99877a9f6e56ddf2", "90ad416247b447d98284d19ea7b2a31e", "7e68259d1657423397b8b60714f14403", "6d443183ed734e0980eea2dbd9150b25", "22799a2ebbf74f1ebd1243c1aa10dce1", "276253ebf7da42afa9ad06a942d1e18e", "2461c9c23a174d80bea5d73694b23542", "4520cc1bac6a41ab8acfa72965a2ebb5", "36ace732ae614a09b8edf544af97c0c4", "bd45e4da56294bc794e49b150cae53dd", "12b95d14e88b44128d3f365bb3738480", "f0e3668715644839802e0c382e6f0c93", "5f161f7f6093415a80d94ba66d325fa1", "6a766720697140759c3c4e9f16e10e4f", "88b136cf8e5e42c68023b5cb5374f19c", "ae22292de70a42fcb860a206af4654ad", "f259e2fea1e5411abd1bdfeecb1556b6", "ac29616dcbcc43db8aca4a294363ab3a", "9be2f79406164d5c9a38a7bf5ff08bc2", "d23f2cc7502049b6bebfabb53851553d", "62e7b2c709fe42518df3c3ac19c3b9da", "268c3a661baa444a97756168a2a1d6e3", "eb66bc1216e443cab8f3d42496e53975"]}
def predict_problem(model, text):
    """Etiquetas (con su probabilidad) y rating predichos para un enunciado ya limpio."""
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    probs = torch.sigmoid(out.tag_logits.float())[0].cpu().numpy()
    thr = np.array(model.config.thresholds or [0.5] * model.config.n_tags)
    tags = {model.config.tag_names[k]: round(float(probs[k]), 3)
            for k in np.argsort(-probs) if probs[k] >= thr[k]}
    rating = float(out.rating_pred[0]) * model.config.rating_std + model.config.rating_mean
    return tags, rating


if PUSH_TO_HUB or not CONFIG["RUN_TRAINING"]:
    demo_model = CFMultiTaskModel.from_pretrained(CONFIG["HUB_MODEL_ID"]).to(DEVICE).eval()
else:
    demo_model = model
sample = data["test"].iloc[0]
pred_tags, pred_rating = predict_problem(demo_model, sample["text"])
print(sample["text"][:500], "...\n")
print("Etiquetas reales :", sample["tags_kept"], "| rating real:", sample["rating"])
print("Etiquetas predichas:", pred_tags, f"| rating predicho: {pred_rating:.0f}")
if demo_model is not model:
    del demo_model

# %% [markdown] id="d6696190"
# ## 5.2. Baselines
#
# Para saber si el ajuste fino aporta algo hay que compararlo con referencias razonables. Todas se
# evalúan sobre el mismo test, con las mismas métricas y con umbrales por etiqueta
# ajustados en validación:
#
# - **B0, trivial.** Cada etiqueta recibe como puntuación su frecuencia en train y el rating es
#   siempre la media de train. Es el mínimo que cualquier modelo debe superar.
# - **B1, TF-IDF + modelos lineales.** Bolsa de unigramas y bigramas con regresión logística y regresión
#   `Ridge` para el rating. Es el baseline clásico en clasificación de textos.
# - **B2, ModernBERT sin ajustar (*linear probe*).** Es el modelo base sin ajustar: se extraen
#   los *embeddings* de ModernBERT-base congelado y se
#   entrenan encima los mismos modelos lineales. Mide cuánto aporta ajustar el *encoder* frente a
#   usar solo sus representaciones preentrenadas.

# %% id="6e9bfeae" outputId="efa36b53-85ec-4c3c-e8b6-98feb686ffc5" colab={"base_uri": "https://localhost:8080/"}
results = {}


def register(name, val_scores, test_scores, val_rating, test_rating):
    """Guarda las salidas de un modelo y ajusta sus umbrales en validación."""
    results[name] = {
        "val_scores": val_scores, "test_scores": test_scores,
        "val_rating": val_rating, "test_rating": test_rating,
        "thresholds": tune_thresholds(Y["val"], val_scores),
    }


register("ModernBERT ajustado", ft_val_probs, ft_test_probs, ft_val_rating, ft_test_rating)

# B0: frecuencia a priori de cada etiqueta y media del rating
prior = Y["train"].mean(axis=0)
register("B0 Trivial",
         np.tile(prior, (len(Y["val"]), 1)), np.tile(prior, (len(Y["test"]), 1)),
         np.full(len(Y["val"]), RATING_MEAN), np.full(len(Y["test"]), RATING_MEAN))


def fit_linear_baselines(X_train, X_val, X_test, name):
    """Regresión logística por etiqueta (C elegido por mAP en val) + Ridge para el rating."""
    best = None
    for C in [0.1, 1, 10]:
        clf = OneVsRestClassifier(LogisticRegression(
            C=C, class_weight="balanced", solver="liblinear", max_iter=2000))
        clf.fit(X_train, Y["train"])
        val_scores = clf.predict_proba(X_val)
        val_map = tag_metrics(Y["val"], val_scores)["map_macro"]
        print(f"  {name}: C={C:<4} mAP macro val = {val_map:.4f}")
        if best is None or val_map > best[0]:
            best = (val_map, C, clf, val_scores)
    _, C, clf, val_scores = best
    m = M["train"].astype(bool)
    ridge = Ridge(alpha=1.0).fit(X_train[m], R["train"][m])
    register(name, val_scores, clf.predict_proba(X_test), ridge.predict(X_val),
             ridge.predict(X_test))
    print(f"  {name}: C elegido = {C}")


# B1: TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
X_tfidf = {"train": tfidf.fit_transform(data["train"]["text"])}
for s in ["val", "test"]:
    X_tfidf[s] = tfidf.transform(data[s]["text"])
fit_linear_baselines(X_tfidf["train"], X_tfidf["val"], X_tfidf["test"], "TF-IDF + LR")

# %% [markdown] id="3cf62796"
# Para el *linear probe* extraemos los *embeddings* de ModernBERT-base congelado, en fp16 y sin
# gradientes, con el mismo *mean pooling* que el modelo ajustado.

# %% id="facf1057" outputId="6706d6dc-b5f7-469a-b1b1-71fcce64c064" colab={"base_uri": "https://localhost:8080/", "height": 136, "referenced_widgets": ["3bcaae9d70d84167a0fbd35fdb836db3", "a2bbd39dc0e941c882dc7c3d32c3fd21", "4386b571a4e74008b558fa77d0caeaaf", "67283be4bf11403981623dc83af84430", "8d40677dcd6c41758b6dc1a2064e3d73", "39bc12fc9b3e41128c2f25cb25fb8180", "544181be820b4bee8c47950d94a479be", "7c586c8cceb64044a0a92ce29efc1be5", "6a0f04267cf94464a506ae8d071ef32d", "917f66179abe4a8d9c9fe0fdd76feb45", "5a1337bfa585447dae7dd95a19eea8e6"]}
@torch.no_grad()
def embed(encoder, ds, batch_size=32):
    """Embeddings (mean pooling) de un encoder congelado, en el orden de `ds`."""
    order, loader = sorted_batches(ds, batch_size)
    out = []
    for batch in loader:
        ids, att = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
        hidden = encoder(input_ids=ids, attention_mask=att).last_hidden_state.float()
        mask = att.unsqueeze(-1).float()
        out.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.concatenate(out)[np.argsort(order)]


frozen = AutoModel.from_pretrained(
    CONFIG["BASE_MODEL"], attn_implementation="sdpa",
    dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32).to(DEVICE).eval()
t0 = time.time()
X_emb = {s: embed(frozen, hf[s]) for s in SPLITS}
print(f"Embeddings extraídos en {time.time() - t0:.0f} s; forma de train: {X_emb['train'].shape}")
del frozen
torch.cuda.empty_cache()

scaler = StandardScaler().fit(X_emb["train"])
X_emb = {s: scaler.transform(x) for s, x in X_emb.items()}
fit_linear_baselines(X_emb["train"], X_emb["val"], X_emb["test"], "Linear probe")

# %% [markdown] id="c5a65a65"
# ## 5.3. Tabla principal de resultados
#
# Todos los modelos sobre el **test completo**, con umbrales ajustados en validación.

# %% id="b8975ad5" outputId="9a630b7e-13f7-4c7e-937e-7cd1badfa651" colab={"base_uri": "https://localhost:8080/", "height": 192}
ROW_ORDER = ["B0 Trivial", "TF-IDF + LR", "Linear probe", "ModernBERT ajustado"]


def evaluate(name, split="test", thresholds=None):
    """Métricas de un modelo en un split (por defecto, con sus umbrales ajustados)."""
    res = results[name]
    thr = res["thresholds"] if thresholds is None else thresholds
    row = tag_metrics(Y[split], res[f"{split}_scores"], thr)
    row.update(rating_metrics(R[split], res[f"{split}_rating"], M[split]))
    return row


main_table = pd.DataFrame({name: evaluate(name) for name in ROW_ORDER}).T
main_table.columns = ["F1 micro", "F1 macro", "F1 samples", "mAP macro", "Subset acc", "MAE",
                      "RMSE", "Spearman", "% |err|≤200"]
print(f"Test: {len(Y['test'])} problemas")
main_table.round(3)

# %% [markdown]
# **Discusión de la tabla principal.**
#
# - El **modelo ajustado** es el mejor en F1 macro (0,530), mAP macro (0,528) y *subset accuracy*
#   (0,029), y en **todas las métricas de rating**: MAE 442, RMSE 584, Spearman 0,713 y un 33,3 % de
#   predicciones con error ≤ 200.
# - **Frente al modelo base sin ajustar (*linear probe*)** la mejora es clara en todo: +0,086 de
#   F1 macro, +0,120 de mAP y 79 puntos menos de MAE. Las representaciones preentrenadas de
#   ModernBERT, usadas tal cual, no bastan: lo que aporta el rendimiento es ajustar el *encoder*.
#   De hecho, el *linear probe* queda por debajo de TF-IDF; los *embeddings* genéricos, promediados
#   sobre todo el texto, recogen peor las palabras clave del dominio que una bolsa de palabras.
# - **Frente a TF-IDF** la diferencia en etiquetas es pequeña: el modelo ajustado gana en F1 macro
#   (+0,011) y mAP (+0,013), empata en F1 por muestra (0,535 frente a 0,537) y pierde ligeramente
#   en F1 micro (0,545 frente a 0,554), una métrica dominada por las etiquetas frecuentes. En el
#   rating la ventaja es más clara (MAE 442 frente a 471; Spearman 0,713 frente a 0,681). Muchas
#   etiquetas tienen pistas léxicas directas (*tree*, *string*, *modulo*) que una bolsa de palabras
#   ya captura bien; la ventaja del *encoder* se nota más en la dificultad, que depende de cómo se
#   combinan la descripción y las restricciones. Con 985 problemas de test, diferencias de un punto
#   pueden deberse al azar, así que en etiquetas lo prudente es hablar de rendimiento comparable.
# - **B0** obtiene un F1 micro de 0,326 solo por predecir las etiquetas más frecuentes, y su mAP
#   (0,172) corresponde al nivel del azar. Su Spearman es NaN porque predice siempre la misma
#   media. Todos los modelos lo superan con amplitud.
# - La ***subset accuracy*** es muy baja en todos los modelos (≤ 0,03): acertar exactamente el
#   conjunto de etiquetas es muy exigente con 20 etiquetas posibles y una media de 3,43 etiquetas
#   por problema en test.

# %% [markdown] id="50f30adf"
# ## 5.4. Efecto del ajuste de umbrales

# %% id="541aaff2" outputId="00126222-ca98-48f1-d7ee-4cd467d586e5" colab={"base_uri": "https://localhost:8080/", "height": 143}
threshold_rows = {}
for name in ROW_ORDER[1:]:
    fixed = evaluate(name, thresholds=0.5)
    tuned = evaluate(name)
    threshold_rows[name] = {"F1 micro (0.5)": fixed["f1_micro"],
                            "F1 micro (ajustado)": tuned["f1_micro"],
                            "F1 macro (0.5)": fixed["f1_macro"],
                            "F1 macro (ajustado)": tuned["f1_macro"]}
pd.DataFrame(threshold_rows).T.round(3)

# %% [markdown]
# **Efecto de los umbrales.**
#
# - En TF-IDF y en el *linear probe*, ajustar un umbral por etiqueta mejora el F1 micro algo más de
#   un punto (0,540 → 0,554 y 0,471 → 0,483) y el F1 macro entre 0,4 y 1,6 puntos.
# - En el modelo ajustado apenas cambia: el F1 micro se queda en 0,545 y el macro sube de 0,524 a
#   0,530. La pérdida con `pos_weight` ya desplaza hacia arriba las probabilidades de las etiquetas
#   raras, así que el umbral 0,5 es un punto de partida razonable.
# - Los umbrales elegidos en validación (sección 5) están casi todos **por encima de 0,5** (entre
#   0,55 y 0,8), salvo `implementation` (0,3), `brute force` (0,4) y `two pointers` (0,5). Con
#   `pos_weight`, el modelo da probabilidades altas con facilidad y conviene exigirle más
#   confianza antes de predecir una etiqueta.

# %% [markdown] id="de289afb"
# ## 5.5. Análisis por etiqueta

# %% id="33723cad" outputId="3aa89c9c-d509-4b2e-c437-3c318a3b8d14" colab={"base_uri": "https://localhost:8080/", "height": 1000}
per_tag = {}
for name in ["ModernBERT ajustado", "TF-IDF + LR", "Linear probe"]:
    res = results[name]
    pred = (res["test_scores"] >= res["thresholds"]).astype(int)
    p, r, f, _ = precision_recall_fscore_support(Y["test"], pred, zero_division=0)
    per_tag[name] = pd.DataFrame({"precisión": p, "recall": r, "F1": f}, index=TAGS)
per_tag_ft = per_tag["ModernBERT ajustado"].copy()
per_tag_ft["frecuencia train"] = Y["train"].sum(axis=0).astype(int)
per_tag_ft["positivos test"] = Y["test"].sum(axis=0).astype(int)

f1_by_model = pd.DataFrame({n: t["F1"] for n, t in per_tag.items()})
f1_by_model = f1_by_model.sort_values("ModernBERT ajustado")
fig, ax = plt.subplots(figsize=(9, 9))
y_pos = np.arange(len(TAGS))
height = 0.26
for i, name in enumerate(f1_by_model.columns):
    ax.barh(y_pos + (i - 1) * height, f1_by_model[name], height=height * 0.9,
            color=MODEL_COLORS[name], label=name)
ax.set_yticks(y_pos, f1_by_model.index)
ax.set(title="F1 por etiqueta en test", xlabel="F1")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.show()
per_tag_ft.sort_values("F1", ascending=False).round(3)

# %% [markdown]
# **Análisis por etiqueta.**
#
# - Las etiquetas que mejor predice el modelo son `trees` (F1 0,790), `combinatorics` (0,730),
#   `strings` (0,728), `greedy` (0,698) y `math` (0,677). Salvo `greedy`, todas tienen un
#   vocabulario muy reconocible en el enunciado: árboles, cadenas o conteos módulo un primo.
# - Las peores son `dsu` (0,234), `two pointers` (0,244), `binary search` (0,339), `sortings`
#   (0,374) y `shortest paths` (0,417). Son técnicas de solución que el enunciado rara vez menciona
#   y que a menudo son **alternativas** a otras: en la tabla de coocurrencia de la sección 3.6,
#   `dsu` aparece con `dfs and similar` con un *lift* de 5,13, y `two pointers` con `binary search`
#   con un *lift* de 3,83. Varias técnicas resuelven el mismo problema, y el modelo no puede saber
#   cuál eligieron quienes etiquetaron.
# - **La frecuencia no lo explica todo**: `implementation` es la etiqueta más frecuente en train
#   (2252 problemas) y obtiene solo 0,490, mientras que `trees` (619) es la mejor. `implementation`,
#   `brute force` y `two pointers` comparten un patrón: *recall* alto (0,753; 0,764; 0,595) y
#   precisión muy baja (0,363; 0,304; 0,154). El modelo las usa como "comodín" cuando el enunciado
#   no da pistas claras.
# - Frente a los baselines (figura), el modelo ajustado destaca sobre todo en `trees`,
#   `shortest paths`, `geometry`, `bitmasks` y `number theory`, mientras que TF-IDF es algo mejor en
#   `graphs`, `sortings` y `binary search`. Algunas etiquetas tienen muy pocos positivos en test
#   (23 en `shortest paths`, 27 en `geometry`), así que sus diferencias son poco fiables.

# %% [markdown] id="d464adbc"
# ## 5.6. Predicción del rating

# %% id="46edc2d3" outputId="6ed3181a-d7fd-43e5-ffe9-b3379f6de85f" colab={"base_uri": "https://localhost:8080/", "height": 689}
mask_test = M["test"].astype(bool)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.scatter(R["test"][mask_test], ft_test_rating[mask_test], s=10, alpha=0.4,
           color=COLORS["train"])
ax.plot([800, 3500], [800, 3500], color="#0b0b0b", linestyle="--", linewidth=1.5)
ax.set(title="Rating predicho frente a real (modelo ajustado)", xlabel="Rating real",
       ylabel="Rating predicho", xlim=(700, 3600), ylim=(700, 3600))

buckets = [(800, 1200), (1300, 1700), (1800, 2200), (2300, 2700), (2800, 3500)]
bucket_rows = {}
for name in ["ModernBERT ajustado", "TF-IDF + LR", "Linear probe"]:
    err = np.abs(results[name]["test_rating"] - R["test"])
    bucket_rows[name] = [err[mask_test & (R["test"] >= lo) & (R["test"] <= hi)].mean()
                         for lo, hi in buckets]
bucket_df = pd.DataFrame(bucket_rows, index=[f"{lo}–{hi}" for lo, hi in buckets])
x = np.arange(len(buckets))
for i, name in enumerate(bucket_df.columns):
    axes[1].bar(x + (i - 1) * 0.26, bucket_df[name], width=0.24, color=MODEL_COLORS[name],
                label=name)
axes[1].set_xticks(x, bucket_df.index)
axes[1].set(title="MAE por rango de dificultad (test)", xlabel="Rating real", ylabel="MAE")
axes[1].grid(axis="x", visible=False)
axes[1].legend(frameon=False)
plt.tight_layout()
plt.show()
bucket_df.assign(problemas=[int((mask_test & (R["test"] >= lo) & (R["test"] <= hi)).sum())
                            for lo, hi in buckets]).round(1)

# %% [markdown]
# **Análisis del rating.**
#
# - La nube de puntos queda por encima de la diagonal en los problemas fáciles y por debajo en los
#   difíciles: el modelo tiende a predecir valores intermedios (**regresión a la media**). El error
#   cuadrático penaliza mucho los errores grandes y, ante la duda, compensa acercarse a la media.
# - Por rangos, el modelo ajustado es el mejor **en los extremos**: en 800–1200 su MAE es 318,
#   frente a 451 de TF-IDF y 515 del *linear probe*, y en 2800–3500 es 939, frente a 990 y 1049. En
#   los rangos centrales TF-IDF es algo mejor (293 frente a 338 en 1300–1700 y 287 frente a 360 en
#   1800–2200).
# - El error se dispara en los problemas más difíciles (MAE 939 en 2800–3500, con 144 problemas).
#   Por encima de 2800, el enunciado no suele dar más pistas que el de un problema de 2200: la
#   dificultad real depende de la idea de la solución, no solo del texto.

# %% [markdown] id="89b914a8"
# ## 5.7. Análisis de errores
#
# Ejemplos de aciertos y fallos del modelo ajustado en test, ordenados por el F1 de cada problema.

# %% id="52dc4232" outputId="606e1a82-809c-48b5-b0a7-15521de7f6b5" colab={"base_uri": "https://localhost:8080/", "height": 582}
pred_test = (ft_test_probs >= ft_thresholds).astype(int)
sample_f1 = f1_score(Y["test"].T, pred_test.T, average=None, zero_division=0)


def show_examples(indices):
    rows = []
    for i in indices:
        row = data["test"].iloc[i]
        statement = row["text"].split("Statement: ", 1)[-1]
        rows.append({
            "título": row["title"],
            "enunciado (fragmento)": statement[:300] + "...",
            "etiquetas reales": ", ".join(row["tags_kept"]),
            "etiquetas predichas": ", ".join(TAGS[k] for k in np.flatnonzero(pred_test[i])),
            "F1": round(sample_f1[i], 2),
            "rating real": row["rating"],
            "rating predicho": round(ft_test_rating[i]),
        })
    return pd.DataFrame(rows)


ranked = np.argsort(-sample_f1, kind="stable")
with pd.option_context("display.max_colwidth", 400):
    print("Aciertos (mayor F1 por problema)")
    display(show_examples(ranked[:3]))
    print("Fallos (menor F1 por problema)")
    display(show_examples(ranked[-3:]))

# %% [markdown]
# **Patrones en los errores.**
#
# - Los aciertos completos son problemas con vocabulario explícito. Por ejemplo, *Short Permutation
#   Problem* pide contar permutaciones "modulo 998 244 353", una pista directa de combinatoria con
#   programación dinámica. Aun así, el rating puede fallar mucho aunque las etiquetas sean perfectas
#   (1900 real frente a 2731 predicho).
# - Los fallos totales (F1 = 0) son problemas cuya técnica no se deduce del enunciado. *Kevin and
#   Nivek* es un problema de `dp` de rating 3500 que el modelo etiqueta con `implementation`,
#   `greedy`, `sortings`, `binary search` y `two pointers`; las etiquetas de *Permutation Factory*
#   (`geometry`, `graphs`, rating 3500) dependen de cómo se modela la solución, no de lo que cuenta
#   el texto.
# - En los fallos, el modelo recurre a las etiquetas genéricas y frecuentes (`implementation`,
#   `greedy`, `constructive algorithms`, `brute force`), justo las de *recall* alto y precisión baja
#   del análisis por etiqueta, y **subestima el rating** de los problemas más difíciles (3500 → 1244
#   y 3500 → 1866).
# - El ejemplo de uso del modelo publicado muestra el mismo sesgo: en un problema fácil de
#   `implementation` (rating 1100) acierta esa etiqueta, pero añade otras cuatro.

# %% [markdown] id="d98f160e"
# ## 5.8. Limitaciones
#
# - **Ruido en las etiquetas.** Las etiquetas las asignan los autores y la comunidad, y un problema
#   puede admitir varias soluciones válidas con técnicas distintas: parte del "error" no es tal.
# - **Desbalance.** Las etiquetas raras tienen pocos ejemplos; `pos_weight` y los umbrales por
#   etiqueta lo mitigan, pero no lo eliminan.
# - **Dificultad intrínseca del rating.** El rating depende de cómo se comportaron los
#   participantes en el concurso, algo que el enunciado no refleja del todo.
# - **Truncamiento.** Los textos más largos que `MAX_LEN` pierden su parte final.
# - **Deriva temporal de las etiquetas.** Los problemas recientes llevan más etiquetas (2,54 por
#   problema en train, 3,15 en validación y 3,43 en test) y algunas ganan mucho peso: `greedy` pasa
#   del 28,2 % de los problemas de train al 47,0 % en test. El modelo aprende de problemas
#   etiquetados de forma más escueta que aquellos con los que se evalúa; los umbrales ajustados en
#   validación lo compensan solo en parte.
# - **Solo inglés** y solo problemas de Codeforces.

# %% [markdown] id="2bee19d5"
# # 6. Conclusiones
#
# ## 6.1. Resumen de resultados
#
# - Se ha ajustado **ModernBERT-base completo**, con un modelo propio de dos cabezas, para predecir
#   a la vez las etiquetas algorítmicas y el rating de problemas de Codeforces a partir del
#   enunciado. El entrenamiento tarda 35,5 minutos en una T4.
# - En test (985 problemas, los más recientes), el modelo ajustado obtiene un **F1 macro de 0,530**,
#   un **mAP de 0,528** y un **MAE de 442 puntos** de rating, con una correlación de Spearman de
#   0,713.
# - **El ajuste fino aporta claramente** frente al modelo base sin ajustar: respecto al *linear
#   probe*, +0,086 de F1 macro, +0,120 de mAP y 79 puntos menos de MAE.
# - Frente a un baseline TF-IDF bien ajustado, el rendimiento en etiquetas es **comparable** (mejor
#   en F1 macro y mAP, algo peor en F1 micro) y en el rating es **mejor en todas las métricas**. En
#   esta tarea las pistas léxicas pesan mucho, y con 7585 problemas de entrenamiento la ventaja de
#   un *encoder* moderno es moderada.
# - Las etiquetas con vocabulario propio (`trees`, `strings`, `combinatorics`) se predicen bien; las
#   que dependen de la idea de la solución o son intercambiables con otras (`dsu`, `two pointers`,
#   `binary search`) son las más difíciles. El rating se predice mejor en el rango medio que en los
#   extremos, sobre todo en los problemas más difíciles, por regresión a la media.
#
# ## 6.2. Diferencias con los notebooks de clase
#
# | | Notebook 6 (BERT) | Notebook 7 (QLoRA) | Este proyecto |
# |---|---|---|---|
# | **Tarea** | Análisis de sentimiento (Yelp, 1–5 estrellas) | Seguimiento de instrucciones (Guanaco) | Etiquetas algorítmicas + dificultad de problemas de Codeforces |
# | **Tipo de salida** | Una clase de 5 (multiclase) | Texto libre | Multi-etiqueta (20 etiquetas) + regresión |
# | **Modelo** | BERT-base-cased y RoBERTa-base | Qwen2.5-0.5B (base) | ModernBERT-base |
# | **Técnica de ajuste** | Ajuste fino completo | QLoRA (4 bits + LoRA r=16) con `SFTTrainer` | Ajuste fino completo con `Trainer` y modelo propio |
# | **Cabeza y pérdida** | `AutoModelForSequenceClassification`, entropía cruzada | Cabeza de LM, pérdida de LM | Cabeza doble propia; BCE ponderada + MSE enmascarado |
# | **Longitud de contexto** | `padding="max_length"` (512) | 1024 | 1024, con padding dinámico y agrupación por longitud |
# | **Datos y división** | 1000 ejemplos aleatorios de train y de test | 1000 ejemplos, sin validación | ~10 000 problemas, división temporal train/val/test |
# | **Umbrales** | No aplica (argmax) | No aplica | Un umbral por etiqueta ajustado en validación |
# | **Evaluación** | *Accuracy* | Comparación cualitativa de respuestas | F1 micro/macro, mAP, MAE, Spearman y 3 baselines |
#
# ## 6.3. Aportaciones
#
# - **Clasificación multi-etiqueta** con pesos por clase y **umbrales por etiqueta**, en lugar de
#   la clasificación multiclase con `argmax` del notebook 6.
# - **Modelo multitarea propio**: subclases de `PretrainedConfig` y `PreTrainedModel` con dos
#   cabezas y una pérdida combinada, compatibles con `Trainer`, `push_to_hub` y `from_pretrained`.
# - **Contexto largo** (1024 tokens) con un *encoder* moderno, padding dinámico y agrupación por
#   longitud.
# - **Evaluación cuidadosa**: división temporal sin mezclar concursos, deduplicación, umbrales
#   elegidos solo con validación y comparación con el modelo base sin ajustar (*linear probe*).
