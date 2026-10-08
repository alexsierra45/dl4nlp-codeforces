# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   accelerator: GPU
#   colab:
#     gpuType: T4
# ---

# %% [markdown]
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
#    (icono 🔑 de la barra lateral). Sin token el notebook funciona igual, pero no sube nada al Hub.
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
# Tiempo aproximado en una T4: <!-- TODO: completar tras ejecución en Colab -->

# %% [markdown]
# # 1. Datos del alumno y título del proyecto
#
# - **Alumno:** Alex Sierra Alcalá
# - **Asignatura:** Deep Learning for NLP — Máster Universitario en Inteligencia Artificial
# - **Proyecto:** *Predicción de etiquetas algorítmicas y dificultad de problemas de Codeforces
#   mediante ajuste fino multitarea de ModernBERT*
# - **Modelo ajustado en el Hub:** [AlexSA45/modernbert-codeforces-tags-rating](
#   https://huggingface.co/AlexSA45/modernbert-codeforces-tags-rating)

# %% [markdown]
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
#   Codeforces superan los 512 tokens (se cuantifica en la sección 3).
# - Tiene 22 capas y **149 millones de parámetros**.
#
# Un *encoder* encaja mejor que un LLM generativo en esta tarea: la salida es un vector de
# probabilidades y un número, no texto.
#
# ## 2.3. Por qué ajuste fino completo (y no LoRA)
#
# Con unos 150M de parámetros, el modelo, sus gradientes y los estados del optimizador caben en los
# 16 GB de una T4 en precisión mixta. Técnicas como LoRA o QLoRA (notebook 7 de clase) están
# pensadas para modelos de miles de millones de parámetros que no caben en memoria; aquí no son
# necesarias, y el ajuste fino completo permite adaptar todas las capas a un dominio muy técnico.
#
# ## 2.4. Por qué multitarea
#
# Etiquetas y dificultad están relacionadas: algunas técnicas aparecen sobre todo en problemas
# difíciles y otras en problemas fáciles (se comprueba en la sección 3). Compartir el *encoder*
# entre las dos tareas permite aprovechar esa relación, y se obtiene un único modelo que resuelve
# ambas con una sola pasada.

# %% [markdown]
# ## 2.5. Preparación del entorno
#
# Instalamos las librerías necesarias. Se fijan las versiones del ecosistema Hugging Face con las
# que se ha probado el notebook, porque su API cambia entre versiones.
#
# - `transformers`: modelos, tokenizadores y la clase `Trainer`.
# - `datasets`: contenedor `Dataset` usado por el `Trainer`.
# - `accelerate`: abstracción del hardware usada internamente por el `Trainer`.
# - `huggingface_hub`: lectura del dataset y subida del modelo al Hub.
# - `scikit-learn`: baselines clásicos y métricas.

# %%
# %pip install -q transformers==5.19.0 datasets==5.1.0 accelerate==1.15.0
# %pip install -q huggingface_hub==1.33.0 scikit-learn pyarrow seaborn

# %% [markdown]
# Importaciones, versiones y semillas. `transformers.set_seed` fija la semilla de `random`, `numpy`
# y `torch` para que los resultados sean reproducibles.

# %%
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

# Salidas limpias: sin avisos repetitivos ni barras de progreso de cada `map`
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
    print("No se ha detectado GPU. En Colab: Entorno de ejecución > Cambiar tipo > GPU T4.")

# %% [markdown]
# ## 2.6. Configuración
#
# Todos los hiperparámetros y *flags* están en esta celda. `SMOKE_TEST` reduce el problema a unas
# decenas de ejemplos y unos pocos pasos de entrenamiento; se usó solo para comprobar en local que
# el notebook se ejecuta de principio a fin (se activa con la variable de entorno
# `CF_SMOKE_TEST=1`). En Colab queda desactivado.

# %%
CONFIG = {
    # Flags de ejecución
    "SMOKE_TEST": os.environ.get("CF_SMOKE_TEST") == "1",
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

# Prueba rápida: pocos ejemplos, secuencias cortas y pocos pasos
SMOKE_SIZES = {"train": 64, "val": 32, "test": 32}
if CONFIG["SMOKE_TEST"]:
    CONFIG.update(MAX_LEN=128, PUSH_TO_HUB=False, RUN_TRAINING=True)
MAX_STEPS = 5 if CONFIG["SMOKE_TEST"] else -1

SEED = CONFIG["SEED"]
set_seed(SEED)
print(json.dumps(CONFIG, indent=1))

# %% [markdown]
# ## 2.7. Autenticación en Hugging Face Hub
#
# El token se lee del secreto `HF_TOKEN` de Colab (o de la variable de entorno del mismo nombre) y
# nunca se escribe en el código. Si no hay token, `PUSH_TO_HUB` pasa a `False` y el notebook
# continúa: los datos y el modelo de partida son públicos.

# %%
HF_TOKEN = os.environ.get("HF_TOKEN")
try:
    from google.colab import userdata  # solo existe en Colab
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

# %% [markdown]
# # 3. Dataset
#
# ## 3.1. Fuente
#
# Se usa [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces), publicado por
# Hugging Face dentro del proyecto Open-R1. Contiene algo más de 10 000 problemas únicos de
# Codeforces, desde 2010 hasta principios de 2025, con el enunciado dividido en campos, sus
# etiquetas y su rating. La *dataset card* declara la licencia **CC-BY-4.0** en los metadatos y
# **ODC-By 4.0** en el texto; ambas permiten el uso con atribución.
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

# %%
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

# %% [markdown]
# ## 3.2. Construcción del texto de entrada
#
# Unimos todos los *splits* originales (volveremos a dividir por fecha en la sección 3.4) y
# construimos un único texto por problema, en este **orden deliberado**:
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

# %%
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
        if value:  # los campos vacíos (p. ej. Interaction) se omiten
            lines.append(f"{name}: {value}")
    return "\n".join(lines)


raw["text"] = raw.apply(build_text, axis=1)

example = raw[raw["input_format"].fillna("").str.contains(r"\\le")].iloc[0]
print("ANTES (input_format):\n", example["input_format"][:400])
print("\nDESPUÉS:\n", clean_latex(example["input_format"])[:400])
print("\nTEXTO COMPLETO (primeros 700 caracteres):\n", example["text"][:700])

# %% [markdown]
# ## 3.3. Limpieza y deduplicación
#
# - Se eliminan los problemas **sin enunciado** (`description` vacía).
# - **Duplicados por alias:** los problemas compartidos entre la Div. 1 y la Div. 2 se registran en
#   `aliases`. Se agrupan con una clave común (el menor identificador del grupo) y se conserva la
#   aparición más antigua.
# - **Duplicados por texto:** se calcula un *hash* del texto normalizado (minúsculas y sin
#   espacios) y también se conserva solo la aparición más antigua.
#
# Si un mismo enunciado apareciera en train y en test, el modelo podría memorizarlo y la
# evaluación sería demasiado optimista; la deduplicación evita esa fuga.

# %%
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

# %% [markdown]
# ## 3.4. División temporal
#
# La división es **por fecha del concurso, nunca aleatoria**: el 80 % más antiguo para train, el
# 10 % siguiente para validación y el 10 % más reciente para test. Así se entrena con problemas
# antiguos y se evalúa con problemas nuevos, igual que ocurriría al usar el modelo en un concurso
# futuro. Una división aleatoria mezclaría problemas del mismo concurso, que comparten autores y
# estilo, entre train y test.

# %%
dates = df["date"].sort_values().reset_index(drop=True)
val_cut = dates.iloc[int(0.8 * len(df))]
test_cut = dates.iloc[int(0.9 * len(df))]
df["split"] = np.where(df["date"] < val_cut, "train",
                       np.where(df["date"] < test_cut, "val", "test"))

# Comprobación: ningún concurso queda repartido entre dos splits
assert (df.groupby("contest_id")["split"].nunique() == 1).all()
print("Corte val :", val_cut)
print("Corte test:", test_cut)

# %% [markdown]
# ## 3.5. Vocabulario de etiquetas y normalización del rating
#
# Todo lo que se "aprende" de los datos se calcula **solo con train**:
#
# - **Vocabulario:** etiquetas con al menos `MIN_TAG_COUNT` apariciones en train, hasta un máximo
#   de `TOP_K_TAGS`. Las etiquetas muy raras (`2-sat`, `schedules`...) tienen demasiado pocos
#   ejemplos para aprenderlas. Se descartan los problemas que se quedan sin ninguna etiqueta del
#   vocabulario.
# - **Rating:** se normaliza con *z-score* usando la media y la desviación típica de train. Los
#   problemas sin rating se conservan (sirven para las etiquetas) con `rating_mask = 0`, de modo
#   que no cuentan ni en la pérdida ni en las métricas de rating.

# %%
train_tag_counts = collections.Counter(
    t for ts in df.loc[df["split"] == "train", "tags_clean"] for t in ts)
TAGS = [t for t, c in train_tag_counts.most_common()
        if c >= CONFIG["MIN_TAG_COUNT"]][:CONFIG["TOP_K_TAGS"]]
TAG2IDX = {t: i for i, t in enumerate(TAGS)}
N_TAGS = len(TAGS)
print(f"{N_TAGS} etiquetas conservadas:", TAGS)

df["tags_kept"] = df["tags_clean"].map(lambda ts: [t for t in ts if t in TAG2IDX])
no_tags = df["tags_kept"].map(len) == 0
print("Problemas sin ninguna etiqueta conservada (eliminados):",
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

# Prueba rápida: muestras pequeñas (el vocabulario ya se calculó con todo train)
if CONFIG["SMOKE_TEST"]:
    df = pd.concat([df[df["split"] == s].sample(n, random_state=SEED)
                    for s, n in SMOKE_SIZES.items()]).reset_index(drop=True)

SPLITS = ["train", "val", "test"]
data = {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}

# %% [markdown]
# ## 3.6. Análisis exploratorio
#
# ### Tamaño y rango de fechas de cada split

# %%
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

# %% [markdown]
# <!-- TODO (autor): comentario sobre tamaños y fechas de los splits -->

# %% [markdown]
# Definimos un estilo común para las figuras: cada split y cada modelo tiene siempre el mismo
# color.

# %%
COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
MODEL_COLORS = {"ModernBERT ajustado": "#2a78d6", "TF-IDF + LR": "#eb6834",
                "Linear probe": "#1baf7a"}
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e5e3", "grid.linewidth": 0.8,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.titleweight": "bold", "axes.axisbelow": True,
})

# %% [markdown]
# ### Frecuencia de las etiquetas

# %%
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

# %% [markdown]
# <!-- TODO (autor): comentario sobre el desbalance de etiquetas -->

# %% [markdown]
# ### Número de etiquetas por problema y distribución del rating

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
max_tags = max(int(d["tags_kept"].map(len).max()) for d in data.values())
bins = np.arange(0.5, max_tags + 1.5)
for s in SPLITS:
    axes[0].hist(data[s]["tags_kept"].map(len), bins=bins, density=True, histtype="step",
                 linewidth=2, color=COLORS[s], label=s)
    axes[1].hist(data[s]["rating"].dropna(), bins=np.arange(750, 3600, 100), density=True,
                 histtype="step", linewidth=2, color=COLORS[s], label=s)
axes[0].set(title="Etiquetas por problema", xlabel="Nº de etiquetas", ylabel="Proporción")
axes[1].set(title="Rating por split", xlabel="Rating", ylabel="Densidad")
for ax in axes:
    ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# <!-- TODO (autor): ¿cambia la distribución del rating con el tiempo? -->

# %% [markdown]
# ### Longitud en tokens
#
# Tokenizamos los textos completos (sin truncar) con el tokenizador de ModernBERT para decidir
# `MAX_LEN` con datos y medir cuánto texto se pierde al truncar.

# %%
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

# %% [markdown]
# <!-- TODO (autor): % de textos truncados con 512 y con 1024 tokens -->

# %% [markdown]
# ### Rating por etiqueta
#
# Si las etiquetas se relacionan con la dificultad, compartir el *encoder* entre las dos tareas
# tiene sentido: es la motivación del enfoque multitarea.

# %%
rated = data["train"][data["train"]["rating_mask"] == 1]
by_tag = rated[["tags_kept", "rating"]].explode("tags_kept")
order = by_tag.groupby("tags_kept")["rating"].median().sort_values().index
fig, ax = plt.subplots(figsize=(8, 7))
sns.boxplot(data=by_tag, y="tags_kept", x="rating", order=order, color="#9cc3ef",
            fliersize=1.5, linewidth=1, ax=ax)
ax.set(title="Rating por etiqueta (train)", xlabel="Rating", ylabel="")
plt.tight_layout()
plt.show()

# %% [markdown]
# <!-- TODO (autor): etiquetas más fáciles y más difíciles -->

# %% [markdown]
# ## 3.7. Tokenización y datasets para el `Trainer`
#
# Tokenizamos con `truncation=True` y `max_length=MAX_LEN`, **sin padding**: el relleno se añade en
# cada lote, solo hasta la longitud del texto más largo del lote (*padding dinámico*), lo que
# ahorra mucho cómputo frente al `padding="max_length"` del notebook 6. Guardamos también la
# longitud (`length`) para agrupar textos de longitud parecida en el mismo lote.

# %%
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

# %% [markdown]
# Subimos al Hub el dataset ya procesado (texto limpio, etiquetas conservadas, rating y split), para
# poder reutilizarlo sin repetir el preprocesamiento.

# %%
if PUSH_TO_HUB:
    keep = ["id", "contest_id", "contest_name", "date", "title", "text", "tags_kept", "rating",
            "split"]
    processed = datasets.DatasetDict({
        s: Dataset.from_pandas(data[s][keep].astype({"date": str}), preserve_index=False)
        for s in SPLITS})
    processed.push_to_hub(CONFIG["HUB_DATASET_ID"])
    print("Dataset procesado subido a", CONFIG["HUB_DATASET_ID"])

# %% [markdown]
# # 4. Implementación
#
# ## 4.1. Arquitectura del modelo
#
# Los notebooks de clase usan cabezas ya hechas (`AutoModelForSequenceClassification`,
# `AutoModelForCausalLM`). Aquí **no existe una clase estándar** que resuelva clasificación
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
# - Heredamos de `PretrainedConfig` y `PreTrainedModel`, con lo que funcionan `save_pretrained`,
#   `push_to_hub` y `from_pretrained`. La configuración guarda todo lo necesario para usar el modelo
#   sin el notebook: nombres de las etiquetas, `pos_weight`, media y desviación del rating y
#   umbrales de decisión por etiqueta.
# - En `__init__` el *encoder* se crea **vacío** con `AutoModel.from_config`; los pesos
#   preentrenados se cargan solo en `from_base`, al empezar el entrenamiento. Así, al cargar el
#   modelo ya ajustado con `from_pretrained`, no se descargan los pesos dos veces.
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

# %%
class CFMultiTaskConfig(PretrainedConfig):
    """Configuración del modelo multitarea (se guarda en config.json en el Hub)."""

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
        # Encoder sin pesos preentrenados: se rellenan con from_base o from_pretrained.
        # La T4 no soporta FlashAttention 2, así que usamos la atención SDPA de PyTorch.
        self.encoder = AutoModel.from_config(enc_config, attn_implementation="sdpa")
        hidden = enc_config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.tags_head = nn.Linear(hidden, config.n_tags)
        self.rating_head = nn.Linear(hidden, 1)
        pos_weight = config.pos_weight or [1.0] * config.n_tags
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        self.post_init()

    def _init_weights(self, module):
        # Solo inicializamos nuestras cabezas; el encoder usa su propia inicialización
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
        # Mean pooling: media de los vectores de los tokens reales (sin padding)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        tag_logits = self.tags_head(pooled)
        rating_pred = self.rating_head(pooled).squeeze(-1)

        loss = None
        if labels_tags is not None:
            # Calculamos la pérdida en float32 para evitar problemas numéricos con fp16
            bce = F.binary_cross_entropy_with_logits(
                tag_logits.float(), labels_tags.float(), pos_weight=self.pos_weight)
            m = rating_mask.float()
            mse = (m * (rating_pred.float() - labels_rating.float()) ** 2).sum() / m.sum().clamp(
                min=1.0)
            loss = bce + self.config.lambda_rating * mse
        return CFMultiTaskOutput(loss=loss, tag_logits=tag_logits, rating_pred=rating_pred)


# pos_weight por etiqueta, calculado solo con train: w_k = min(neg_k / pos_k, 10)
Y_train = np.stack(data["train"]["y_tags"])
pos = Y_train.sum(axis=0)
POS_WEIGHT = np.minimum((len(Y_train) - pos) / np.maximum(pos, 1),
                        CONFIG["POS_WEIGHT_CLIP"]).round(3).tolist()
pd.DataFrame({"positivos en train": pos.astype(int), "pos_weight": POS_WEIGHT}, index=TAGS).T

# %% [markdown]
# ## 4.3. *Collator* y métricas
#
# El *collator* forma cada lote: rellena `input_ids` y `attention_mask` hasta la longitud máxima
# del lote y apila las tres etiquetas en tensores `float32`.
#
# Las métricas se definen una sola vez y se usan para **todos** los modelos (el ajustado y los
# baselines), para que la comparación sea justa:
#
# - **Etiquetas:** F1 micro (dominado por las etiquetas frecuentes), F1 macro (media por etiqueta,
#   sensible a las raras), F1 por muestra, mAP macro (calidad del orden de las probabilidades,
#   independiente del umbral) y *subset accuracy* (acertar exactamente el conjunto de etiquetas).
# - **Rating** (en la escala original y solo con `rating_mask = 1`): MAE, RMSE, correlación de
#   Spearman y % de predicciones con error absoluto ≤ 200.

# %%
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
    has_pos = y_true.sum(axis=0) > 0  # el mAP no está definido para etiquetas sin positivos
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

# %% [markdown]
# ## 4.4. Hiperparámetros
#
# | Parámetro | Valor | Justificación |
# |---|---|---|
# | `max_length` | 1024 | Cubre la mayoría de enunciados (sección 3.6); el resto se trunca por el final |
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

# %%
training_args = TrainingArguments(
    output_dir="runs/main",
    num_train_epochs=CONFIG["EPOCHS"],
    max_steps=MAX_STEPS,
    learning_rate=CONFIG["LR"],
    per_device_train_batch_size=CONFIG["BATCH_SIZE"],
    per_device_eval_batch_size=2 * CONFIG["BATCH_SIZE"],
    gradient_accumulation_steps=CONFIG["GRAD_ACCUM"],
    warmup_steps=CONFIG["WARMUP_RATIO"],          # float < 1: proporción de pasos
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
    logging_steps=1 if CONFIG["SMOKE_TEST"] else 50,
    report_to="none",
    seed=SEED,
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=CONFIG["HUB_MODEL_ID"] if PUSH_TO_HUB else None,
    hub_strategy="every_save",
)

# %% [markdown]
# ## 4.5. Ajuste fino
#
# Con `RUN_TRAINING = True` se crea el modelo a partir de ModernBERT-base y se entrena; con
# `False` se descarga el ya ajustado desde el Hub junto con el historial del entrenamiento. Al
# terminar, el `Trainer` recarga los pesos de la **mejor época** según el F1 macro de validación.

# %%
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

# %% [markdown]
# ### Curvas de entrenamiento

# %%
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
# <!-- TODO (autor): tiempo real de entrenamiento, época elegida, ¿hay sobreajuste? -->

# %% [markdown]
# # 5. Resultados y discusión
#
# ## 5.1. Predicciones y ajuste de umbrales
#
# El modelo devuelve una probabilidad por etiqueta. Usar el mismo umbral (0.5) para todas no es
# óptimo: las etiquetas raras suelen necesitar umbrales distintos. Ajustamos un **umbral por
# etiqueta en validación** (rejilla de 0.05 a 0.95 con paso 0.05, maximizando el F1 de cada
# etiqueta) y lo aplicamos sin cambios a test. Test se usa una única vez, para el informe final.
#
# Para la inferencia ordenamos los textos por longitud: los lotes tienen menos relleno y la
# predicción es más rápida.

# %%
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

# %% [markdown]
# Guardamos los umbrales en la configuración del modelo y subimos la versión final al Hub, junto
# con el historial del entrenamiento (necesario para dibujar las curvas con `RUN_TRAINING=False`).

# %%
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

# %% [markdown]
# ### Uso del modelo publicado
#
# Esta celda carga el modelo **desde el Hub** y predice las etiquetas y la dificultad de un
# enunciado de test. Es todo lo que necesita un usuario del modelo: el tokenizador, la clase
# `CFMultiTaskModel` y los umbrales guardados en la configuración.

# %%
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
else:  # sin acceso de escritura al Hub, se usa el modelo recién entrenado
    demo_model = model
sample = data["test"].iloc[0]
pred_tags, pred_rating = predict_problem(demo_model, sample["text"])
print(sample["text"][:500], "...\n")
print("Etiquetas reales :", sample["tags_kept"], "| rating real:", sample["rating"])
print("Etiquetas predichas:", pred_tags, f"| rating predicho: {pred_rating:.0f}")
if demo_model is not model:
    del demo_model

# %% [markdown]
# ## 5.2. Baselines
#
# Para saber si el ajuste fino aporta algo hay que compararlo con referencias razonables. Todas se
# evalúan sobre el **mismo test**, con las **mismas métricas** y con umbrales por etiqueta
# ajustados en validación:
#
# - **B0, trivial.** Cada etiqueta recibe como puntuación su frecuencia en train y el rating es
#   siempre la media de train. Es el mínimo que cualquier modelo debe superar.
# - **B1, TF-IDF + modelos lineales.** Bolsa de unigramas y bigramas con regresión logística
#   (una por etiqueta, `class_weight="balanced"`, con `C` elegido en validación) y regresión
#   `Ridge` para el rating. Es el baseline clásico en clasificación de textos.
# - **B2, ModernBERT sin ajustar (*linear probe*).** Es el **modelo base sin ajustar**: se extraen
#   los *embeddings* de ModernBERT-base **congelado** (mismo *mean pooling* y `MAX_LEN`) y se
#   entrenan encima los mismos modelos lineales. Mide cuánto aporta ajustar el *encoder* frente a
#   usar solo sus representaciones preentrenadas.

# %%
results = {}  # nombre -> puntuaciones, ratings y umbrales de val y test


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

# %% [markdown]
# Para el *linear probe* extraemos los *embeddings* de ModernBERT-base congelado, en fp16 y sin
# gradientes, con el mismo *mean pooling* que el modelo ajustado.

# %%
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

# %% [markdown]
# ## 5.3. Tabla principal de resultados
#
# Todos los modelos sobre el **test completo**, con umbrales ajustados en validación.

# %%
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
# <!-- TODO (autor): discusión de la tabla principal -->

# %% [markdown]
# ## 5.4. Efecto del ajuste de umbrales

# %%
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
# <!-- TODO (autor): cuánto aportan los umbrales -->

# %% [markdown]
# ## 5.5. Análisis por etiqueta

# %%
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
# <!-- TODO (autor): ¿qué etiquetas se predicen bien y cuáles mal? ¿influye la frecuencia? -->

# %% [markdown]
# ## 5.6. Predicción del rating

# %%
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
# <!-- TODO (autor): ¿en qué rangos de dificultad se equivoca más el modelo? -->

# %% [markdown]
# ## 5.7. Análisis de errores
#
# Ejemplos de aciertos y fallos del modelo ajustado en test, ordenados por el F1 de cada problema
# (el F1 entre sus etiquetas reales y las predichas).

# %%
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
# <!-- TODO (autor): patrones en los errores -->

# %% [markdown]
# ## 5.8. Limitaciones
#
# - **Ruido en las etiquetas.** Las etiquetas las asignan los autores y la comunidad, y un problema
#   puede admitir varias soluciones válidas con técnicas distintas: parte del "error" no es tal.
# - **Desbalance.** Las etiquetas raras tienen pocos ejemplos; `pos_weight` y los umbrales por
#   etiqueta lo mitigan, pero no lo eliminan.
# - **Dificultad intrínseca del rating.** El rating depende de cómo se comportaron los
#   participantes en el concurso, algo que el enunciado no refleja del todo.
# - **Truncamiento.** Los textos más largos que `MAX_LEN` pierden su parte final.
# - **Solo inglés** y solo problemas de Codeforces.

# %% [markdown]
# # 6. Conclusiones
#
# ## 6.1. Resumen de resultados
#
# <!-- TODO (autor): resumen de resultados -->
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
#
# ## 6.4. Trabajo futuro
#
# - Probar `ModernBERT-large` (395M de parámetros).
# - Añadir como entrada el código de soluciones aceptadas (`open-r1/codeforces-submissions`).
# - Comparar con un LLM generalista sin ajustar (*zero-shot*).
#
# # 7. Referencias
#
# - Warner, B., Chaffin, A., Clavié, B., et al. (2024). *Smarter, Better, Faster, Longer: A Modern
#   Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference*.
#   arXiv:2412.13663.
# - Model card de [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base).
# - Dataset card de [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces).
# - Documentación de Hugging Face
#   [`transformers` (Trainer)](https://huggingface.co/docs/transformers/main_classes/trainer)
#   y de [scikit-learn](https://scikit-learn.org/stable/).
# - Serrano, E. Notebooks de clase *6. GenAI&LMs Fine-Tuning BERT* y *7. GenAI&LMs QLoRA*,
#   Universidad Politécnica de Madrid.
