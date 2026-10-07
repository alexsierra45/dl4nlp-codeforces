# DL4NLP: etiquetas algorítmicas y dificultad de problemas de Codeforces

Ajuste fino multitarea de [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
para predecir, a partir del enunciado de un problema de Codeforces, sus **etiquetas
algorítmicas** (clasificación multi-etiqueta) y su **rating de dificultad** (regresión).

Proyecto individual de la asignatura *Deep Learning for NLP* (MUIA). Autor: Alex Sierra Alcalá.

## Ficheros

| Fichero | Contenido |
|---|---|
| `Sierra_Alcala_Alex.ipynb` | Notebook entregable (autocontenido, pensado para Colab con GPU T4) |
| `notebook.py` | Fuente del notebook en formato jupytext *percent* |
| `requirements-local.txt` | Versiones usadas en la prueba local |

## Uso

Abrir `Sierra_Alcala_Alex.ipynb` en Google Colab, seleccionar una GPU T4 y ejecutar todo. Los datos
([`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces)) y los modelos se
descargan del Hugging Face Hub.

Para regenerar el notebook desde la fuente y hacer la prueba rápida en local:

```bash
jupytext --to ipynb notebook.py -o Sierra_Alcala_Alex.ipynb
CF_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute Sierra_Alcala_Alex.ipynb --output smoke.ipynb
```
