# Sistema de Clasificación de Buzón Institucional

## Arquitectura del Proyecto

Este proyecto implementa un **pipeline de clasificación en dos etapas** para comentarios del buzón institucional (quejas, sugerencias, felicitaciones):

1. **Clasificación de Intención** (`clasificacion_intencion/`): Modelo TF-IDF + Regresión Logística que clasifica comentarios en: queja, sugerencia, o felicitación
2. **Clasificación de Departamento** (`clasificacion_departamentos/`): Sistema multi-estrategia usando SetFit + FlashText + Anchoring semántico para enrutar al departamento responsable

### Flujo de Datos
```
Comentario → [Intención] → [Departamento] → Enrutamiento
```

## Estructura de Modelos

### `/models/`
- **`clasificador_departamentos_setfit/`**: Modelo SetFit entrenado (paraphrase-multilingual-MiniLM-L12-v2)
- **`clasificador_departamentos_config.json`**: Mapeos label2id/id2label, anclas semánticas, umbral de confianza (0.35)
- **`clasificador_encoder_onnx/`**: Exportación ONNX del encoder (para optimización CPU)
- TF-IDF vectorizer y clasificador pkl (en `clasificacion_intencion/`)

## Patrones Específicos del Proyecto

### 1. Sistema de Clasificación Multi-Estrategia (Departamentos)

El clasificador de departamentos usa **3 capas** en cascada:

```python
# 1. FlashText: O(n) keyword matching instantáneo
dept, conf = clasificar_por_keywords(texto)
if conf == 1.0:
    return dept  # Match directo

# 2. SetFit: Few-shot learning con contrastive learning
prediccion = model.predict([texto])[0]

# 3. Anchoring: Validación semántica con umbrales
embedding_texto = encoder.encode([texto])
similaridad = cosine_similarity(embedding_texto, anclas_embeddings)
if max(similaridad) < UMBRAL_CONFIANZA:  # Default: 0.35
    return "Clasificación incierta"
```

**Razón arquitectónica**: Few-shot learning (SetFit) permite entrenar con SAMPLES_PER_CLASS = 8 ejemplos por departamento, crítico dado el dataset limitado de comentarios institucionales.

### 2. Anclas Semánticas

Ver `clasificacion_departamentos/departamentos_funcion.json` - cada departamento tiene descripciones funcionales concatenadas que sirven como **embeddings de referencia**:

```json
{
  "anclas": {
    "Dirección": "Coordina y da seguimiento a los procesos institucionales. Verifica que los procesos se realicen conforme a normas...",
    "Subdirección Académica": "Selecciona y contrata docentes. Asigna grupos, horarios y salones..."
  }
}
```

Estas anclas se comparan vía similitud coseno para detectar out-of-distribution (OOD) predictions.

### 3. Data Augmentation para Clases Desbalanceadas

En `clasificacion_intencion/TF_IDF.ipynb`, la clase "queja" estaba subrepresentada, por lo que se añadieron **15 ejemplos sintéticos** antes del entrenamiento:

```python
nuevas_quejas = [
    "El sistema presenta fallas constantes",
    "La atención fue muy lenta",
    # ... 13 más
]
```

**Regla**: Al modificar datasets, mantener el balanceo sintético para "queja" o ajustar `class_weight="balanced"` en LogisticRegression.

### 4. FlashText Keywords Dictionary

Los keywords en `KEYWORDS_POR_DEPARTAMENTO` proveen clasificación O(n) para casos obvios:

```python
KEYWORDS_POR_DEPARTAMENTO = {
    "Subdirección Administrativa": ["aire acondicionado", "clima", "limpieza", ...],
    "Gestión Escolar": ["constancia", "certificado", "kardex", ...],
    # ...
}
```

**Al añadir departamentos**: actualizar este diccionario para cubrir términos específicos del dominio.

## Workflows de Desarrollo

### Entrenamiento de Modelo de Departamentos

Ejecutar en orden (en `clasificacion_departamentos/`):

1. **`Preprocesamiento.ipynb`**: Limpia y prepara dataset_comentarios.json
2. **`Clasificacion_departamento.ipynb`**: Entrena SetFit, genera checkpoints/, guarda modelos en `../models/`

Variables clave a ajustar:
- `SAMPLES_PER_CLASS = 8`: Ejemplos por departamento para few-shot
- `UMBRAL_CONFIANZA = 0.35`: Threshold de similitud coseno (menor = más estricto)
- `MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"`: Modelo base optimizado para español

### Entrenamiento de Modelo de Intención

Ejecutar en orden (en `clasificacion_intencion/`):

1. **`limpieza.ipynb`**: Limpia Excel original → `buzon_limpio.csv`
2. **`TF_IDF.ipynb`**: Data augmentation + vectorización → `buzon_balanceado.csv`, `tfidf_vectorizer.pkl`
3. **`modelo.ipynb`**: Entrena LogisticRegression → `clasificador.pkl`

### Exportación ONNX (Opcional)

Para inferencia 3-4x más rápida en CPU:

```bash
optimum-cli export onnx --model models/clasificador_departamentos_setfit/sentence_transformer models/onnx
optimum-cli onnxruntime quantize --onnx_model models/onnx -o models/onnx_int8
```

Reduce tamaño de 420MB → 105MB (INT8 quantization).

## Convenciones de Código

### Rutas de Archivos

Usar `Path` de pathlib para compatibilidad cross-platform:

```python
from pathlib import Path
carpeta_raiz = Path.cwd().parent
ruta_comentarios = Path("dataset_comentarios.json")
```

### Carga de Modelos en Producción

```python
from setfit import SetFitModel
import json

# SetFit
model = SetFitModel.from_pretrained("../models/clasificador_departamentos_setfit")
with open("../models/clasificador_departamentos_config.json") as f:
    config = json.load(f)

# TF-IDF
import joblib
vectorizador = joblib.load("../models/tfidf_vectorizer.pkl")
modelo_intencion = joblib.load("clasificacion_intencion/clasificador.pkl")
```

### Notebooks como Fuente de Verdad

Los notebooks **NO** son experimentales - son el código de producción:
- Ejecutar celdas en orden secuencial
- Las variables del kernel (ver `copilot_getNotebookSummary`) reflejan el estado del entrenamiento
- Checkpoints se guardan automáticamente cada N pasos en `checkpoints/`

## Dependencias Críticas

Instalación completa:

```bash
pip install setfit sentence-transformers flashtext scikit-learn pandas openpyxl joblib
pip install optimum[onnxruntime]  # Solo para exportación ONNX
```

Versiones clave:
- **SetFit**: Framework para few-shot text classification
- **FlashText**: Reemplazo rápido de regex para keyword extraction
- **paraphrase-multilingual-MiniLM-L12-v2**: Modelo multilingüe optimizado para español

## Debugging y Métricas

### Ver Matriz de Confusión

Ambos notebooks incluyen secciones de evaluación con:
- Confusion matrix via `sklearn.metrics.confusion_matrix`
- Classification report con precision/recall/f1 por clase
- Visualizaciones con seaborn heatmaps

### Detectar Clasificaciones Inciertas

Cuando `max(cosine_similarity) < UMBRAL_CONFIANZA`, el modelo rechaza la predicción. Esto evita errores silenciosos en comentarios fuera de distribución.

```python
# Revisar distribución de confianzas
for texto in test_dataset:
    sim = max(cosine_similarity(encoder.encode([texto]), anclas_embeddings)[0])
    print(f"{sim:.3f} - {texto[:50]}")
```

## Datos de Referencia

- **`dataset_comentarios.json`**: ~200 comentarios etiquetados por departamento
- **`departamentos_funcion.json`**: 17 departamentos con descripciones funcionales
- **`buzon_balanceado.csv`**: Dataset aumentado para clasificación de intención
- **17 departamentos válidos**: Ver `clasificador_departamentos_config.json` → `id2label`

## Notas de Integración

Si integras este sistema en producción:
1. Usar ONNX INT8 para reducir latencia en CPU
2. Mantener umbral de confianza configurable (actualmente 0.35)
3. Implementar fallback manual cuando `similaridad < umbral`
4. Loguear predicciones de baja confianza para reentrenamiento futuro
