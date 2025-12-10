# 🚀 Inicio Rápido - Tennis ML Predictor

¿Primera vez usando este proyecto? Esta guía te llevará de 0 a modelo entrenado en **30-40 minutos**.

---

## ⚡ Opción A: Pipeline Completo (Recomendado)

**Para**: Usuarios nuevos que clonan el repositorio por primera vez

### Paso 1: Clonar y Preparar

```bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/tennis-ml-predictor.git
cd tennis-ml-predictor

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2: Ejecutar Pipeline Completo

```bash
python setup_and_train.py --full
```

**¿Qué hace este comando?**
1. ✅ Verifica dependencias
2. ✅ Crea estructura de carpetas
3. ✅ Descarga datos (TML Database ~25,000 partidos)
4. ✅ Procesa datos
5. ✅ Genera 149 features avanzadas
6. ✅ Entrena 4 modelos (RF, XGBoost, GB, LR)
7. ✅ Optimiza hiperparámetros
8. ✅ Calibra probabilidades
9. ✅ Ejecuta Walk-Forward Validation
10. ✅ Genera reportes

**Tiempo estimado**: 30-40 minutos

**Resultado**: Modelo entrenado, validado y listo para usar

---

## 🎯 Opción B: Solo Entrenar

**Para**: Ya tienes los datos descargados, solo quieres re-entrenar

```bash
python setup_and_train.py --train-only
```

**Tiempo**: ~20 minutos

---

## 🔍 Opción C: Solo Validar

**Para**: Ya tienes el modelo entrenado, solo quieres validar

```bash
python setup_and_train.py --validate-only
```

**Tiempo**: ~15 minutos

---

## 📊 Verificar Resultados

Después de ejecutar el pipeline, verifica:

### 1. Modelos Entrenados
```bash
ls modelos/
# Deberías ver:
# - random_forest_calibrado.pkl
# - xgboost_calibrado.pkl
# - gradient_boosting_calibrado.pkl
# - logistic_regression_calibrado.pkl
```

### 2. Reportes Generados
```bash
ls resultados/
# Deberías ver:
# - REPORTE_FASE_2.html (abre en navegador)
# - walk_forward/REPORTE_VALIDACION_FINAL.txt
```

### 3. Métricas Esperadas

Abre `resultados/walk_forward/REPORTE_VALIDACION_FINAL.txt`:

```
Accuracy (último fold): ~71-72%
Brier Score: ~0.19
Tendencia: IMPROVING
```

---

## 🎮 Usar el Modelo

Una vez entrenado, usa el modelo para predicciones:

```python
from predictor_calibrado import PredictorCalibrado

# Cargar modelo
predictor = PredictorCalibrado("modelos/random_forest_calibrado.pkl")

# Hacer predicción (ejemplo con features)
# Ver README.md para detalles completos
```

---

## ❓ Troubleshooting

### Error: "ModuleNotFoundError"
**Solución**: 
```bash
pip install -r requirements.txt
```

### Error: "No such file or directory: datos/..."
**Solución**: Ejecuta con `--full` para descargar datos primero

### Pipeline muy lento
**Solución**: Es normal, el entrenamiento tarda ~30-40 minutos

### Error en descarga de datos
**Solución**: Verifica conexión a internet, TML Database requiere conexión

---

## 📚 Siguiente Paso

Una vez completado el pipeline:

1. **Ver resultados**: Abre `resultados/REPORTE_FASE_2.html` en tu navegador
2. **Leer documentación**: Ver `README.md` para uso avanzado
3. **Explorar código**: Ver `src/` para entender implementación
4. **Hacer predicciones**: Ver sección "Uso del Modelo" en README

---

## 🆘 ¿Necesitas Ayuda?

- **Documentación completa**: Ver `README.md`
- **Paso a paso detallado**: Ver sección "Pipeline Completo - Paso a Paso" en README
- **Resultados Fase 3**: Ver `FASE_3_RESULTADOS.md`
- **Issues**: Abre un issue en GitHub

---

**¡Listo!** En 30-40 minutos tendrás un modelo de predicción de tenis completamente funcional 🎾
