# Scripts Internos - Tennis ML Predictor

## 📁 Scripts de uso ocasional o interno

Esta carpeta contiene scripts que se usan ocasionalmente o que son llamados internamente por otros scripts.

## Scripts Disponibles

### 🔄 Actualización y Tracking
- **`actualizar_tracking.py`** - Actualiza el sistema de tracking con resultados reales
  ```bash
  python scripts/internal/actualizar_tracking.py
  ```

### 📊 Análisis y Reportes
- **`backtesting_fase2.py`** - Ejecuta backtesting con diferentes umbrales
  ```bash
  python scripts/internal/backtesting_fase2.py
  ```

- **`generar_reporte_fase2.py`** - Genera reportes HTML de resultados
  ```bash
  python scripts/internal/generar_reporte_fase2.py
  ```

### 🤖 Entrenamiento y Optimización
- **`run_fase3_optimization.py`** - Optimización de hiperparámetros
  ```bash
  python scripts/internal/run_fase3_optimization.py
  ```
  > **Nota**: Este script ya está incluido en `setup_and_train.py`

- **`run_feature_engineering_fase3.py`** - Feature engineering completo
  ```bash
  python scripts/internal/run_feature_engineering_fase3.py
  ```
  > **Nota**: Este script ya está incluido en `setup_and_train.py`

### ✅ Validación
- **`walk_forward_validation.py`** - Walk-Forward Validation
  ```bash
  python scripts/internal/walk_forward_validation.py
  ```
  > **Nota**: También disponible en `validate.py --component walkforward`

## 💡 Cuándo usar estos scripts

- **Desarrollo**: Cuando necesitas ejecutar una fase específica del pipeline
- **Debugging**: Para probar componentes individuales
- **Análisis**: Para generar reportes o análisis específicos
- **Mantenimiento**: Para actualizar tracking o regenerar features

## 🎯 Scripts Principales

Para uso general, usa los scripts principales en la raíz:

- **`setup_and_train.py`** - Pipeline completo de entrenamiento
- **`validate.py`** - Todas las validaciones
- **`demo.py`** - Todas las demos

```bash
# Ver opciones
python setup_and_train.py --help
python validate.py --help
python demo.py --help
```

## 📚 Documentación

Para más información sobre cada script, consulta:
- `README.md` en la raíz del proyecto
- `QUICK_START.md` para guía rápida
- Comentarios dentro de cada script
