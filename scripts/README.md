# Scripts Directory - Tennis ML Predictor

## 📁 Organización de Scripts

Esta carpeta contiene scripts organizados por frecuencia de uso y estado.

## Estructura

```
scripts/
├── internal/          # Scripts de uso ocasional o interno
│   ├── actualizar_tracking.py
│   ├── backtesting_fase2.py
│   ├── generar_reporte_fase2.py
│   ├── run_fase3_optimization.py
│   ├── run_feature_engineering_fase3.py
│   ├── walk_forward_validation.py
│   └── README.md
│
└── deprecated/        # Scripts consolidados (mantener para referencia)
    ├── validacion_calibracion.py
    ├── validacion_final_fase3.py
    ├── validacion_kelly_fase5.py
    ├── validacion_fase6.py
    ├── validacion_fase7.py
    ├── demo_tracking_fase4.py
    ├── demo_kelly_fase5.py
    ├── demo_multibookmaker_fase6.py
    ├── demo_fase6_simulado.py
    └── README.md
```

## 🎯 Scripts Principales (en raíz)

Para uso diario, usa los scripts en la raíz del proyecto:

### Pipeline Completo
```bash
python setup_and_train.py          # Pipeline maestro
python setup_and_train.py --help   # Ver opciones
```

### Validación Unificada
```bash
python validate.py --all            # Todas las validaciones
python validate.py --phase 2        # Validación específica
python validate.py --help           # Ver opciones
```

### Demos Unificadas
```bash
python demo.py --all                # Todas las demos
python demo.py --feature kelly      # Demo específica
python demo.py --help               # Ver opciones
```

## 📚 Más Información

- Consulta `internal/README.md` para scripts de uso ocasional
- Consulta `deprecated/README.md` para scripts consolidados
- Consulta el `README.md` principal para documentación completa
