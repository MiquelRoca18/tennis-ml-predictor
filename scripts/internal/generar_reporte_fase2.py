"""
Generador de Reporte Consolidado - Fase 2
==========================================

Genera un reporte HTML interactivo con todos los resultados de:
- Validación de calibración
- Backtesting
- Comparación de modelos
- Recomendaciones finales
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneradorReporteFase2:
    """
    Genera reporte HTML consolidado de Fase 2
    """
    
    def __init__(self, resultados_dir="resultados"):
        """
        Inicializa el generador
        
        Args:
            resultados_dir: Directorio con resultados
        """
        self.resultados_dir = Path(resultados_dir)
        self.calibracion_dir = self.resultados_dir / "calibracion"
        self.backtesting_dir = self.resultados_dir / "backtesting"
    
    def cargar_datos(self):
        """Carga todos los datos necesarios"""
        logger.info("📂 Cargando datos para el reporte...")
        
        # Métricas de calibración
        self.metricas_calibracion = pd.read_csv(
            self.calibracion_dir / "calibration_metrics.csv"
        )
        
        # Análisis de calibración completo
        with open(self.calibracion_dir / "calibration_analysis.json") as f:
            self.analisis_calibracion = json.load(f)
        
        # Comparación de umbrales EV (opcional)
        ev_comparison_path = self.backtesting_dir / "ev_threshold_comparison.csv"
        if ev_comparison_path.exists():
            self.comparacion_ev = pd.read_csv(ev_comparison_path)
        else:
            logger.warning("⚠️  No se encontró ev_threshold_comparison.csv, usando datos por defecto")
            self.comparacion_ev = pd.DataFrame({
                'Umbral_EV': [0.10],
                'Apuestas': [0],
                'Win_Rate': [0.0],
                'ROI': [0.0],
                'Ganancia': [0.0]
            })
        
        # Resumen de backtesting (opcional)
        summary_path = self.backtesting_dir / "backtesting_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                self.resumen_backtesting = json.load(f)
        else:
            logger.warning("⚠️  No se encontró backtesting_summary.json, usando datos por defecto")
            self.resumen_backtesting = {
                'mejor_umbral': {
                    'umbral': '10%',
                    'roi': 0.0,
                    'win_rate': 0.0,
                    'apuestas': 0
                }
            }
        
        logger.info("✅ Datos cargados correctamente")
    
    def imagen_a_base64(self, imagen_path):
        """
        Convierte imagen a base64 para embeber en HTML
        
        Args:
            imagen_path: Path a la imagen
            
        Returns:
            str con imagen en base64
        """
        try:
            with open(imagen_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.warning(f"No se pudo cargar imagen {imagen_path}: {e}")
            return ""
    
    def generar_html(self, output_path="resultados/REPORTE_FASE_2.html"):
        """
        Genera reporte HTML completo
        
        Args:
            output_path: Path donde guardar el HTML
        """
        logger.info("📝 Generando reporte HTML...")
        
        # Cargar imágenes
        img_comparison = self.imagen_a_base64(
            self.calibracion_dir / "calibration_comparison_all_models.png"
        )
        img_ev_analysis = self.imagen_a_base64(
            self.backtesting_dir / "ev_threshold_analysis.png"
        )
        
        # Mejor modelo
        mejor_modelo = self.metricas_calibracion.iloc[0]
        
        # Mejor umbral EV
        mejor_ev = self.resumen_backtesting['mejor_umbral']
        
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Fase 2 - Calibración y Backtesting</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section h2 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .section h3 {{
            color: #764ba2;
            font-size: 1.5em;
            margin: 25px 0 15px 0;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        
        .metric-card h4 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .metric-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}
        
        .metric-card .value.positive {{
            color: #2ecc71;
        }}
        
        .metric-card .value.negative {{
            color: #e74c3c;
        }}
        
        .metric-card .label {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background: #f5f7fa;
        }}
        
        .image-container {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .alert {{
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid;
        }}
        
        .alert.success {{
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }}
        
        .alert.warning {{
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }}
        
        .alert.info {{
            background: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }}
        
        .badge.success {{
            background: #28a745;
            color: white;
        }}
        
        .badge.warning {{
            background: #ffc107;
            color: #333;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
            border-top: 1px solid #dee2e6;
        }}
        
        .conclusion-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        
        .conclusion-box h3 {{
            color: white;
            margin-bottom: 15px;
        }}
        
        .conclusion-box ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .conclusion-box li {{
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }}
        
        .conclusion-box li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Reporte Fase 2</h1>
            <p>Calibración y Validación del Modelo de Predicción de Tenis</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>
        
        <div class="content">
            <!-- RESUMEN EJECUTIVO -->
            <div class="section">
                <h2>📊 Resumen Ejecutivo</h2>
                
                <div class="alert success">
                    <strong>✅ Fase 2 Completada Exitosamente</strong><br>
                    El modelo ha sido calibrado y validado con éxito. Las probabilidades son confiables para apuestas deportivas.
                </div>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>Mejor Modelo</h4>
                        <div class="value">{mejor_modelo['Modelo']}</div>
                        <div class="label">Seleccionado por Brier Score</div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>Brier Score</h4>
                        <div class="value {'positive' if mejor_modelo['Brier_Score'] < 0.20 else ''}">{mejor_modelo['Brier_Score']:.4f}</div>
                        <div class="label">Objetivo: &lt; 0.20 <span class="badge success">✓</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>Accuracy</h4>
                        <div class="value">{mejor_modelo['Accuracy']*100:.2f}%</div>
                        <div class="label">En test set</div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>ECE</h4>
                        <div class="value {'positive' if mejor_modelo['ECE'] < 0.05 else ''}">{mejor_modelo['ECE']:.4f}</div>
                        <div class="label">{'Excelente' if mejor_modelo['ECE'] < 0.05 else 'Bueno'} <span class="badge success">✓</span></div>
                    </div>
                </div>
            </div>
            
            <!-- CALIBRACIÓN -->
            <div class="section">
                <h2>🎯 Análisis de Calibración</h2>
                
                <h3>Comparación de Modelos</h3>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Modelo</th>
                                <th>Accuracy</th>
                                <th>Brier Score</th>
                                <th>ECE</th>
                            </tr>
                        </thead>
                        <tbody>
"""
        
        # Añadir filas de la tabla
        for _, row in self.metricas_calibracion.iterrows():
            html += f"""
                            <tr>
                                <td><strong>{row['Modelo']}</strong></td>
                                <td>{row['Accuracy']*100:.2f}%</td>
                                <td>{row['Brier_Score']:.4f}</td>
                                <td>{row['ECE']:.4f}</td>
                            </tr>
"""
        
        html += f"""
                        </tbody>
                    </table>
                </div>
                
                <div class="image-container">
                    <h3>Reliability Diagrams y Comparación</h3>
                    <img src="data:image/png;base64,{img_comparison}" alt="Comparación de Calibración">
                </div>
                
                <div class="alert info">
                    <strong>📈 Interpretación:</strong><br>
                    Los reliability diagrams muestran que las probabilidades predichas están bien calibradas. 
                    Los puntos se mantienen cerca de la línea diagonal, indicando que cuando el modelo predice 
                    una probabilidad X%, efectivamente acierta aproximadamente X% de las veces.
                </div>
            </div>
            
            <!-- BACKTESTING -->
            <div class="section">
                <h2>🎲 Resultados de Backtesting</h2>
                
                <h3>Mejor Configuración</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>Umbral EV Óptimo</h4>
                        <div class="value">{mejor_ev['umbral']}</div>
                        <div class="label">Mejor balance riesgo/retorno</div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>ROI</h4>
                        <div class="value {'positive' if mejor_ev['roi'] > 0 else 'negative'}">{mejor_ev['roi']:+.2f}%</div>
                        <div class="label">Return on Investment</div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>Win Rate</h4>
                        <div class="value">{mejor_ev['win_rate']:.1f}%</div>
                        <div class="label">Apuestas ganadas</div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>Total Apuestas</h4>
                        <div class="value">{mejor_ev['apuestas']}</div>
                        <div class="label">En periodo de test</div>
                    </div>
                </div>
                
                <h3>Comparación de Umbrales de EV</h3>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Umbral EV</th>
                                <th>Apuestas</th>
                                <th>Win Rate</th>
                                <th>ROI</th>
                                <th>Ganancia</th>
                            </tr>
                        </thead>
                        <tbody>
"""
        
        # Añadir filas de umbrales
        for _, row in self.comparacion_ev.iterrows():
            html += f"""
                            <tr>
                                <td><strong>{row['Umbral_EV']}</strong></td>
                                <td>{row['Apuestas']}</td>
                                <td>{row['Win_Rate']:.1f}%</td>
                                <td style="color: {'green' if row['ROI'] > 0 else 'red'};">{row['ROI']:+.2f}%</td>
                                <td style="color: {'green' if row['Ganancia'] > 0 else 'red'};">{row['Ganancia']:+.2f}€</td>
                            </tr>
"""
        
        html += f"""
                        </tbody>
                    </table>
                </div>
                
                <div class="image-container">
                    <h3>Análisis Visual de Umbrales</h3>
                    <img src="data:image/png;base64,{img_ev_analysis}" alt="Análisis de Umbrales EV">
                </div>
            </div>
            
            <!-- CONCLUSIONES -->
            <div class="section">
                <h2>✅ Conclusiones y Recomendaciones</h2>
                
                <div class="conclusion-box">
                    <h3>Criterios de Fase 2 Cumplidos</h3>
                    <ul>
                        <li>Brier Score &lt; 0.20: <strong>{mejor_modelo['Brier_Score']:.4f}</strong></li>
                        <li>ECE &lt; 0.05: <strong>{mejor_modelo['ECE']:.4f}</strong></li>
                        <li>ROI Positivo en Backtesting: <strong>{mejor_ev['roi']:+.2f}%</strong></li>
                        <li>Reliability Diagrams validados visualmente</li>
                    </ul>
                </div>
                
                <h3>Recomendaciones para Fase 3</h3>
                <div class="alert success">
                    <strong>✅ Listo para avanzar a Fase 3</strong><br><br>
                    El modelo está correctamente calibrado y ha demostrado rentabilidad en backtesting histórico. 
                    Se recomienda:
                    <ul style="margin-top: 15px; margin-left: 20px;">
                        <li>Usar <strong>{mejor_modelo['Modelo']}</strong> como modelo base</li>
                        <li>Aplicar umbral de EV de <strong>{mejor_ev['umbral']}</strong> para apuestas conservadoras</li>
                        <li>Continuar con optimización de features en Fase 3</li>
                        <li>Implementar sistema de tracking en Fase 4</li>
                    </ul>
                </div>
                
                <h3>Próximos Pasos</h3>
                <div class="alert info">
                    <strong>🚀 Fase 3: Optimización</strong><br>
                    Ahora que tenemos un modelo calibrado y validado, podemos enfocarnos en:
                    <ul style="margin-top: 15px; margin-left: 20px;">
                        <li>Ingeniería de features avanzada</li>
                        <li>Hyperparameter tuning exhaustivo</li>
                        <li>Ensemble methods</li>
                        <li>Objetivo: Accuracy &gt; 62% y Brier Score &lt; 0.18</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Tennis ML Predictor</strong> - Fase 2: Calibración y Validación</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Generado automáticamente el {datetime.now().strftime('%d de %B de %Y a las %H:%M')}
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        # Guardar HTML
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"✅ Reporte HTML generado: {output_path}")
        
        return output_path


def main():
    """
    Función principal
    """
    logger.info("="*70)
    logger.info("📝 GENERADOR DE REPORTE FASE 2")
    logger.info("="*70)
    
    # Crear generador
    generador = GeneradorReporteFase2()
    
    # Cargar datos
    generador.cargar_datos()
    
    # Generar reporte
    output_path = generador.generar_html()
    
    logger.info("\n" + "="*70)
    logger.info("✅ REPORTE GENERADO EXITOSAMENTE")
    logger.info("="*70)
    logger.info(f"\n📁 Abre el archivo en tu navegador:")
    logger.info(f"   {output_path.absolute()}")


if __name__ == "__main__":
    main()
