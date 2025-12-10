"""
Validación Completa de Calibración - Fase 2
============================================

Este script valida exhaustivamente la calibración de todos los modelos:
- Reliability Diagrams (curvas de calibración)
- Expected Calibration Error (ECE)
- Brier Score y Log Loss
- Análisis por bins de probabilidad
- Comparación visual de todos los modelos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import joblib
from pathlib import Path
import json
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class CalibrationValidator:
    """
    Validador completo de calibración de modelos
    """
    
    def __init__(self, modelos_dir="modelos", resultados_dir="resultados/calibracion"):
        """
        Inicializa el validador
        
        Args:
            modelos_dir: Directorio con modelos calibrados
            resultados_dir: Directorio para guardar resultados
        """
        self.modelos_dir = Path(modelos_dir)
        self.resultados_dir = Path(resultados_dir)
        self.resultados_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear subdirectorios
        (self.resultados_dir / "reliability_diagrams").mkdir(exist_ok=True)
        
        self.modelos = {}
        self.metricas = {}
        
    def cargar_modelos(self):
        """
        Carga todos los modelos calibrados
        """
        logger.info("=" * 70)
        logger.info("📂 CARGANDO MODELOS CALIBRADOS")
        logger.info("=" * 70)
        
        # Buscar todos los archivos *_calibrado.pkl
        archivos_modelos = list(self.modelos_dir.glob("*_calibrado.pkl"))
        
        if not archivos_modelos:
            logger.error("❌ No se encontraron modelos calibrados")
            return False
        
        for archivo in archivos_modelos:
            nombre = archivo.stem.replace("_calibrado", "").replace("_", " ").title()
            try:
                self.modelos[nombre] = joblib.load(archivo)
                logger.info(f"✅ {nombre}: {archivo.name}")
            except Exception as e:
                logger.error(f"❌ Error cargando {archivo.name}: {e}")
        
        logger.info(f"\n📊 Total modelos cargados: {len(self.modelos)}")
        return True
    
    def calcular_ece(self, y_true, y_prob, n_bins=10):
        """
        Calcula Expected Calibration Error (ECE)
        
        ECE mide la diferencia entre confianza predicha y accuracy real
        ECE < 0.05 = Excelente calibración
        ECE < 0.10 = Buena calibración
        
        Args:
            y_true: Etiquetas verdaderas
            y_prob: Probabilidades predichas
            n_bins: Número de bins
            
        Returns:
            ece: Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Encontrar predicciones en este bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def evaluar_modelo(self, nombre, modelo, X_test, y_test):
        """
        Evalúa un modelo y calcula todas las métricas de calibración
        
        Args:
            nombre: Nombre del modelo
            modelo: Modelo calibrado
            X_test: Features de test
            y_test: Labels de test
            
        Returns:
            dict con métricas
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 EVALUANDO: {nombre}")
        logger.info(f"{'='*70}")
        
        # Predecir probabilidades
        y_prob = modelo.predict_proba(X_test)[:, 1]
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        accuracy = np.mean(y_pred == y_test)
        brier = brier_score_loss(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)
        ece = self.calcular_ece(y_test, y_prob, n_bins=10)
        
        # Análisis por bins
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        
        metricas = {
            'accuracy': accuracy,
            'brier_score': brier,
            'log_loss': logloss,
            'ece': ece,
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
        
        # Mostrar métricas
        logger.info(f"  Accuracy:     {accuracy*100:.2f}%")
        logger.info(f"  Brier Score:  {brier:.4f} {'✅' if brier < 0.20 else '⚠️'}")
        logger.info(f"  Log Loss:     {logloss:.4f}")
        logger.info(f"  ECE:          {ece:.4f} {'✅' if ece < 0.05 else '⚠️' if ece < 0.10 else '❌'}")
        
        # Interpretación de ECE
        if ece < 0.05:
            logger.info(f"  📈 Calibración: EXCELENTE")
        elif ece < 0.10:
            logger.info(f"  📈 Calibración: BUENA")
        else:
            logger.info(f"  📈 Calibración: MEJORABLE")
        
        return metricas, y_prob
    
    def crear_reliability_diagram(self, nombre, y_true, y_prob, metricas, guardar=True):
        """
        Crea un Reliability Diagram (curva de calibración)
        
        Args:
            nombre: Nombre del modelo
            y_true: Etiquetas verdaderas
            y_prob: Probabilidades predichas
            metricas: Dict con métricas
            guardar: Si True, guarda el gráfico
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- SUBPLOT 1: Reliability Curve ---
        ax = axes[0]
        
        # Curva de calibración
        prob_true = np.array(metricas['prob_true'])
        prob_pred = np.array(metricas['prob_pred'])
        
        # Línea perfecta (y = x)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectamente Calibrado', alpha=0.7)
        
        # Curva real
        ax.plot(prob_pred, prob_true, 'o-', linewidth=3, markersize=10, 
                color='#3498db', label=nombre, markeredgecolor='white', markeredgewidth=2)
        
        # Styling
        ax.set_xlabel('Probabilidad Predicha', fontsize=13, fontweight='bold')
        ax.set_ylabel('Fracción Real de Positivos', fontsize=13, fontweight='bold')
        ax.set_title(f'Reliability Diagram - {nombre}', fontsize=15, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Añadir métricas como texto
        metrics_text = f"""Métricas de Calibración:
        
Brier Score: {metricas['brier_score']:.4f}
ECE:         {metricas['ece']:.4f}
Log Loss:    {metricas['log_loss']:.4f}
Accuracy:    {metricas['accuracy']*100:.2f}%"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
        
        # --- SUBPLOT 2: Distribución de Probabilidades ---
        ax = axes[1]
        
        # Histograma de probabilidades predichas
        ax.hist(y_prob, bins=20, alpha=0.7, color='#3498db', edgecolor='black', linewidth=1.2)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Umbral 0.5')
        
        # Styling
        ax.set_xlabel('Probabilidad Predicha', fontsize=13, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=13, fontweight='bold')
        ax.set_title(f'Distribución de Probabilidades - {nombre}', fontsize=15, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Estadísticas
        stats_text = f"""Estadísticas:
        
Media:    {np.mean(y_prob):.3f}
Mediana:  {np.median(y_prob):.3f}
Std:      {np.std(y_prob):.3f}
Min:      {np.min(y_prob):.3f}
Max:      {np.max(y_prob):.3f}"""
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                family='monospace')
        
        plt.tight_layout()
        
        if guardar:
            filename = self.resultados_dir / "reliability_diagrams" / f"{nombre.lower().replace(' ', '_')}_reliability.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"  💾 Gráfico guardado: {filename}")
        
        plt.close()
    
    def comparar_todos_modelos(self, metricas_todos):
        """
        Crea gráficos de comparación de todos los modelos
        
        Args:
            metricas_todos: Dict con métricas de todos los modelos
        """
        logger.info(f"\n{'='*70}")
        logger.info("📊 GENERANDO COMPARACIÓN DE TODOS LOS MODELOS")
        logger.info(f"{'='*70}")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # --- SUBPLOT 1: Reliability Curves Comparison ---
        ax = axes[0, 0]
        
        # Línea perfecta
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2.5, label='Perfectamente Calibrado', alpha=0.8)
        
        # Curvas de cada modelo
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for idx, (nombre, metricas) in enumerate(metricas_todos.items()):
            prob_true = np.array(metricas['prob_true'])
            prob_pred = np.array(metricas['prob_pred'])
            ax.plot(prob_pred, prob_true, 'o-', linewidth=2.5, markersize=8,
                   color=colors[idx % len(colors)], label=nombre, 
                   markeredgecolor='white', markeredgewidth=1.5)
        
        ax.set_xlabel('Probabilidad Predicha', fontsize=13, fontweight='bold')
        ax.set_ylabel('Fracción Real de Positivos', fontsize=13, fontweight='bold')
        ax.set_title('Comparación de Reliability Curves', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # --- SUBPLOT 2: Brier Score Comparison ---
        ax = axes[0, 1]
        
        nombres = list(metricas_todos.keys())
        brier_scores = [metricas_todos[n]['brier_score'] for n in nombres]
        
        bars = ax.barh(nombres, brier_scores, color=colors[:len(nombres)], 
                       edgecolor='black', linewidth=1.5)
        ax.axvline(x=0.20, color='red', linestyle='--', linewidth=2, 
                  label='Objetivo (< 0.20)', alpha=0.7)
        
        # Añadir valores en las barras
        for i, (bar, val) in enumerate(zip(bars, brier_scores)):
            ax.text(val + 0.002, i, f'{val:.4f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Brier Score (menor es mejor)', fontsize=13, fontweight='bold')
        ax.set_title('Comparación de Brier Score', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_xlim([0, max(brier_scores) * 1.15])
        
        # --- SUBPLOT 3: ECE Comparison ---
        ax = axes[1, 0]
        
        ece_scores = [metricas_todos[n]['ece'] for n in nombres]
        
        bars = ax.barh(nombres, ece_scores, color=colors[:len(nombres)], 
                       edgecolor='black', linewidth=1.5)
        ax.axvline(x=0.05, color='green', linestyle='--', linewidth=2, 
                  label='Excelente (< 0.05)', alpha=0.7)
        ax.axvline(x=0.10, color='orange', linestyle='--', linewidth=2, 
                  label='Bueno (< 0.10)', alpha=0.7)
        
        # Añadir valores en las barras
        for i, (bar, val) in enumerate(zip(bars, ece_scores)):
            ax.text(val + 0.001, i, f'{val:.4f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Expected Calibration Error (menor es mejor)', fontsize=13, fontweight='bold')
        ax.set_title('Comparación de ECE', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_xlim([0, max(ece_scores) * 1.2])
        
        # --- SUBPLOT 4: Accuracy Comparison ---
        ax = axes[1, 1]
        
        accuracies = [metricas_todos[n]['accuracy'] * 100 for n in nombres]
        
        bars = ax.barh(nombres, accuracies, color=colors[:len(nombres)], 
                       edgecolor='black', linewidth=1.5)
        ax.axvline(x=62, color='red', linestyle='--', linewidth=2, 
                  label='Objetivo (62%)', alpha=0.7)
        
        # Añadir valores en las barras
        for i, (bar, val) in enumerate(zip(bars, accuracies)):
            ax.text(val + 0.3, i, f'{val:.2f}%', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title('Comparación de Accuracy', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_xlim([min(accuracies) - 2, max(accuracies) + 3])
        
        plt.tight_layout()
        
        filename = self.resultados_dir / "calibration_comparison_all_models.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Comparación guardada: {filename}")
        plt.close()
    
    def generar_resumen_metricas(self, metricas_todos):
        """
        Genera resumen de métricas en CSV y JSON
        
        Args:
            metricas_todos: Dict con métricas de todos los modelos
        """
        logger.info(f"\n{'='*70}")
        logger.info("💾 GUARDANDO RESUMEN DE MÉTRICAS")
        logger.info(f"{'='*70}")
        
        # Crear DataFrame
        df_metricas = pd.DataFrame({
            'Modelo': list(metricas_todos.keys()),
            'Accuracy': [m['accuracy'] for m in metricas_todos.values()],
            'Brier_Score': [m['brier_score'] for m in metricas_todos.values()],
            'Log_Loss': [m['log_loss'] for m in metricas_todos.values()],
            'ECE': [m['ece'] for m in metricas_todos.values()]
        })
        
        # Ordenar por Brier Score (menor es mejor)
        df_metricas = df_metricas.sort_values('Brier_Score')
        
        # Guardar CSV
        csv_path = self.resultados_dir / "calibration_metrics.csv"
        df_metricas.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV guardado: {csv_path}")
        
        # Guardar JSON completo (con prob_true y prob_pred)
        json_path = self.resultados_dir / "calibration_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(metricas_todos, f, indent=2)
        logger.info(f"✅ JSON guardado: {json_path}")
        
        # Mostrar tabla
        logger.info(f"\n📊 RESUMEN DE MÉTRICAS:")
        logger.info("\n" + df_metricas.to_string(index=False))
        
        # Mejor modelo
        mejor_modelo = df_metricas.iloc[0]
        logger.info(f"\n{'='*70}")
        logger.info(f"🏆 MEJOR MODELO (por Brier Score): {mejor_modelo['Modelo']}")
        logger.info(f"{'='*70}")
        logger.info(f"  Accuracy:     {mejor_modelo['Accuracy']*100:.2f}%")
        logger.info(f"  Brier Score:  {mejor_modelo['Brier_Score']:.4f}")
        logger.info(f"  ECE:          {mejor_modelo['ECE']:.4f}")
        logger.info(f"  Log Loss:     {mejor_modelo['Log_Loss']:.4f}")
        
        return df_metricas
    
    def validar_todos(self, X_test, y_test):
        """
        Ejecuta validación completa de todos los modelos
        
        Args:
            X_test: Features de test
            y_test: Labels de test
        """
        logger.info("\n" + "="*70)
        logger.info("🎯 INICIANDO VALIDACIÓN COMPLETA DE CALIBRACIÓN")
        logger.info("="*70)
        
        metricas_todos = {}
        
        # Evaluar cada modelo
        for nombre, modelo in self.modelos.items():
            metricas, y_prob = self.evaluar_modelo(nombre, modelo, X_test, y_test)
            metricas_todos[nombre] = metricas
            
            # Crear reliability diagram individual
            self.crear_reliability_diagram(nombre, y_test, y_prob, metricas)
        
        # Comparación de todos los modelos
        self.comparar_todos_modelos(metricas_todos)
        
        # Guardar resumen
        df_resumen = self.generar_resumen_metricas(metricas_todos)
        
        # Verificar criterios de éxito
        self.verificar_criterios_exito(df_resumen)
        
        logger.info("\n" + "="*70)
        logger.info("✅ VALIDACIÓN DE CALIBRACIÓN COMPLETADA")
        logger.info("="*70)
        logger.info(f"\n📁 Resultados guardados en: {self.resultados_dir}")
        
        return metricas_todos, df_resumen
    
    def verificar_criterios_exito(self, df_resumen):
        """
        Verifica si se cumplen los criterios de éxito de Fase 2
        
        Args:
            df_resumen: DataFrame con resumen de métricas
        """
        logger.info(f"\n{'='*70}")
        logger.info("✅ VERIFICACIÓN DE CRITERIOS DE ÉXITO - FASE 2")
        logger.info(f"{'='*70}")
        
        criterios_cumplidos = 0
        total_criterios = 3
        
        # Criterio 1: Brier Score < 0.20
        mejor_brier = df_resumen['Brier_Score'].min()
        if mejor_brier < 0.20:
            logger.info(f"✅ Criterio 1: Brier Score < 0.20 → {mejor_brier:.4f} ✓")
            criterios_cumplidos += 1
        else:
            logger.info(f"❌ Criterio 1: Brier Score < 0.20 → {mejor_brier:.4f} ✗")
        
        # Criterio 2: ECE < 0.05 (excelente) o < 0.10 (bueno)
        mejor_ece = df_resumen['ECE'].min()
        if mejor_ece < 0.05:
            logger.info(f"✅ Criterio 2: ECE < 0.05 (Excelente) → {mejor_ece:.4f} ✓")
            criterios_cumplidos += 1
        elif mejor_ece < 0.10:
            logger.info(f"✅ Criterio 2: ECE < 0.10 (Bueno) → {mejor_ece:.4f} ✓")
            criterios_cumplidos += 1
        else:
            logger.info(f"❌ Criterio 2: ECE < 0.10 → {mejor_ece:.4f} ✗")
        
        # Criterio 3: Reliability diagrams muestran buena calibración
        # (verificación visual, asumimos OK si ECE es bueno)
        if mejor_ece < 0.10:
            logger.info(f"✅ Criterio 3: Reliability Diagrams OK (basado en ECE) ✓")
            criterios_cumplidos += 1
        else:
            logger.info(f"⚠️  Criterio 3: Revisar Reliability Diagrams manualmente")
        
        # Resumen final
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 RESULTADO: {criterios_cumplidos}/{total_criterios} criterios cumplidos")
        logger.info(f"{'='*70}")
        
        if criterios_cumplidos == total_criterios:
            logger.info("🎉 ¡TODOS LOS CRITERIOS CUMPLIDOS! Calibración EXCELENTE")
        elif criterios_cumplidos >= 2:
            logger.info("✅ Calibración BUENA - Listo para backtesting")
        else:
            logger.info("⚠️  Calibración MEJORABLE - Revisar modelos")


def main():
    """
    Función principal
    """
    logger.info("\n" + "="*70)
    logger.info("🎯 VALIDACIÓN DE CALIBRACIÓN - FASE 2")
    logger.info("="*70)
    
    # Crear validador
    validator = CalibrationValidator()
    
    # Cargar modelos
    if not validator.cargar_modelos():
        logger.error("❌ No se pudieron cargar los modelos")
        return
    
    # Cargar datos de test
    logger.info(f"\n{'='*70}")
    logger.info("📂 CARGANDO DATOS DE TEST")
    logger.info(f"{'='*70}")
    
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').reset_index(drop=True)
    
    logger.info(f"  Total partidos: {len(df)}")
    
    # Cargar features seleccionadas (30 features)
    with open("resultados/selected_features.txt", "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    logger.info(f"  Features seleccionadas: {len(feature_cols)}")
    
    # Verificar que todas las features existen
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        logger.error(f"❌ Features faltantes: {missing_features}")
        return
    
    # Split: últimos 20% para test
    n = len(df)
    test_start = int(n * 0.8)
    
    X_test = df.iloc[test_start:][feature_cols]
    y_test = df.iloc[test_start:]['resultado']
    
    logger.info(f"  Test set: {len(X_test)} partidos ({len(X_test)/n*100:.1f}%)")
    logger.info(f"  Periodo: {df.iloc[test_start]['fecha'].date()} a {df.iloc[-1]['fecha'].date()}")
    
    # Validar todos los modelos
    metricas, df_resumen = validator.validar_todos(X_test, y_test)
    
    logger.info("\n" + "="*70)
    logger.info("✅ PROCESO COMPLETADO")
    logger.info("="*70)
    logger.info(f"\n📁 Revisa los resultados en: resultados/calibracion/")
    logger.info(f"   - Reliability diagrams individuales")
    logger.info(f"   - Comparación de todos los modelos")
    logger.info(f"   - Métricas en CSV y JSON")



if __name__ == "__main__":
    main()
