"""
Sistema de Backtesting Completo - Fase 2
=========================================

Simula apuestas en datos históricos para validar rentabilidad del modelo:
- Múltiples umbrales de EV
- Análisis financiero completo (ROI, Win Rate, Profit/Loss)
- Análisis de drawdown
- Análisis por categorías (superficie, ranking, etc.)
- Curvas de ganancias acumuladas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import json
import logging
from datetime import datetime
import sys
import warnings

# Suprimir warnings de sklearn para output más limpio
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


class BacktestingSystem:
    """
    Sistema completo de backtesting para validar rentabilidad
    """
    
    def __init__(self, modelo_path, resultados_dir="resultados/backtesting"):
        """
        Inicializa el sistema de backtesting
        
        Args:
            modelo_path: Path al modelo calibrado
            resultados_dir: Directorio para guardar resultados
        """
        self.modelo_path = Path(modelo_path)
        self.resultados_dir = Path(resultados_dir)
        self.resultados_dir.mkdir(parents=True, exist_ok=True)
        
        self.modelo = None
        self.nombre_modelo = self.modelo_path.stem.replace("_calibrado", "").replace("_", " ").title()
        
    def cargar_modelo(self):
        """Carga el modelo calibrado"""
        logger.info(f"📂 Cargando modelo: {self.modelo_path.name}")
        self.modelo = joblib.load(self.modelo_path)
        logger.info(f"✅ Modelo cargado: {self.nombre_modelo}")
    
    def simular_cuotas(self, df):
        """
        Simula cuotas realistas basadas en rankings
        
        Args:
            df: DataFrame con partidos
            
        Returns:
            DataFrame con columna 'cuota' añadida
        """
        logger.info("📊 Simulando cuotas basadas en rankings...")
        
        df = df.copy()
        
        # Cuota basada en diferencia de ranking
        # Mejor ranking (menor número) = menor cuota
        # Fórmula: cuota = 1.5 + (rank_diff / 100) * 2.5
        
        df['rank_diff'] = df['jugador_rank'] - df['oponente_rank']
        
        # Cuota base
        df['cuota'] = 1.8 + (df['rank_diff'] / 100) * 2.0
        
        # Ajustar por rankings absolutos
        # Si jugador es top 10, reducir cuota
        df.loc[df['jugador_rank'] <= 10, 'cuota'] *= 0.85
        df.loc[df['jugador_rank'] <= 5, 'cuota'] *= 0.90
        
        # Si oponente es top 10, aumentar cuota
        df.loc[df['oponente_rank'] <= 10, 'cuota'] *= 1.15
        
        # Limitar cuotas a rango realista
        df['cuota'] = df['cuota'].clip(1.15, 8.00)
        
        logger.info(f"  Cuota media: {df['cuota'].mean():.2f}")
        logger.info(f"  Cuota min: {df['cuota'].min():.2f}, max: {df['cuota'].max():.2f}")
        
        return df
    
    def calcular_ev(self, prob, cuota):
        """
        Calcula Expected Value
        
        EV = (probabilidad * cuota) - 1
        
        Args:
            prob: Probabilidad del modelo
            cuota: Cuota de la casa de apuestas
            
        Returns:
            EV (Expected Value)
        """
        return (prob * cuota) - 1
    
    def simular_apuestas(self, df_test, umbral_ev=0.03, stake=10.0):
        """
        Simula apuestas en datos históricos
        
        Args:
            df_test: DataFrame con partidos de test
            umbral_ev: EV mínimo para apostar
            stake: Cantidad a apostar por partido (flat betting)
            
        Returns:
            DataFrame con resultados de apuestas
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🎲 SIMULANDO APUESTAS - Umbral EV: {umbral_ev*100:.1f}%")
        logger.info(f"{'='*70}")
        
        resultados = []
        
        for idx, row in df_test.iterrows():
            # Predecir probabilidad
            features = row[self.feature_cols].values.reshape(1, -1)
            prob = self.modelo.predict_proba(features)[0, 1]
            
            # Calcular EV
            cuota = row['cuota']
            ev = self.calcular_ev(prob, cuota)
            
            # Decisión de apuesta
            if ev > umbral_ev:
                # APOSTAR
                resultado_real = row['resultado']
                
                if resultado_real == 1:
                    # GANAMOS
                    ganancia = stake * (cuota - 1)  # Ganancia neta
                    retorno = stake * cuota  # Retorno total
                else:
                    # PERDIMOS
                    ganancia = -stake
                    retorno = 0
                
                resultados.append({
                    'fecha': row['fecha'],
                    'jugador_rank': row['jugador_rank'],
                    'oponente_rank': row['oponente_rank'],
                    'superficie': row.get('superficie', 'Unknown'),
                    'prob_modelo': prob,
                    'cuota': cuota,
                    'ev': ev,
                    'stake': stake,
                    'resultado': resultado_real,
                    'ganancia': ganancia,
                    'retorno': retorno
                })
        
        df_resultados = pd.DataFrame(resultados)
        
        if len(df_resultados) == 0:
            logger.warning(f"⚠️  No se encontraron apuestas con EV > {umbral_ev*100:.1f}%")
            return df_resultados
        
        logger.info(f"  Total apuestas: {len(df_resultados)}")
        logger.info(f"  Periodo: {df_resultados['fecha'].min().date()} a {df_resultados['fecha'].max().date()}")
        
        return df_resultados
    
    def calcular_metricas_financieras(self, df_resultados):
        """
        Calcula métricas financieras completas
        
        Args:
            df_resultados: DataFrame con resultados de apuestas
            
        Returns:
            dict con métricas
        """
        if len(df_resultados) == 0:
            return None
        
        # Métricas básicas
        total_apuestas = len(df_resultados)
        ganadas = (df_resultados['resultado'] == 1).sum()
        perdidas = (df_resultados['resultado'] == 0).sum()
        win_rate = ganadas / total_apuestas
        
        # Métricas financieras
        total_apostado = df_resultados['stake'].sum()
        total_retornado = df_resultados['retorno'].sum()
        ganancia_neta = df_resultados['ganancia'].sum()
        roi = (ganancia_neta / total_apostado) * 100
        
        # Profit Factor (total ganado / total perdido)
        total_ganado = df_resultados[df_resultados['ganancia'] > 0]['ganancia'].sum()
        total_perdido = abs(df_resultados[df_resultados['ganancia'] < 0]['ganancia'].sum())
        profit_factor = total_ganado / total_perdido if total_perdido > 0 else np.inf
        
        # Drawdown
        df_resultados['ganancia_acumulada'] = df_resultados['ganancia'].cumsum()
        running_max = df_resultados['ganancia_acumulada'].cummax()
        drawdown = df_resultados['ganancia_acumulada'] - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / total_apostado) * 100 if total_apostado > 0 else 0
        
        # Sharpe Ratio (simplificado)
        returns = df_resultados['ganancia'] / df_resultados['stake']
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # EV promedio
        ev_promedio = df_resultados['ev'].mean()
        
        metricas = {
            'total_apuestas': total_apuestas,
            'apuestas_ganadas': ganadas,
            'apuestas_perdidas': perdidas,
            'win_rate': win_rate * 100,
            'total_apostado': total_apostado,
            'total_retornado': total_retornado,
            'ganancia_neta': ganancia_neta,
            'roi': roi,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'ev_promedio': ev_promedio * 100
        }
        
        return metricas
    
    def mostrar_metricas(self, metricas, umbral_ev):
        """
        Muestra métricas en consola
        
        Args:
            metricas: Dict con métricas
            umbral_ev: Umbral de EV usado
        """
        if metricas is None:
            return
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 RESULTADOS - Umbral EV: {umbral_ev*100:.1f}%")
        logger.info(f"{'='*70}")
        
        logger.info(f"\n🎯 PERFORMANCE:")
        logger.info(f"  Total apuestas:     {metricas['total_apuestas']}")
        logger.info(f"  Apuestas ganadas:   {metricas['apuestas_ganadas']} ({metricas['win_rate']:.1f}%)")
        logger.info(f"  Apuestas perdidas:  {metricas['apuestas_perdidas']} ({100-metricas['win_rate']:.1f}%)")
        
        logger.info(f"\n💰 FINANCIERO:")
        logger.info(f"  Total apostado:     {metricas['total_apostado']:.2f}€")
        logger.info(f"  Total retornado:    {metricas['total_retornado']:.2f}€")
        logger.info(f"  Ganancia neta:      {metricas['ganancia_neta']:+.2f}€")
        logger.info(f"  ROI:                {metricas['roi']:+.2f}%")
        logger.info(f"  Profit Factor:      {metricas['profit_factor']:.2f}")
        
        logger.info(f"\n📉 RIESGO:")
        logger.info(f"  Max Drawdown:       {metricas['max_drawdown']:.2f}€ ({metricas['max_drawdown_pct']:.1f}%)")
        logger.info(f"  Sharpe Ratio:       {metricas['sharpe_ratio']:.2f}")
        
        logger.info(f"\n📈 MODELO:")
        logger.info(f"  EV promedio:        {metricas['ev_promedio']:+.2f}%")
        
        # Evaluación
        if metricas['roi'] > 5:
            logger.info(f"\n🎉 ¡EXCELENTE! ROI > 5%")
        elif metricas['roi'] > 0:
            logger.info(f"\n✅ Rentable (ROI positivo)")
        else:
            logger.info(f"\n⚠️  No rentable con este umbral")
    
    def analizar_por_superficie(self, df_resultados):
        """
        Analiza resultados por superficie
        
        Args:
            df_resultados: DataFrame con resultados
            
        Returns:
            DataFrame con análisis por superficie
        """
        if len(df_resultados) == 0:
            return pd.DataFrame()
        
        logger.info(f"\n{'='*70}")
        logger.info("🎾 ANÁLISIS POR SUPERFICIE")
        logger.info(f"{'='*70}")
        
        analisis = []
        
        for superficie in df_resultados['superficie'].unique():
            df_sup = df_resultados[df_resultados['superficie'] == superficie]
            
            if len(df_sup) == 0:
                continue
            
            total = len(df_sup)
            ganadas = (df_sup['resultado'] == 1).sum()
            win_rate = ganadas / total * 100
            ganancia = df_sup['ganancia'].sum()
            apostado = df_sup['stake'].sum()
            roi = (ganancia / apostado) * 100 if apostado > 0 else 0
            
            analisis.append({
                'Superficie': superficie,
                'Apuestas': total,
                'Ganadas': ganadas,
                'Win_Rate': win_rate,
                'Ganancia': ganancia,
                'ROI': roi
            })
            
            logger.info(f"\n{superficie}:")
            logger.info(f"  Apuestas: {total} | Win Rate: {win_rate:.1f}% | ROI: {roi:+.2f}%")
        
        return pd.DataFrame(analisis)
    
    def analizar_por_rango_ev(self, df_resultados):
        """
        Analiza resultados por rangos de EV
        
        Args:
            df_resultados: DataFrame con resultados
            
        Returns:
            DataFrame con análisis por rango de EV
        """
        if len(df_resultados) == 0:
            return pd.DataFrame()
        
        logger.info(f"\n{'='*70}")
        logger.info("📊 ANÁLISIS POR RANGO DE EV")
        logger.info(f"{'='*70}")
        
        # Definir bins de EV
        df_resultados['ev_bin'] = pd.cut(
            df_resultados['ev'],
            bins=[0, 0.03, 0.05, 0.08, 0.15, 1.0],
            labels=['0-3%', '3-5%', '5-8%', '8-15%', '>15%']
        )
        
        analisis = []
        
        for ev_range in ['0-3%', '3-5%', '5-8%', '8-15%', '>15%']:
            df_ev = df_resultados[df_resultados['ev_bin'] == ev_range]
            
            if len(df_ev) == 0:
                continue
            
            total = len(df_ev)
            ganadas = (df_ev['resultado'] == 1).sum()
            win_rate = ganadas / total * 100
            ganancia = df_ev['ganancia'].sum()
            apostado = df_ev['stake'].sum()
            roi = (ganancia / apostado) * 100 if apostado > 0 else 0
            
            analisis.append({
                'Rango_EV': ev_range,
                'Apuestas': total,
                'Ganadas': ganadas,
                'Win_Rate': win_rate,
                'Ganancia': ganancia,
                'ROI': roi
            })
            
            logger.info(f"\n{ev_range}:")
            logger.info(f"  Apuestas: {total} | Win Rate: {win_rate:.1f}% | ROI: {roi:+.2f}%")
        
        return pd.DataFrame(analisis)
    
    def crear_curva_ganancias(self, df_resultados, umbral_ev, guardar=True):
        """
        Crea curva de ganancias acumuladas
        
        Args:
            df_resultados: DataFrame con resultados
            umbral_ev: Umbral de EV usado
            guardar: Si True, guarda el gráfico
        """
        if len(df_resultados) == 0:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # --- SUBPLOT 1: Ganancia Acumulada ---
        ax = axes[0]
        
        df_sorted = df_resultados.sort_index()
        ganancia_acum = df_sorted['ganancia'].cumsum()
        
        # Plotear ganancia acumulada
        ax.plot(range(1, len(ganancia_acum) + 1), ganancia_acum.values, 
                linewidth=2.5, color='#2ecc71', label='Ganancia Acumulada')
        ax.fill_between(range(1, len(ganancia_acum) + 1), 0, ganancia_acum.values, 
                        alpha=0.2, color='#2ecc71')
        
        # Línea de 0
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Styling
        ax.set_xlabel('Número de Apuesta', fontsize=13, fontweight='bold')
        ax.set_ylabel('Ganancia Acumulada (€)', fontsize=13, fontweight='bold')
        ax.set_title(f'Curva de Ganancias Acumuladas - {self.nombre_modelo} (EV > {umbral_ev*100:.1f}%)', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Añadir estadísticas
        ganancia_final = ganancia_acum.iloc[-1]
        color = 'green' if ganancia_final > 0 else 'red'
        
        stats_text = f"""Estadísticas Finales:
        
Ganancia Final:  {ganancia_final:+.2f}€
Total Apuestas:  {len(df_resultados)}
Win Rate:        {(df_resultados['resultado']==1).sum()/len(df_resultados)*100:.1f}%
ROI:             {(ganancia_final/df_resultados['stake'].sum())*100:+.2f}%"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                family='monospace')
        
        # --- SUBPLOT 2: Drawdown ---
        ax = axes[1]
        
        running_max = ganancia_acum.cummax()
        drawdown = ganancia_acum - running_max
        
        ax.fill_between(range(1, len(drawdown) + 1), 0, drawdown.values, 
                        alpha=0.5, color='#e74c3c', label='Drawdown')
        ax.plot(range(1, len(drawdown) + 1), drawdown.values, 
                linewidth=2, color='#c0392b')
        
        # Styling
        ax.set_xlabel('Número de Apuesta', fontsize=13, fontweight='bold')
        ax.set_ylabel('Drawdown (€)', fontsize=13, fontweight='bold')
        ax.set_title('Análisis de Drawdown', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Añadir max drawdown
        max_dd = drawdown.min()
        max_dd_pct = (max_dd / df_resultados['stake'].sum()) * 100
        
        dd_text = f"""Max Drawdown:
        
Absoluto:  {max_dd:.2f}€
Relativo:  {max_dd_pct:.1f}%"""
        
        ax.text(0.02, 0.02, dd_text, transform=ax.transAxes,
                verticalalignment='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3),
                family='monospace')
        
        plt.tight_layout()
        
        if guardar:
            filename = self.resultados_dir / f"cumulative_profit_ev{int(umbral_ev*100)}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"  💾 Gráfico guardado: {filename}")
        
        plt.close()
    
    def comparar_umbrales_ev(self, df_test, umbrales=[0.00, 0.03, 0.05, 0.08]):
        """
        Compara resultados con diferentes umbrales de EV
        
        Args:
            df_test: DataFrame con datos de test
            umbrales: Lista de umbrales de EV a probar
            
        Returns:
            DataFrame con comparación
        """
        logger.info(f"\n{'='*70}")
        logger.info("🔍 COMPARANDO DIFERENTES UMBRALES DE EV")
        logger.info(f"{'='*70}")
        
        resultados_comparacion = []
        
        for umbral in umbrales:
            df_apuestas = self.simular_apuestas(df_test, umbral_ev=umbral)
            
            if len(df_apuestas) > 0:
                metricas = self.calcular_metricas_financieras(df_apuestas)
                self.mostrar_metricas(metricas, umbral)
                
                # Guardar curva de ganancias
                self.crear_curva_ganancias(df_apuestas, umbral)
                
                resultados_comparacion.append({
                    'Umbral_EV': f"{umbral*100:.1f}%",
                    'Apuestas': metricas['total_apuestas'],
                    'Win_Rate': metricas['win_rate'],
                    'ROI': metricas['roi'],
                    'Ganancia': metricas['ganancia_neta'],
                    'Max_Drawdown': metricas['max_drawdown'],
                    'Profit_Factor': metricas['profit_factor']
                })
        
        df_comparacion = pd.DataFrame(resultados_comparacion)
        
        # Guardar comparación
        csv_path = self.resultados_dir / "ev_threshold_comparison.csv"
        df_comparacion.to_csv(csv_path, index=False)
        logger.info(f"\n💾 Comparación guardada: {csv_path}")
        
        # Visualizar comparación
        self.visualizar_comparacion_umbrales(df_comparacion)
        
        return df_comparacion
    
    def visualizar_comparacion_umbrales(self, df_comparacion):
        """
        Visualiza comparación de umbrales de EV
        
        Args:
            df_comparacion: DataFrame con comparación
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        umbrales = df_comparacion['Umbral_EV']
        
        # --- ROI ---
        ax = axes[0, 0]
        bars = ax.bar(umbrales, df_comparacion['ROI'], color='#3498db', edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        for bar, val in zip(bars, df_comparacion['ROI']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', 
                   fontsize=10, fontweight='bold')
        
        ax.set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
        ax.set_title('ROI por Umbral de EV', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # --- Win Rate ---
        ax = axes[0, 1]
        ax.plot(umbrales, df_comparacion['Win_Rate'], 'o-', linewidth=2.5, markersize=10, color='#2ecc71')
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='50%')
        ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Win Rate por Umbral de EV', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # --- Número de Apuestas ---
        ax = axes[1, 0]
        ax.bar(umbrales, df_comparacion['Apuestas'], color='#f39c12', edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Número de Apuestas', fontsize=12, fontweight='bold')
        ax.set_title('Volumen de Apuestas por Umbral', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # --- Ganancia Neta ---
        ax = axes[1, 1]
        colors = ['green' if x > 0 else 'red' for x in df_comparacion['Ganancia']]
        bars = ax.bar(umbrales, df_comparacion['Ganancia'], color=colors, edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        for bar, val in zip(bars, df_comparacion['Ganancia']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}€', ha='center', va='bottom' if val > 0 else 'top', 
                   fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Ganancia Neta (€)', fontsize=12, fontweight='bold')
        ax.set_title('Ganancia Neta por Umbral', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = self.resultados_dir / "ev_threshold_analysis.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Análisis de umbrales guardado: {filename}")
        plt.close()
    
    def ejecutar_backtesting_completo(self, df_test, feature_cols):
        """
        Ejecuta backtesting completo con todos los análisis
        
        Args:
            df_test: DataFrame con datos de test
            feature_cols: Lista de columnas de features
        """
        self.feature_cols = feature_cols
        
        logger.info("\n" + "="*70)
        logger.info(f"🎯 BACKTESTING COMPLETO - {self.nombre_modelo}")
        logger.info("="*70)
        
        # Simular cuotas
        df_test = self.simular_cuotas(df_test)
        
        # Comparar diferentes umbrales de EV
        df_comparacion = self.comparar_umbrales_ev(df_test)
        
        # Análisis detallado con umbral óptimo (el que da mejor ROI)
        mejor_umbral_idx = df_comparacion['ROI'].idxmax()
        mejor_umbral = float(df_comparacion.iloc[mejor_umbral_idx]['Umbral_EV'].rstrip('%')) / 100
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🏆 MEJOR UMBRAL: {mejor_umbral*100:.1f}% (ROI: {df_comparacion.iloc[mejor_umbral_idx]['ROI']:.2f}%)")
        logger.info(f"{'='*70}")
        
        # Simular con mejor umbral
        df_apuestas = self.simular_apuestas(df_test, umbral_ev=mejor_umbral)
        
        if len(df_apuestas) > 0:
            # Análisis por superficie
            df_superficie = self.analizar_por_superficie(df_apuestas)
            if len(df_superficie) > 0:
                df_superficie.to_csv(self.resultados_dir / "analysis_by_surface.csv", index=False)
            
            # Análisis por rango de EV
            df_ev_ranges = self.analizar_por_rango_ev(df_apuestas)
            if len(df_ev_ranges) > 0:
                df_ev_ranges.to_csv(self.resultados_dir / "analysis_by_ev_range.csv", index=False)
            
            # Guardar todas las apuestas
            df_apuestas.to_csv(self.resultados_dir / "all_bets_detailed.csv", index=False)
            logger.info(f"\n💾 Apuestas detalladas guardadas: {self.resultados_dir / 'all_bets_detailed.csv'}")
        
        # Resumen final
        self.generar_resumen_final(df_comparacion)
        
        logger.info("\n" + "="*70)
        logger.info("✅ BACKTESTING COMPLETADO")
        logger.info("="*70)
        logger.info(f"\n📁 Resultados guardados en: {self.resultados_dir}")
    
    def generar_resumen_final(self, df_comparacion):
        """
        Genera resumen final en JSON
        
        Args:
            df_comparacion: DataFrame con comparación de umbrales
        """
        resumen = {
            'modelo': self.nombre_modelo,
            'fecha_analisis': datetime.now().isoformat(),
            'umbrales_analizados': df_comparacion.to_dict('records'),
            'mejor_umbral': {
                'umbral': df_comparacion.loc[df_comparacion['ROI'].idxmax(), 'Umbral_EV'],
                'roi': float(df_comparacion['ROI'].max()),
                'apuestas': int(df_comparacion.loc[df_comparacion['ROI'].idxmax(), 'Apuestas']),
                'win_rate': float(df_comparacion.loc[df_comparacion['ROI'].idxmax(), 'Win_Rate'])
            }
        }
        
        json_path = self.resultados_dir / "backtesting_summary.json"
        with open(json_path, 'w') as f:
            json.dump(resumen, f, indent=2)
        
        logger.info(f"💾 Resumen JSON guardado: {json_path}")


def main():
    """
    Función principal
    """
    logger.info("\n" + "="*70)
    logger.info("🎲 BACKTESTING FASE 2 - SISTEMA COMPLETO")
    logger.info("="*70)
    
    # Seleccionar mejor modelo (Random Forest según resultados anteriores)
    modelo_path = "modelos/random_forest_calibrado.pkl"
    
    # Crear sistema de backtesting
    backtester = BacktestingSystem(modelo_path)
    backtester.cargar_modelo()
    
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
    
    df_test = df.iloc[test_start:].copy()
    
    logger.info(f"  Test set: {len(df_test)} partidos ({len(df_test)/n*100:.1f}%)")
    logger.info(f"  Periodo: {df_test['fecha'].min().date()} a {df_test['fecha'].max().date()}")
    
    # Ejecutar backtesting completo
    backtester.ejecutar_backtesting_completo(df_test, feature_cols)
    
    logger.info("\n" + "="*70)
    logger.info("✅ PROCESO COMPLETADO")
    logger.info("="*70)
    logger.info(f"\n📁 Revisa los resultados en: resultados/backtesting/")



if __name__ == "__main__":
    main()
