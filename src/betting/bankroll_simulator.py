"""
Simulador de Bankroll para comparar estrategias de apuestas

Este módulo simula la evolución del bankroll usando diferentes estrategias:
- Flat Betting: apuesta fija por partido
- Kelly Criterion: apuesta optimizada según probabilidad y cuota
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Añadir src al path
sys.path.append(str(Path(__file__).parent))

from kelly_calculator import KellyCalculator


class BankrollSimulator:
    """
    Simula la evolución del bankroll con diferentes estrategias de apuestas
    """
    
    def __init__(self, bankroll_inicial=1000):
        """
        Inicializa el simulador
        
        Args:
            bankroll_inicial: Capital inicial en euros
        """
        self.bankroll_inicial = bankroll_inicial
        self.kelly_calc = KellyCalculator(fraccion=0.25)
    
    def simular_estrategia(self, df_apuestas, estrategia='flat', apuesta_flat=10):
        """
        Simula una estrategia de apuestas completa
        
        Args:
            df_apuestas: DataFrame con columnas:
                - prob_modelo: probabilidad predicha por el modelo (0-1)
                - cuota: cuota decimal del bookmaker
                - resultado: 1 si ganó, 0 si perdió
            estrategia: 'flat' o 'kelly'
            apuesta_flat: cantidad fija si estrategia='flat'
        
        Returns:
            tuple: (historial_bankroll, df_detalle_apuestas)
        """
        bankroll = self.bankroll_inicial
        historial = [bankroll]
        apuestas_realizadas = []
        
        for idx, row in df_apuestas.iterrows():
            # Determinar tamaño de apuesta según estrategia
            if estrategia == 'flat':
                apuesta = apuesta_flat
            else:  # kelly
                apuesta = self.kelly_calc.calcular_con_limites(
                    prob=row['prob_modelo'],
                    cuota=row['cuota'],
                    bankroll=bankroll,
                    min_apuesta=5,
                    max_apuesta_pct=0.05
                )
            
            # Si apuesta es 0, skip (no hay valor o muy poco)
            if apuesta == 0:
                continue
            
            # No apostar si no hay suficiente bankroll
            if apuesta > bankroll:
                print(f"⚠️  Bankroll insuficiente en apuesta #{len(historial)}")
                break
            
            # Calcular resultado de la apuesta
            if row['resultado'] == 1:
                # Ganó
                ganancia = apuesta * (row['cuota'] - 1)
            else:
                # Perdió
                ganancia = -apuesta
            
            # Actualizar bankroll
            bankroll += ganancia
            historial.append(bankroll)
            
            # Guardar detalle
            apuestas_realizadas.append({
                'apuesta_num': len(historial) - 1,
                'apuesta': apuesta,
                'cuota': row['cuota'],
                'prob_modelo': row['prob_modelo'],
                'resultado': row['resultado'],
                'ganancia': ganancia,
                'bankroll': bankroll,
                'pct_bankroll': (apuesta / (bankroll - ganancia)) * 100
            })
            
            # Si bankroll cae a 0 o menos, terminar (bancarrota)
            if bankroll <= 0:
                print(f"💥 BANCARROTA en apuesta #{len(historial)}")
                bankroll = 0
                break
        
        return historial, pd.DataFrame(apuestas_realizadas)
    
    def comparar_estrategias(self, df_apuestas, apuesta_flat=10):
        """
        Compara Flat Betting vs Kelly Criterion
        
        Args:
            df_apuestas: DataFrame con datos de apuestas
            apuesta_flat: cantidad fija para flat betting
        
        Returns:
            dict: Resultados de ambas estrategias
        """
        print("\n" + "="*60)
        print("🎲 SIMULACIÓN DE ESTRATEGIAS DE BANKROLL")
        print("="*60)
        print(f"\n💵 Bankroll inicial: {self.bankroll_inicial}€")
        print(f"📊 Total apuestas disponibles: {len(df_apuestas)}")
        
        # Simular Flat Betting
        print(f"\n⏳ Simulando Flat Betting ({apuesta_flat}€ fijo)...")
        hist_flat, df_flat = self.simular_estrategia(df_apuestas, 'flat', apuesta_flat)
        
        # Simular Kelly Criterion
        print(f"⏳ Simulando Kelly Criterion (25% fraccional)...")
        hist_kelly, df_kelly = self.simular_estrategia(df_apuestas, 'kelly')
        
        # Calcular estadísticas
        resultados = self._calcular_estadisticas(hist_flat, hist_kelly, df_flat, df_kelly)
        
        # Mostrar resultados
        self._mostrar_resultados(resultados, df_flat, df_kelly)
        
        # Visualizar
        self._visualizar_comparacion(hist_flat, hist_kelly, resultados)
        
        return resultados
    
    def _calcular_estadisticas(self, hist_flat, hist_kelly, df_flat, df_kelly):
        """Calcula estadísticas de ambas estrategias"""
        
        # Flat Betting
        final_flat = hist_flat[-1]
        ganancia_flat = final_flat - self.bankroll_inicial
        roi_flat = (ganancia_flat / self.bankroll_inicial) * 100
        
        # Kelly Criterion
        final_kelly = hist_kelly[-1]
        ganancia_kelly = final_kelly - self.bankroll_inicial
        roi_kelly = (ganancia_kelly / self.bankroll_inicial) * 100
        
        # Drawdown (máxima caída desde el pico)
        def calcular_drawdown(historial):
            peak = historial[0]
            max_dd = 0
            for valor in historial:
                if valor > peak:
                    peak = valor
                dd = (peak - valor) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            return max_dd
        
        dd_flat = calcular_drawdown(hist_flat)
        dd_kelly = calcular_drawdown(hist_kelly)
        
        # Volatilidad (desviación estándar de retornos)
        if len(df_flat) > 1:
            vol_flat = df_flat['ganancia'].std()
        else:
            vol_flat = 0
            
        if len(df_kelly) > 1:
            vol_kelly = df_kelly['ganancia'].std()
        else:
            vol_kelly = 0
        
        return {
            'flat': {
                'bankroll_final': final_flat,
                'ganancia': ganancia_flat,
                'roi': roi_flat,
                'num_apuestas': len(df_flat),
                'drawdown': dd_flat,
                'volatilidad': vol_flat
            },
            'kelly': {
                'bankroll_final': final_kelly,
                'ganancia': ganancia_kelly,
                'roi': roi_kelly,
                'num_apuestas': len(df_kelly),
                'drawdown': dd_kelly,
                'volatilidad': vol_kelly
            }
        }
    
    def _mostrar_resultados(self, resultados, df_flat, df_kelly):
        """Muestra los resultados de la comparación"""
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DE LA SIMULACIÓN")
        print("="*60)
        
        # Flat Betting
        print(f"\n💵 FLAT BETTING:")
        print(f"   Bankroll inicial:  {self.bankroll_inicial:,.2f}€")
        print(f"   Bankroll final:    {resultados['flat']['bankroll_final']:,.2f}€")
        print(f"   Ganancia/Pérdida:  {resultados['flat']['ganancia']:+,.2f}€")
        print(f"   ROI:               {resultados['flat']['roi']:+.2f}%")
        print(f"   Apuestas:          {resultados['flat']['num_apuestas']}")
        print(f"   Max Drawdown:      {resultados['flat']['drawdown']:.2f}%")
        print(f"   Volatilidad:       {resultados['flat']['volatilidad']:.2f}€")
        
        # Kelly Criterion
        print(f"\n💎 KELLY CRITERION (25%):")
        print(f"   Bankroll inicial:  {self.bankroll_inicial:,.2f}€")
        print(f"   Bankroll final:    {resultados['kelly']['bankroll_final']:,.2f}€")
        print(f"   Ganancia/Pérdida:  {resultados['kelly']['ganancia']:+,.2f}€")
        print(f"   ROI:               {resultados['kelly']['roi']:+.2f}%")
        print(f"   Apuestas:          {resultados['kelly']['num_apuestas']}")
        print(f"   Max Drawdown:      {resultados['kelly']['drawdown']:.2f}%")
        print(f"   Volatilidad:       {resultados['kelly']['volatilidad']:.2f}€")
        
        # Comparación
        print(f"\n🏆 COMPARACIÓN:")
        diff_ganancia = resultados['kelly']['ganancia'] - resultados['flat']['ganancia']
        diff_roi = resultados['kelly']['roi'] - resultados['flat']['roi']
        
        if diff_ganancia > 0:
            print(f"   ✅ Kelly supera Flat por: {diff_ganancia:+,.2f}€ ({diff_roi:+.2f}% ROI)")
            mejora_pct = (diff_ganancia / abs(resultados['flat']['ganancia'])) * 100 if resultados['flat']['ganancia'] != 0 else float('inf')
            print(f"   📈 Mejora relativa: {mejora_pct:+.1f}%")
        elif diff_ganancia < 0:
            print(f"   ⚠️  Flat supera Kelly por: {abs(diff_ganancia):,.2f}€")
            print(f"   ℹ️  Esto puede ocurrir por varianza en muestras pequeñas")
        else:
            print(f"   ➡️  Ambas estrategias obtienen el mismo resultado")
    
    def _visualizar_comparacion(self, hist_flat, hist_kelly, resultados):
        """Genera visualización de la evolución del bankroll"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico 1: Evolución del bankroll
        ax1.plot(hist_flat, label='Flat Betting', linewidth=2.5, color='#3498db', alpha=0.8)
        ax1.plot(hist_kelly, label='Kelly Criterion (25%)', linewidth=2.5, color='#2ecc71', alpha=0.8)
        ax1.axhline(y=self.bankroll_inicial, linestyle='--', color='red', 
                   alpha=0.5, linewidth=1.5, label='Bankroll Inicial')
        
        ax1.set_xlabel('Número de Apuesta', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Bankroll (€)', fontsize=12, fontweight='bold')
        ax1.set_title('Evolución del Bankroll: Flat vs Kelly', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
        
        # Gráfico 2: Comparación de métricas
        metricas = ['ROI (%)', 'Drawdown (%)', 'Num Apuestas']
        flat_vals = [
            resultados['flat']['roi'],
            resultados['flat']['drawdown'],
            resultados['flat']['num_apuestas'] / 10  # Escalar para visualización
        ]
        kelly_vals = [
            resultados['kelly']['roi'],
            resultados['kelly']['drawdown'],
            resultados['kelly']['num_apuestas'] / 10
        ]
        
        x = np.arange(len(metricas))
        width = 0.35
        
        ax2.bar(x - width/2, flat_vals, width, label='Flat', color='#3498db', alpha=0.8)
        ax2.bar(x + width/2, kelly_vals, width, label='Kelly', color='#2ecc71', alpha=0.8)
        
        ax2.set_ylabel('Valor', fontsize=12, fontweight='bold')
        ax2.set_title('Comparación de Métricas', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metricas)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        
        # Guardar
        output_path = Path(__file__).parent.parent / 'resultados' / 'bankroll_comparison.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Gráfico guardado: {output_path}")
        
        plt.close()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de ejemplo
    np.random.seed(42)
    
    # Simular 100 apuestas con modelo calibrado
    n_apuestas = 100
    df_ejemplo = pd.DataFrame({
        'prob_modelo': np.random.uniform(0.50, 0.70, n_apuestas),
        'cuota': np.random.uniform(1.80, 2.20, n_apuestas),
        'resultado': np.random.binomial(1, 0.55, n_apuestas)  # 55% win rate
    })
    
    # Simular
    sim = BankrollSimulator(bankroll_inicial=1000)
    resultados = sim.comparar_estrategias(df_ejemplo, apuesta_flat=10)
    
    print("\n✅ Simulación de ejemplo completada!")
