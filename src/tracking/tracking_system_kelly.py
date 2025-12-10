"""
Sistema de Tracking con Kelly Criterion integrado
Extensión del TrackingSystem para incluir gestión de bankroll optimizada
"""

import pandas as pd
from pathlib import Path
import sys

# Añadir paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tracking.tracking_system import TrackingSystem
from src.kelly_calculator import KellyCalculator


class TrackingSystemKelly(TrackingSystem):
    """
    Sistema de tracking con Kelly Criterion para optimización de apuestas
    
    Extiende TrackingSystem añadiendo:
    - Cálculo automático de tamaño de apuesta con Kelly
    - Gestión de bankroll dinámico
    - Límites de seguridad
    """
    
    def __init__(self, modelo_path, db_path="apuestas_tracker.db", 
                 bankroll_actual=1000, usar_kelly=True, kelly_fraccion=0.25):
        """
        Inicializa el sistema con Kelly Criterion
        
        Args:
            modelo_path: Path al modelo calibrado
            db_path: Path a la base de datos
            bankroll_actual: Capital actual disponible (default: 1000€)
            usar_kelly: Si True, usa Kelly; si False, usa flat betting
            kelly_fraccion: Fracción de Kelly a usar (default: 0.25 = 25%)
        """
        super().__init__(modelo_path, db_path)
        self.bankroll_actual = bankroll_actual
        self.bankroll_inicial = bankroll_actual
        self.usar_kelly = usar_kelly
        self.kelly_calc = KellyCalculator(fraccion=kelly_fraccion)
        
        print(f"\n💎 Kelly Criterion {'ACTIVADO' if usar_kelly else 'DESACTIVADO'}")
        print(f"💰 Bankroll inicial: {bankroll_actual}€")
        if usar_kelly:
            print(f"📊 Kelly fraccional: {kelly_fraccion*100:.0f}%")
    
    def predecir_y_registrar(self, partido_info, umbral_ev=0.03):
        """
        Predice un partido y calcula el tamaño de apuesta con Kelly
        
        Args:
            partido_info: Información del partido
            umbral_ev: EV mínimo para apostar
        
        Returns:
            dict con resultado incluyendo apuesta_cantidad
        """
        # Predicción normal del sistema base
        resultado = super().predecir_y_registrar(partido_info, umbral_ev)
        
        # Si decidimos apostar, calcular cantidad con Kelly
        if 'APOSTAR' in resultado['decision'] and self.usar_kelly:
            apuesta_kelly = self.kelly_calc.calcular_con_limites(
                prob=resultado['prob_modelo'],
                cuota=resultado['cuota'],
                bankroll=self.bankroll_actual,
                min_apuesta=5,
                max_apuesta_pct=0.05
            )
            
            # Actualizar en la base de datos
            self.db.conn.execute('''
                UPDATE predicciones
                SET apuesta_cantidad = ?
                WHERE id = ?
            ''', (apuesta_kelly, resultado['prediccion_id']))
            self.db.conn.commit()
            
            resultado['apuesta_cantidad'] = apuesta_kelly
            resultado['pct_bankroll'] = (apuesta_kelly / self.bankroll_actual) * 100
            
            print(f"   💰 Apuesta Kelly (25%): {apuesta_kelly:.2f}€ ({resultado['pct_bankroll']:.1f}% del bankroll)")
        elif 'APOSTAR' in resultado['decision']:
            # Flat betting
            apuesta_flat = 10  # Cantidad fija
            resultado['apuesta_cantidad'] = apuesta_flat
            resultado['pct_bankroll'] = (apuesta_flat / self.bankroll_actual) * 100
            
            print(f"   💰 Apuesta Flat: {apuesta_flat:.2f}€")
        else:
            resultado['apuesta_cantidad'] = 0
            resultado['pct_bankroll'] = 0
        
        return resultado
    
    def actualizar_resultado_y_bankroll(self, prediccion_id, resultado_real):
        """
        Actualiza el resultado de una predicción y el bankroll
        
        Args:
            prediccion_id: ID de la predicción
            resultado_real: 1 si ganó, 0 si perdió
        
        Returns:
            dict con información de la actualización
        """
        # Obtener información de la predicción
        pred = self.db.conn.execute('''
            SELECT apuesta_cantidad, cuota
            FROM predicciones
            WHERE id = ?
        ''', (prediccion_id,)).fetchone()
        
        if not pred:
            print(f"⚠️  Predicción {prediccion_id} no encontrada")
            return None
        
        apuesta, cuota = pred
        
        # Calcular ganancia/pérdida
        if resultado_real == 1:
            ganancia = apuesta * (cuota - 1)
        else:
            ganancia = -apuesta
        
        # Actualizar bankroll
        self.bankroll_actual += ganancia
        
        # Actualizar en DB
        self.db.conn.execute('''
            UPDATE predicciones
            SET resultado_real = ?,
                ganancia = ?,
                bankroll_despues = ?
            WHERE id = ?
        ''', (resultado_real, ganancia, self.bankroll_actual, prediccion_id))
        self.db.conn.commit()
        
        # Mostrar resultado
        resultado_texto = "✅ GANÓ" if resultado_real == 1 else "❌ PERDIÓ"
        print(f"\n{resultado_texto} - Predicción #{prediccion_id}")
        print(f"   Apuesta: {apuesta:.2f}€")
        print(f"   Ganancia: {ganancia:+.2f}€")
        print(f"   Bankroll: {self.bankroll_actual:.2f}€ ({((self.bankroll_actual/self.bankroll_inicial)-1)*100:+.1f}%)")
        
        return {
            'prediccion_id': prediccion_id,
            'resultado': resultado_real,
            'apuesta': apuesta,
            'ganancia': ganancia,
            'bankroll': self.bankroll_actual,
            'roi': ((self.bankroll_actual / self.bankroll_inicial) - 1) * 100
        }
    
    def generar_reporte_kelly(self):
        """
        Genera un reporte completo incluyendo métricas de Kelly
        """
        print("\n" + "="*60)
        print("📊 REPORTE KELLY CRITERION")
        print("="*60)
        
        # Reporte base
        super().generar_reporte()
        
        # Métricas adicionales de bankroll
        print("\n" + "="*60)
        print("💰 GESTIÓN DE BANKROLL")
        print("="*60)
        
        print(f"\n💵 Bankroll:")
        print(f"   Inicial:  {self.bankroll_inicial:,.2f}€")
        print(f"   Actual:   {self.bankroll_actual:,.2f}€")
        print(f"   Cambio:   {self.bankroll_actual - self.bankroll_inicial:+,.2f}€")
        
        roi = ((self.bankroll_actual / self.bankroll_inicial) - 1) * 100
        print(f"\n📈 ROI Total: {roi:+.2f}%")
        
        # Distribución de apuestas
        apuestas = pd.read_sql_query('''
            SELECT apuesta_cantidad, cuota, prob_modelo, ganancia
            FROM predicciones
            WHERE decision LIKE '%APOSTAR%'
            ORDER BY fecha_prediccion DESC
        ''', self.db.conn)
        
        if len(apuestas) > 0:
            print(f"\n💰 Distribución de Apuestas:")
            print(f"   Media:    {apuestas['apuesta_cantidad'].mean():.2f}€")
            print(f"   Mediana:  {apuestas['apuesta_cantidad'].median():.2f}€")
            print(f"   Mínima:   {apuestas['apuesta_cantidad'].min():.2f}€")
            print(f"   Máxima:   {apuestas['apuesta_cantidad'].max():.2f}€")
            
            # Apuestas completadas
            completadas = apuestas[apuestas['ganancia'].notna()]
            if len(completadas) > 0:
                print(f"\n📊 Resultados de Apuestas:")
                print(f"   Total apostado: {completadas['apuesta_cantidad'].sum():.2f}€")
                print(f"   Ganancia total: {completadas['ganancia'].sum():+.2f}€")
                print(f"   ROI apuestas:   {(completadas['ganancia'].sum() / completadas['apuesta_cantidad'].sum())*100:+.2f}%")


# Ejemplo de uso
if __name__ == "__main__":
    print("\n" + "="*60)
    print("💎 TRACKING SYSTEM CON KELLY CRITERION")
    print("="*60)
    
    # Crear sistema con Kelly
    sistema = TrackingSystemKelly(
        modelo_path="modelos/xgboost_optimizado_2022_2025.pkl",
        db_path="apuestas_tracker_kelly.db",
        bankroll_actual=1000,
        usar_kelly=True,
        kelly_fraccion=0.25
    )
    
    print("\n✅ Sistema inicializado con Kelly Criterion")
    print("📊 Listo para procesar predicciones con gestión optimizada de bankroll")
    
    # Generar reporte
    sistema.generar_reporte_kelly()
