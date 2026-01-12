"""
Sistema de Tracking Integrado con Predictor
Incluye soporte opcional para Kelly Criterion
Fase 4-5: Tennis ML Predictor
"""

import pandas as pd
from pathlib import Path
import sys

# Añadir path para imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tracking.database_setup import TennisDatabase
from src.prediction.predictor_calibrado import PredictorCalibrado


class TrackingSystem:
    """
    Sistema integrado de predicción y tracking con soporte opcional para Kelly Criterion

    Características:
    - Predicción y registro automático
    - Gestión de bankroll (opcional con Kelly)
    - Tracking de resultados
    - Generación de reportes
    """

    def __init__(
        self,
        modelo_path,
        db_path="apuestas_tracker.db",
        bankroll_actual=None,
        usar_kelly=False,
        kelly_fraccion=0.25,
    ):
        """
        Inicializa el sistema

        Args:
            modelo_path: Path al modelo calibrado
            db_path: Path a la base de datos SQLite
            bankroll_actual: Capital actual disponible (opcional, para Kelly)
            usar_kelly: Si True, usa Kelly Criterion; si False, usa flat betting
            kelly_fraccion: Fracción de Kelly a usar (default: 0.25 = 25%)
        """
        self.predictor = PredictorCalibrado(modelo_path)
        self.db = TennisDatabase(db_path)
        self.modelo_usado = Path(modelo_path).stem

        # Kelly Criterion (opcional)
        self.usar_kelly = usar_kelly
        self.bankroll_actual = bankroll_actual
        self.bankroll_inicial = bankroll_actual
        self.kelly_calc = None

        if usar_kelly:
            if bankroll_actual is None:
                raise ValueError("bankroll_actual es requerido cuando usar_kelly=True")
            from src.kelly_calculator import KellyCalculator

            self.kelly_calc = KellyCalculator(fraccion=kelly_fraccion)
            print(f"\n💎 Kelly Criterion ACTIVADO")
            print(f"💰 Bankroll inicial: {bankroll_actual}€")
            print(f"📊 Kelly fraccional: {kelly_fraccion*100:.0f}%")

    def predecir_y_registrar(self, partido_info, umbral_ev=0.03):
        """
        Predice un partido y registra en la base de datos

        Args:
            partido_info: dict con información del partido
                {
                    'fecha_partido': '2024-12-10',
                    'jugador_nombre': 'Alcaraz',
                    'jugador_rank': 3,
                    'oponente_nombre': 'Sinner',
                    'oponente_rank': 1,
                    'superficie': 'Hard',
                    'torneo': 'ATP Finals',
                    'cuota': 2.10,
                    'bookmaker': 'Bet365',
                    'features': {...}  # Features preparadas
                }
            umbral_ev: EV mínimo para apostar

        Returns:
            dict con resultado completo
        """

        # Realizar predicción
        resultado = self.predictor.recomendar_apuesta(
            features=partido_info["features"], cuota=partido_info["cuota"], umbral_ev=umbral_ev
        )

        # Calcular cantidad de apuesta
        apuesta_cantidad = 0
        if resultado["apostar"]:
            if self.usar_kelly and self.kelly_calc:
                # Usar Kelly Criterion
                apuesta_cantidad = self.kelly_calc.calcular_con_limites(
                    prob=resultado["probabilidad"],
                    cuota=resultado["cuota"],
                    bankroll=self.bankroll_actual,
                    min_apuesta=5,
                    max_apuesta_pct=0.05,
                )
                pct_bankroll = (apuesta_cantidad / self.bankroll_actual) * 100
                print(
                    f"   💰 Apuesta Kelly: {apuesta_cantidad:.2f}€ ({pct_bankroll:.1f}% del bankroll)"
                )
            else:
                # Flat betting
                apuesta_cantidad = 10.0
                print(f"   💰 Apuesta Flat: {apuesta_cantidad:.2f}€")

        # Preparar para insertar en DB
        db_entry = {
            "fecha_partido": partido_info["fecha_partido"],
            "jugador_nombre": partido_info["jugador_nombre"],
            "jugador_rank": partido_info.get("jugador_rank"),
            "oponente_nombre": partido_info["oponente_nombre"],
            "oponente_rank": partido_info.get("oponente_rank"),
            "superficie": partido_info.get("superficie"),
            "torneo": partido_info.get("torneo"),
            "ronda": partido_info.get("ronda"),
            "prob_modelo": resultado["probabilidad"],
            "prob_modelo_calibrada": resultado["probabilidad"],  # Ya está calibrada
            "cuota": resultado["cuota"],
            "bookmaker": partido_info.get("bookmaker"),
            "ev": resultado["ev"],
            "umbral_ev": umbral_ev,
            "decision": resultado["decision"],
            "modelo_usado": self.modelo_usado,
            "version_modelo": "v3.0",
            "apuesta_cantidad": apuesta_cantidad if resultado["apostar"] else None,
        }

        # Insertar en DB
        prediccion_id = self.db.insertar_prediccion(db_entry)

        # Añadir información al resultado
        resultado["prediccion_id"] = prediccion_id
        resultado["apuesta_cantidad"] = apuesta_cantidad
        if self.usar_kelly and apuesta_cantidad > 0:
            resultado["pct_bankroll"] = (apuesta_cantidad / self.bankroll_actual) * 100

        return resultado

    def procesar_jornada(self, partidos_df, umbral_ev=0.03):
        """
        Procesa una jornada completa de partidos

        Args:
            partidos_df: DataFrame con partidos del día
                Columnas: fecha_partido, jugador_nombre, jugador_rank,
                         oponente_nombre, oponente_rank, superficie, cuota, features, etc.

        Returns:
            DataFrame con resultados
        """

        print("=" * 60)
        print("📅 PROCESANDO JORNADA")
        print("=" * 60)
        print(f"\n📊 Total de partidos: {len(partidos_df)}")

        resultados = []

        for idx, row in partidos_df.iterrows():
            print(f"\n{idx+1}. {row['jugador_nombre']} vs {row['oponente_nombre']}")

            resultado = self.predecir_y_registrar(partido_info=row.to_dict(), umbral_ev=umbral_ev)

            resultados.append(resultado)

            # Mostrar resumen
            if resultado["apostar"]:
                print(f"   ✅ APOSTAR - EV: +{resultado['ev']*100:.2f}%")
            else:
                print(f"   ❌ NO APOSTAR - EV: {resultado['ev']*100:+.2f}%")

        df_resultados = pd.DataFrame(resultados)

        # Resumen
        apuestas = df_resultados[df_resultados["apostar"] == True]

        print("\n" + "=" * 60)
        print("📊 RESUMEN DE LA JORNADA")
        print("=" * 60)
        print(f"Total evaluado: {len(df_resultados)}")
        print(
            f"Apuestas recomendadas: {len(apuestas)} ({len(apuestas)/len(df_resultados)*100:.1f}%)"
        )

        if len(apuestas) > 0:
            print(f"EV promedio: +{apuestas['ev'].mean()*100:.2f}%")
            print(f"Total a apostar: {len(apuestas) * 10:.0f}€")

        return df_resultados

    def actualizar_resultados_batch(self, resultados_reales):
        """
        Actualiza resultados de múltiples partidos

        Args:
            resultados_reales: DataFrame con columnas:
                - prediccion_id
                - resultado (1 o 0)
        """

        print("=" * 60)
        print("🔄 ACTUALIZANDO RESULTADOS")
        print("=" * 60)

        for idx, row in resultados_reales.iterrows():
            # Obtener info de la predicción
            df_pred = self.db.obtener_predicciones()
            df_pred = df_pred[df_pred["id"] == row["prediccion_id"]]

            if len(df_pred) == 0:
                print(f"⚠️  Predicción {row['prediccion_id']} no encontrada")
                continue

            pred = df_pred.iloc[0]

            # Calcular ganancia si hubo apuesta
            if pred["decision"] == "APOSTAR" or pred["apuesta_cantidad"] is not None:
                if row["resultado"] == 1:
                    ganancia = pred["apuesta_cantidad"] * (pred["cuota"] - 1)
                else:
                    ganancia = -pred["apuesta_cantidad"]

                # Actualizar bankroll si Kelly está activado
                if self.usar_kelly and ganancia is not None:
                    self.bankroll_actual += ganancia
            else:
                ganancia = None

            # Actualizar
            self.db.actualizar_resultado(
                prediccion_id=row["prediccion_id"], resultado=row["resultado"], ganancia=ganancia
            )

            resultado_str = "✅ GANÓ" if row["resultado"] == 1 else "❌ PERDIÓ"
            ganancia_str = f"{ganancia:+.2f}€" if ganancia is not None else "N/A"
            print(f"ID {row['prediccion_id']}: {resultado_str}, Ganancia: {ganancia_str}")

            if self.usar_kelly and ganancia is not None:
                roi_actual = ((self.bankroll_actual / self.bankroll_inicial) - 1) * 100
                print(f"   💰 Bankroll: {self.bankroll_actual:.2f}€ (ROI: {roi_actual:+.1f}%)")

        print(f"\n✅ {len(resultados_reales)} resultados actualizados")

    def actualizar_resultado_y_bankroll(self, prediccion_id, resultado_real):
        """
        Actualiza el resultado de una predicción y el bankroll (para uso con Kelly)

        Args:
            prediccion_id: ID de la predicción
            resultado_real: 1 si ganó, 0 si perdió

        Returns:
            dict con información de la actualización
        """
        # Obtener información de la predicción
        df_pred = self.db.obtener_predicciones()
        df_pred = df_pred[df_pred["id"] == prediccion_id]

        if len(df_pred) == 0:
            print(f"⚠️  Predicción {prediccion_id} no encontrada")
            return None

        pred = df_pred.iloc[0]
        apuesta = pred["apuesta_cantidad"]
        cuota = pred["cuota"]

        # Calcular ganancia/pérdida
        if resultado_real == 1:
            ganancia = apuesta * (cuota - 1)
        else:
            ganancia = -apuesta

        # Actualizar bankroll si Kelly está activado
        if self.usar_kelly:
            self.bankroll_actual += ganancia

        # Actualizar en DB
        self.db.actualizar_resultado(
            prediccion_id=prediccion_id, resultado=resultado_real, ganancia=ganancia
        )

        # Mostrar resultado
        resultado_texto = "✅ GANÓ" if resultado_real == 1 else "❌ PERDIÓ"
        print(f"\n{resultado_texto} - Predicción #{prediccion_id}")
        print(f"   Apuesta: {apuesta:.2f}€")
        print(f"   Ganancia: {ganancia:+.2f}€")

        if self.usar_kelly:
            roi = ((self.bankroll_actual / self.bankroll_inicial) - 1) * 100
            print(f"   Bankroll: {self.bankroll_actual:.2f}€ ({roi:+.1f}%)")

        return {
            "prediccion_id": prediccion_id,
            "resultado": resultado_real,
            "apuesta": apuesta,
            "ganancia": ganancia,
            "bankroll": self.bankroll_actual if self.usar_kelly else None,
            "roi": (
                ((self.bankroll_actual / self.bankroll_inicial) - 1) * 100
                if self.usar_kelly
                else None
            ),
        }

        print(f"\n✅ {len(resultados_reales)} resultados actualizados")

    def generar_reporte(self):
        """
        Genera un reporte completo del sistema (incluye métricas de Kelly si está activado)
        """

        metricas = self.db.calcular_metricas()

        print("\n" + "=" * 60)
        if self.usar_kelly:
            print("📊 REPORTE COMPLETO - CON KELLY CRITERION")
        else:
            print("📊 REPORTE COMPLETO")
        print("=" * 60)

        print(f"\n💰 FINANCIERO:")
        print(f"   Total apostado:   {metricas['total_apostado']:.2f}€")
        print(f"   Total retornado:  {metricas['total_ganado']:.2f}€")
        print(f"   Ganancia neta:    {metricas['ganancia_neta']:+.2f}€")
        print(f"   ROI:              {metricas['roi']:+.2f}%")

        print(f"\n🎯 PERFORMANCE:")
        print(f"   Apuestas totales: {metricas['total_apuestas']}")
        print(f"   Ganadas:          {metricas['apuestas_ganadas']}")
        print(f"   Perdidas:         {metricas['apuestas_perdidas']}")
        print(f"   Win Rate:         {metricas['win_rate']:.1f}%")

        print(f"\n📈 MODELO:")
        print(f"   EV promedio:      +{metricas['ev_promedio']:.2f}%")

        # Métricas adicionales de Kelly
        if self.usar_kelly:
            print("\n" + "=" * 60)
            print("💎 GESTIÓN DE BANKROLL (KELLY CRITERION)")
            print("=" * 60)

            print(f"\n💵 Bankroll:")
            print(f"   Inicial:  {self.bankroll_inicial:,.2f}€")
            print(f"   Actual:   {self.bankroll_actual:,.2f}€")
            print(f"   Cambio:   {self.bankroll_actual - self.bankroll_inicial:+,.2f}€")

            roi_bankroll = ((self.bankroll_actual / self.bankroll_inicial) - 1) * 100
            print(f"\n📈 ROI Bankroll: {roi_bankroll:+.2f}%")

            # Distribución de apuestas
            apuestas = pd.read_sql_query(
                """
                SELECT apuesta_cantidad, cuota, prob_modelo, ganancia
                FROM predicciones
                WHERE decision LIKE '%APOSTAR%' AND apuesta_cantidad IS NOT NULL
                ORDER BY fecha_prediccion DESC
            """,
                self.db.conn,
            )

            if len(apuestas) > 0:
                print(f"\n💰 Distribución de Apuestas:")
                print(f"   Media:    {apuestas['apuesta_cantidad'].mean():.2f}€")
                print(f"   Mediana:  {apuestas['apuesta_cantidad'].median():.2f}€")
                print(f"   Mínima:   {apuestas['apuesta_cantidad'].min():.2f}€")
                print(f"   Máxima:   {apuestas['apuesta_cantidad'].max():.2f}€")

        return metricas


# Ejemplo de uso
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎾 TRACKING SYSTEM - TENNIS ML PREDICTOR")
    print("=" * 60)

    # Ejemplo 1: Sistema básico (sin Kelly)
    print("\n📊 Ejemplo 1: Sistema básico")
    sistema_basico = TrackingSystem(
        modelo_path="modelos/random_forest_calibrado.pkl", db_path="apuestas_tracker.db"
    )
    print("✅ Sistema de tracking inicializado (Flat Betting)")

    # Ejemplo 2: Sistema con Kelly Criterion
    print("\n💎 Ejemplo 2: Sistema con Kelly Criterion")
    sistema_kelly = TrackingSystem(
        modelo_path="modelos/random_forest_calibrado.pkl",
        db_path="apuestas_tracker_kelly.db",
        bankroll_actual=1000,
        usar_kelly=True,
        kelly_fraccion=0.25,
    )
    print("✅ Sistema de tracking inicializado (Kelly 25%)")

    # Generar reporte
    print("\n" + "=" * 60)
    sistema_basico.generar_reporte()
