"""
OddsComparator - Comparación de cuotas entre bookmakers

Este módulo maneja:
- Comparación de cuotas de múltiples bookmakers
- Identificación de mejor cuota disponible
- Cálculo de EV con cada bookmaker
- Análisis de savings por line shopping
- Generación de reportes
"""

import pandas as pd


class OddsComparator:
    """
    Compara cuotas de múltiples bookmakers y encuentra la mejor

    Características:
    - Búsqueda de mejor cuota por jugador
    - Cálculo de EV con múltiples bookmakers
    - Análisis de diferencias de cuotas
    - Métricas de savings por line shopping
    """

    def __init__(self, df_cuotas):
        """
        Inicializa el comparador con un DataFrame de cuotas

        Args:
            df_cuotas: DataFrame con columnas:
                - bookmaker: nombre del bookmaker
                - jugador1, jugador2: nombres de jugadores
                - cuota_jugador1, cuota_jugador2: cuotas decimales
                - fecha: fecha del partido
        """
        self.df = df_cuotas

        if len(self.df) == 0:
            print("⚠️  DataFrame de cuotas vacío")
        else:
            num_partidos = self.df.groupby(["jugador1", "jugador2"]).ngroups
            num_bookmakers = self.df["bookmaker"].nunique()

            print(f"✅ OddsComparator inicializado")
            print(f"   Partidos: {num_partidos}")
            print(f"   Bookmakers: {num_bookmakers}")

    def encontrar_mejor_cuota(self, jugador1, jugador2):
        """
        Encuentra la mejor cuota disponible para un partido

        Args:
            jugador1: Nombre del jugador 1
            jugador2: Nombre del jugador 2

        Returns:
            dict con mejor cuota para cada jugador, o None si no se encuentra
        """
        # Filtrar partido (case-insensitive)
        df_partido = self.df[
            (self.df["jugador1"].str.lower() == jugador1.lower())
            & (self.df["jugador2"].str.lower() == jugador2.lower())
        ]

        if len(df_partido) == 0:
            print(f"⚠️  Partido no encontrado: {jugador1} vs {jugador2}")
            return None

        # Mejor cuota para jugador 1
        idx_mejor_j1 = df_partido["cuota_jugador1"].idxmax()
        mejor_j1 = df_partido.loc[idx_mejor_j1]

        # Mejor cuota para jugador 2
        idx_mejor_j2 = df_partido["cuota_jugador2"].idxmax()
        mejor_j2 = df_partido.loc[idx_mejor_j2]

        # Cuotas promedio (para comparación)
        cuota_promedio_j1 = df_partido["cuota_jugador1"].mean()
        cuota_promedio_j2 = df_partido["cuota_jugador2"].mean()

        resultado = {
            "jugador1": jugador1,
            "jugador2": jugador2,
            "mejor_cuota_j1": mejor_j1["cuota_jugador1"],
            "bookmaker_j1": mejor_j1["bookmaker"],
            "mejor_cuota_j2": mejor_j2["cuota_jugador2"],
            "bookmaker_j2": mejor_j2["bookmaker"],
            "cuota_promedio_j1": cuota_promedio_j1,
            "cuota_promedio_j2": cuota_promedio_j2,
            "cuotas_disponibles": len(df_partido),
            "bookmakers": df_partido["bookmaker"].unique().tolist(),
        }

        return resultado

    def calcular_ev(self, prob, cuota):
        """
        Calcula Expected Value

        Args:
            prob: Probabilidad de ganar (0-1)
            cuota: Cuota decimal

        Returns:
            float: EV como decimal (ej: 0.05 = +5%)
        """
        return (prob * cuota) - 1

    def analizar_partido_completo(self, jugador1, jugador2, prob_j1):
        """
        Análisis completo de un partido con probabilidades del modelo

        Args:
            jugador1: Nombre del jugador 1
            jugador2: Nombre del jugador 2
            prob_j1: Probabilidad de que gane jugador 1 (0-1)

        Returns:
            dict con análisis completo y mejor apuesta
        """
        mejor = self.encontrar_mejor_cuota(jugador1, jugador2)

        if mejor is None:
            return None

        prob_j2 = 1 - prob_j1

        # Calcular EV con mejor cuota
        ev_j1 = self.calcular_ev(prob_j1, mejor["mejor_cuota_j1"])
        ev_j2 = self.calcular_ev(prob_j2, mejor["mejor_cuota_j2"])

        # Calcular EV con cuota promedio (para comparar savings)
        ev_promedio_j1 = self.calcular_ev(prob_j1, mejor["cuota_promedio_j1"])
        ev_promedio_j2 = self.calcular_ev(prob_j2, mejor["cuota_promedio_j2"])

        # Determinar mejor apuesta
        if ev_j1 > ev_j2:
            mejor_apuesta = {
                "jugador": jugador1,
                "oponente": jugador2,
                "cuota": mejor["mejor_cuota_j1"],
                "cuota_promedio": mejor["cuota_promedio_j1"],
                "bookmaker": mejor["bookmaker_j1"],
                "prob": prob_j1,
                "ev": ev_j1,
                "ev_promedio": ev_promedio_j1,
                "saving_vs_promedio": ev_j1 - ev_promedio_j1,
            }
        else:
            mejor_apuesta = {
                "jugador": jugador2,
                "oponente": jugador1,
                "cuota": mejor["mejor_cuota_j2"],
                "cuota_promedio": mejor["cuota_promedio_j2"],
                "bookmaker": mejor["bookmaker_j2"],
                "prob": prob_j2,
                "ev": ev_j2,
                "ev_promedio": ev_promedio_j2,
                "saving_vs_promedio": ev_j2 - ev_promedio_j2,
            }

        # Añadir información adicional
        mejor_apuesta["bookmakers_disponibles"] = mejor["cuotas_disponibles"]
        mejor_apuesta["todos_bookmakers"] = mejor["bookmakers"]

        return mejor_apuesta

    def calcular_savings(self, mejor_apuesta, apuesta_cantidad=100):
        """
        Calcula el ahorro por usar line shopping vs cuota promedio

        Args:
            mejor_apuesta: Dict del análisis completo
            apuesta_cantidad: Cantidad a apostar (para calcular ahorro en euros)

        Returns:
            dict con métricas de savings
        """
        if mejor_apuesta is None:
            return None

        # Ganancia esperada con mejor cuota
        ganancia_mejor = apuesta_cantidad * mejor_apuesta["ev"]

        # Ganancia esperada con cuota promedio
        ganancia_promedio = apuesta_cantidad * mejor_apuesta["ev_promedio"]

        # Ahorro absoluto
        ahorro_euros = ganancia_mejor - ganancia_promedio

        # Ahorro porcentual
        ahorro_pct = mejor_apuesta["saving_vs_promedio"]

        return {
            "ganancia_mejor_cuota": ganancia_mejor,
            "ganancia_cuota_promedio": ganancia_promedio,
            "ahorro_euros": ahorro_euros,
            "ahorro_porcentual": ahorro_pct,
            "mejora_ev": ahorro_pct * 100,  # En puntos porcentuales
        }

    def generar_reporte_comparacion(self, top_n=None):
        """
        Genera reporte de comparación de todos los partidos

        Args:
            top_n: Si se especifica, muestra solo los top N partidos
        """
        partidos_unicos = self.df.groupby(["jugador1", "jugador2"]).size().reset_index(name="count")

        if top_n:
            partidos_unicos = partidos_unicos.head(top_n)

        print("\n" + "=" * 60)
        print("📊 REPORTE DE COMPARACIÓN DE CUOTAS")
        print("=" * 60)

        for idx, row in partidos_unicos.iterrows():
            mejor = self.encontrar_mejor_cuota(row["jugador1"], row["jugador2"])

            if mejor:
                print(f"\n🎾 {mejor['jugador1']} vs {mejor['jugador2']}")
                print(f"   📍 Casas disponibles: {mejor['cuotas_disponibles']}")

                # Jugador 1
                diff_j1 = mejor["mejor_cuota_j1"] - mejor["cuota_promedio_j1"]
                print(f"\n   {mejor['jugador1']}:")
                print(f"      Mejor: @{mejor['mejor_cuota_j1']:.2f} ({mejor['bookmaker_j1']})")
                print(f"      Promedio: @{mejor['cuota_promedio_j1']:.2f}")
                print(
                    f"      Diferencia: {diff_j1:+.3f} ({(diff_j1/mejor['cuota_promedio_j1'])*100:+.1f}%)"
                )

                # Jugador 2
                diff_j2 = mejor["mejor_cuota_j2"] - mejor["cuota_promedio_j2"]
                print(f"\n   {mejor['jugador2']}:")
                print(f"      Mejor: @{mejor['mejor_cuota_j2']:.2f} ({mejor['bookmaker_j2']})")
                print(f"      Promedio: @{mejor['cuota_promedio_j2']:.2f}")
                print(
                    f"      Diferencia: {diff_j2:+.3f} ({(diff_j2/mejor['cuota_promedio_j2'])*100:+.1f}%)"
                )

        print("\n" + "=" * 60)

    def obtener_todos_partidos(self):
        """
        Obtiene lista de todos los partidos únicos

        Returns:
            list: Lista de tuplas (jugador1, jugador2)
        """
        partidos = self.df.groupby(["jugador1", "jugador2"]).size().reset_index()
        return [(row["jugador1"], row["jugador2"]) for _, row in partidos.iterrows()]


# Ejemplo de uso
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔍 ODDS COMPARATOR - DEMO")
    print("=" * 60)

    # Datos de ejemplo
    datos_ejemplo = [
        {
            "bookmaker": "Bet365",
            "jugador1": "Carlos Alcaraz",
            "jugador2": "Jannik Sinner",
            "cuota_jugador1": 2.00,
            "cuota_jugador2": 1.85,
            "fecha": pd.Timestamp.now(),
        },
        {
            "bookmaker": "Pinnacle",
            "jugador1": "Carlos Alcaraz",
            "jugador2": "Jannik Sinner",
            "cuota_jugador1": 2.10,
            "cuota_jugador2": 1.80,
            "fecha": pd.Timestamp.now(),
        },
        {
            "bookmaker": "Betfair",
            "jugador1": "Carlos Alcaraz",
            "jugador2": "Jannik Sinner",
            "cuota_jugador1": 2.05,
            "cuota_jugador2": 1.83,
            "fecha": pd.Timestamp.now(),
        },
    ]

    df = pd.DataFrame(datos_ejemplo)

    # Crear comparador
    comparador = OddsComparator(df)

    # Analizar partido
    print("\n🎯 Análisis de partido:")
    resultado = comparador.analizar_partido_completo(
        jugador1="Carlos Alcaraz", jugador2="Jannik Sinner", prob_j1=0.48
    )

    if resultado:
        print(f"\n🏆 MEJOR OPORTUNIDAD:")
        print(f"   Apostar a: {resultado['jugador']}")
        print(f"   Bookmaker: {resultado['bookmaker']}")
        print(f"   Cuota: @{resultado['cuota']:.2f}")
        print(f"   Probabilidad: {resultado['prob']*100:.1f}%")
        print(f"   EV: {resultado['ev']*100:+.2f}%")

        # Calcular savings
        savings = comparador.calcular_savings(resultado, apuesta_cantidad=100)
        print(f"\n💰 SAVINGS por Line Shopping (apuesta de 100€):")
        print(f"   Ganancia esperada (mejor cuota): {savings['ganancia_mejor_cuota']:+.2f}€")
        print(f"   Ganancia esperada (cuota promedio): {savings['ganancia_cuota_promedio']:+.2f}€")
        print(f"   Ahorro: {savings['ahorro_euros']:+.2f}€")
        print(f"   Mejora de EV: {savings['mejora_ev']:+.2f} puntos porcentuales")

    # Reporte completo
    comparador.generar_reporte_comparacion()

    print(f"\n✅ Demo completado!")
