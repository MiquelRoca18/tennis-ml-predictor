"""
Kelly Criterion Calculator para optimización de apuestas

Este módulo implementa el Kelly Criterion para calcular el tamaño óptimo
de las apuestas basándose en la probabilidad del modelo y las cuotas del bookmaker.
"""

import numpy as np


class KellyCalculator:
    """
    Calculadora de Kelly Criterion con límites de seguridad

    El Kelly Criterion es una fórmula matemática que maximiza el crecimiento
    logarítmico del bankroll a largo plazo.

    Fórmula: f = (bp - q) / b
    Donde:
        f = fracción del bankroll a apostar
        b = ganancia neta por euro (cuota - 1)
        p = probabilidad de ganar (del modelo)
        q = probabilidad de perder (1 - p)
    """

    def __init__(self, fraccion=0.25):
        """
        Inicializa el calculador de Kelly

        Args:
            fraccion: Fracción del Kelly completo a usar (default: 0.25 = 25%)
                     Kelly completo es muy agresivo, por eso se usa fraccional
        """
        self.fraccion = fraccion

    def calcular_kelly(self, prob, cuota, bankroll):
        """
        Calcula el tamaño de apuesta óptimo usando Kelly Criterion

        Args:
            prob: Probabilidad de ganar según el modelo (0-1)
            cuota: Cuota decimal del bookmaker (ej: 2.00)
            bankroll: Capital actual disponible

        Returns:
            float: Cantidad óptima a apostar en euros
        """
        # Validaciones
        if prob <= 0 or prob >= 1:
            return 0
        if cuota <= 1:
            return 0
        if bankroll <= 0:
            return 0

        q = 1 - prob  # Probabilidad de perder
        b = cuota - 1  # Ganancia neta por euro apostado

        # Kelly completo
        kelly_full = (b * prob - q) / b

        # Si Kelly es negativo o cero, no hay ventaja -> no apostar
        if kelly_full <= 0:
            return 0

        # Kelly fraccional (más conservador)
        kelly_frac = kelly_full * self.fraccion

        # Cantidad a apostar
        apuesta = bankroll * kelly_frac

        return apuesta

    def calcular_con_limites(self, prob, cuota, bankroll, min_apuesta=5, max_apuesta_pct=0.05):
        """
        Calcula Kelly con límites de seguridad adicionales

        Args:
            prob: Probabilidad de ganar (0-1)
            cuota: Cuota decimal
            bankroll: Capital actual
            min_apuesta: Apuesta mínima en euros (default: 5€)
            max_apuesta_pct: Máximo porcentaje del bankroll (default: 0.05 = 5%)

        Returns:
            float: Cantidad a apostar con límites aplicados
        """
        # Calcular Kelly base
        apuesta = self.calcular_kelly(prob, cuota, bankroll)

        # Aplicar límites de seguridad
        max_apuesta = bankroll * max_apuesta_pct

        if apuesta < min_apuesta:
            # Si es muy poco, mejor no apostar (no vale la pena)
            apuesta = 0
        elif apuesta > max_apuesta:
            # Limitar al máximo permitido (protección contra errores del modelo)
            apuesta = max_apuesta

        return apuesta

    def calcular_ev(self, prob, cuota):
        """
        Calcula el Expected Value (EV) de una apuesta

        Args:
            prob: Probabilidad de ganar (0-1)
            cuota: Cuota decimal

        Returns:
            float: EV como porcentaje (ej: 0.05 = +5% EV)
        """
        ev = (prob * cuota) - 1
        return ev

    def comparar_estrategias(self, prob, cuota, bankroll=1000, apuesta_flat=10):
        """
        Compara Kelly vs Flat Betting para una apuesta específica

        Args:
            prob: Probabilidad de ganar
            cuota: Cuota decimal
            bankroll: Bankroll de ejemplo (default: 1000€)
            apuesta_flat: Cantidad fija para flat betting (default: 10€)

        Returns:
            dict: Comparación de ambas estrategias
        """
        # Calcular Kelly
        apuesta_kelly = self.calcular_kelly(prob, cuota, bankroll)

        # Calcular EV
        ev = self.calcular_ev(prob, cuota)

        # Resultados
        print(f"\n{'='*60}")
        print(f"📊 COMPARACIÓN DE ESTRATEGIAS")
        print(f"{'='*60}")
        print(f"\n📈 Análisis del Partido:")
        print(f"   Probabilidad modelo: {prob*100:.1f}%")
        print(f"   Cuota bookmaker:     @{cuota:.2f}")
        print(f"   Expected Value (EV): {ev*100:+.2f}%")

        print(f"\n💰 Tamaño de Apuesta:")
        print(
            f"   Flat Betting:  {apuesta_flat:.2f}€ ({apuesta_flat/bankroll*100:.1f}% del bankroll)"
        )
        print(
            f"   Kelly (25%):   {apuesta_kelly:.2f}€ ({apuesta_kelly/bankroll*100:.1f}% del bankroll)"
        )

        if apuesta_kelly > apuesta_flat:
            print(f"\n✅ Kelly apuesta MÁS (mayor ventaja detectada)")
        elif apuesta_kelly < apuesta_flat:
            print(f"\n⚠️  Kelly apuesta MENOS (ventaja marginal)")
        else:
            print(f"\n➡️  Ambas estrategias apuestan igual")

        return {
            "prob": prob,
            "cuota": cuota,
            "ev": ev,
            "apuesta_flat": apuesta_flat,
            "apuesta_kelly": apuesta_kelly,
            "pct_flat": apuesta_flat / bankroll,
            "pct_kelly": apuesta_kelly / bankroll,
        }

    def analizar_sensibilidad(self, cuota, bankroll=1000):
        """
        Analiza cómo varía Kelly según diferentes probabilidades

        Args:
            cuota: Cuota fija del bookmaker
            bankroll: Bankroll de ejemplo
        """
        print(f"\n{'='*60}")
        print(f"📊 ANÁLISIS DE SENSIBILIDAD - Cuota @{cuota:.2f}")
        print(f"{'='*60}")
        print(f"\n{'Prob':<8} {'EV':<10} {'Kelly Full':<12} {'Kelly 25%':<12} {'Apuesta':<10}")
        print(f"{'-'*60}")

        for prob in np.arange(0.45, 0.75, 0.05):
            ev = self.calcular_ev(prob, cuota)

            # Kelly completo
            q = 1 - prob
            b = cuota - 1
            kelly_full = max(0, (b * prob - q) / b)

            # Kelly fraccional
            apuesta = self.calcular_kelly(prob, cuota, bankroll)

            print(
                f"{prob*100:5.1f}%   {ev*100:+6.2f}%   {kelly_full*100:8.2f}%   "
                f"{(apuesta/bankroll)*100:8.2f}%   {apuesta:7.2f}€"
            )


# Ejemplos de uso
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("💰 KELLY CRITERION CALCULATOR - EJEMPLOS")
    print("=" * 60)

    calc = KellyCalculator(fraccion=0.25)

    # Ejemplo 1: Alto valor
    print("\n\n🎯 EJEMPLO 1: Alto valor (EV +20%)")
    calc.comparar_estrategias(prob=0.60, cuota=2.00)

    # Ejemplo 2: Valor moderado
    print("\n\n🎯 EJEMPLO 2: Valor moderado (EV +10%)")
    calc.comparar_estrategias(prob=0.55, cuota=2.00)

    # Ejemplo 3: Bajo valor
    print("\n\n🎯 EJEMPLO 3: Bajo valor (EV +4%)")
    calc.comparar_estrategias(prob=0.52, cuota=2.00)

    # Ejemplo 4: Sin valor (no apostar)
    print("\n\n🎯 EJEMPLO 4: Sin valor (EV -2%)")
    calc.comparar_estrategias(prob=0.49, cuota=2.00)

    # Análisis de sensibilidad
    print("\n\n")
    calc.analizar_sensibilidad(cuota=2.00)

    print("\n\n✅ Ejemplos completados!")
