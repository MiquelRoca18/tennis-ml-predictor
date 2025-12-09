"""
Calculadora de Expected Value para apuestas de tenis
"""
import pandas as pd
import numpy as np


def calcular_ev(probabilidad_modelo, cuota):
    """
    Calcula el Expected Value (EV) de una apuesta
    
    Args:
        probabilidad_modelo: float, probabilidad estimada por el modelo (0-1)
        cuota: float, cuota del bookmaker (ej: 2.00)
        
    Returns:
        float: EV como decimal
        
    Ejemplo:
        >>> calcular_ev(0.60, 2.00)
        0.20  # 20% EV positivo
    """
    ev = (probabilidad_modelo * cuota) - 1
    return ev


def probabilidad_implicita(cuota):
    """
    Calcula la probabilidad implícita de una cuota
    
    Args:
        cuota: float, cuota del bookmaker
        
    Returns:
        float: Probabilidad implícita (0-1)
        
    Ejemplo:
        >>> probabilidad_implicita(2.00)
        0.50  # 50%
    """
    return 1 / cuota


def tiene_valor(probabilidad_modelo, cuota, umbral_ev=0.03):
    """
    Determina si una apuesta tiene valor suficiente
    
    Args:
        probabilidad_modelo: float, probabilidad del modelo
        cuota: float, cuota del bookmaker
        umbral_ev: float, EV mínimo requerido (default: 3%)
        
    Returns:
        bool: True si tiene valor, False si no
    """
    ev = calcular_ev(probabilidad_modelo, cuota)
    return ev > umbral_ev


def calcular_ganancia_esperada(stake, probabilidad_modelo, cuota):
    """
    Calcula la ganancia esperada de una apuesta
    
    Args:
        stake: float, cantidad apostada
        probabilidad_modelo: float, probabilidad del modelo
        cuota: float, cuota del bookmaker
        
    Returns:
        float: Ganancia esperada
        
    Ejemplo:
        >>> calcular_ganancia_esperada(10, 0.60, 2.00)
        2.0  # Ganancia esperada de 2€
    """
    ev = calcular_ev(probabilidad_modelo, cuota)
    return stake * ev


def analizar_apuesta(probabilidad_modelo, cuota, stake=10, umbral_ev=0.03):
    """
    Análisis completo de una apuesta
    
    Args:
        probabilidad_modelo: float, probabilidad del modelo
        cuota: float, cuota del bookmaker
        stake: float, cantidad a apostar (default: 10)
        umbral_ev: float, EV mínimo (default: 3%)
        
    Returns:
        dict con análisis completo
    """
    
    ev = calcular_ev(probabilidad_modelo, cuota)
    prob_impl = probabilidad_implicita(cuota)
    ganancia_esp = calcular_ganancia_esperada(stake, probabilidad_modelo, cuota)
    valor = tiene_valor(probabilidad_modelo, cuota, umbral_ev)
    
    return {
        'probabilidad_modelo': probabilidad_modelo,
        'probabilidad_implicita': prob_impl,
        'cuota': cuota,
        'ev': ev,
        'ev_pct': ev * 100,
        'ganancia_esperada': ganancia_esp,
        'tiene_valor': valor,
        'decision': 'APOSTAR' if valor else 'NO APOSTAR',
        'stake_sugerido': stake if valor else 0
    }


if __name__ == "__main__":
    # Ejemplos de uso
    print("=" * 60)
    print("💰 CALCULADORA DE EXPECTED VALUE")
    print("=" * 60)
    
    # Ejemplo 1: Apuesta con valor positivo
    print("\n📌 EJEMPLO 1: Apuesta con valor")
    analisis1 = analizar_apuesta(
        probabilidad_modelo=0.60,
        cuota=2.00,
        stake=10
    )
    print(f"Probabilidad modelo: {analisis1['probabilidad_modelo']*100:.1f}%")
    print(f"Probabilidad implícita: {analisis1['probabilidad_implicita']*100:.1f}%")
    print(f"EV: {analisis1['ev_pct']:+.2f}%")
    print(f"Ganancia esperada: {analisis1['ganancia_esperada']:.2f}€")
    print(f"Decisión: {analisis1['decision']}")
    
    # Ejemplo 2: Apuesta sin valor
    print("\n📌 EJEMPLO 2: Apuesta sin valor")
    analisis2 = analizar_apuesta(
        probabilidad_modelo=0.45,
        cuota=2.00,
        stake=10
    )
    print(f"Probabilidad modelo: {analisis2['probabilidad_modelo']*100:.1f}%")
    print(f"Probabilidad implícita: {analisis2['probabilidad_implicita']*100:.1f}%")
    print(f"EV: {analisis2['ev_pct']:+.2f}%")
    print(f"Ganancia esperada: {analisis2['ganancia_esperada']:.2f}€")
    print(f"Decisión: {analisis2['decision']}")
    
    # Ejemplo 3: Apuesta con valor marginal
    print("\n📌 EJEMPLO 3: Apuesta con valor marginal")
    analisis3 = analizar_apuesta(
        probabilidad_modelo=0.52,
        cuota=2.00,
        stake=10,
        umbral_ev=0.03
    )
    print(f"Probabilidad modelo: {analisis3['probabilidad_modelo']*100:.1f}%")
    print(f"Probabilidad implícita: {analisis3['probabilidad_implicita']*100:.1f}%")
    print(f"EV: {analisis3['ev_pct']:+.2f}%")
    print(f"Ganancia esperada: {analisis3['ganancia_esperada']:.2f}€")
    print(f"Decisión: {analisis3['decision']}")
