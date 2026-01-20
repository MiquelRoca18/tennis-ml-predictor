"""
Match Statistics Service
========================

Servicio para calcular estadísticas detalladas de partidos de tenis.
Procesa datos de la API (scores, pointbypoint) y genera:
- Estadísticas básicas (juegos, sets)
- Estadísticas avanzadas (% saque, break points)
- Análisis profundo (momentum, timeline, puntos clave)
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class MatchStatsService:
    """
    Servicio para calcular estadísticas de partidos
    """

    def calcular_estadisticas_basicas(self, scores: List[Dict]) -> Dict:
        """
        Calcula estadísticas básicas del partido

        Args:
            scores: Lista de sets con marcadores
                [{"score_first": "6", "score_second": "4", "score_set": "1"}, ...]

        Returns:
            Dict con estadísticas básicas
        """
        if not scores:
            return {}

        total_juegos_j1 = sum(int(s.get("score_first", 0)) for s in scores)
        total_juegos_j2 = sum(int(s.get("score_second", 0)) for s in scores)
        total_sets = len(scores)

        sets_ganados_j1 = sum(
            1 for s in scores if int(s.get("score_first", 0)) > int(s.get("score_second", 0))
        )
        sets_ganados_j2 = total_sets - sets_ganados_j1

        return {
            "total_sets": total_sets,
            "sets_ganados_jugador1": sets_ganados_j1,
            "sets_ganados_jugador2": sets_ganados_j2,
            "total_juegos": total_juegos_j1 + total_juegos_j2,
            "juegos_ganados_jugador1": total_juegos_j1,
            "juegos_ganados_jugador2": total_juegos_j2,
            "marcador_por_sets": [
                {
                    "set": int(s.get("score_set", 0)),
                    "jugador1": int(s.get("score_first", 0)),
                    "jugador2": int(s.get("score_second", 0)),
                }
                for s in scores
            ],
        }

    def calcular_estadisticas_avanzadas(
        self, pointbypoint: List[Dict], scores: List[Dict]
    ) -> Dict:
        """
        Calcula estadísticas avanzadas del partido

        Args:
            pointbypoint: Datos punto por punto
            scores: Marcadores por set

        Returns:
            Dict con estadísticas avanzadas para cada jugador
        """
        if not pointbypoint:
            return {}

        stats_j1 = {
            "juegos_al_saque": 0,
            "juegos_al_resto": 0,
            "juegos_ganados_al_saque": 0,
            "juegos_ganados_al_resto": 0,
            "break_points_enfrentados": 0,
            "break_points_salvados": 0,
            "break_points_a_favor": 0,
            "break_points_convertidos": 0,
            "puntos_totales": 0,
        }

        stats_j2 = stats_j1.copy()

        # Procesar cada juego
        for game in pointbypoint:
            servidor = game.get("player_served")
            ganador = game.get("serve_winner")
            puntos = game.get("points", [])

            # Contar puntos totales
            for punto in puntos:
                # Determinar quién ganó el punto (simplificado)
                stats_j1["puntos_totales"] += 1  # Placeholder

            # Estadísticas de saque
            if servidor == "First Player":
                stats_j1["juegos_al_saque"] += 1
                stats_j2["juegos_al_resto"] += 1
                if ganador == "First Player":
                    stats_j1["juegos_ganados_al_saque"] += 1
                else:
                    stats_j2["juegos_ganados_al_resto"] += 1
            else:
                stats_j2["juegos_al_saque"] += 1
                stats_j1["juegos_al_resto"] += 1
                if ganador == "Second Player":
                    stats_j2["juegos_ganados_al_saque"] += 1
                else:
                    stats_j1["juegos_ganados_al_resto"] += 1

            # Break points
            for punto in puntos:
                if punto.get("break_point"):
                    if servidor == "First Player":
                        stats_j1["break_points_enfrentados"] += 1
                        stats_j2["break_points_a_favor"] += 1
                        if ganador != "First Player":
                            stats_j2["break_points_convertidos"] += 1
                        else:
                            stats_j1["break_points_salvados"] += 1
                    else:
                        stats_j2["break_points_enfrentados"] += 1
                        stats_j1["break_points_a_favor"] += 1
                        if ganador != "Second Player":
                            stats_j1["break_points_convertidos"] += 1
                        else:
                            stats_j2["break_points_salvados"] += 1

        # Calcular porcentajes
        stats_j1["porcentaje_saque"] = (
            (stats_j1["juegos_ganados_al_saque"] / stats_j1["juegos_al_saque"] * 100)
            if stats_j1["juegos_al_saque"] > 0
            else 0
        )
        stats_j2["porcentaje_saque"] = (
            (stats_j2["juegos_ganados_al_saque"] / stats_j2["juegos_al_saque"] * 100)
            if stats_j2["juegos_al_saque"] > 0
            else 0
        )

        return {"jugador1": stats_j1, "jugador2": stats_j2}

    def extraer_puntos_clave(self, pointbypoint: List[Dict]) -> List[Dict]:
        """
        Extrae puntos clave del partido (break points, set points, match points)

        Args:
            pointbypoint: Datos punto por punto

        Returns:
            Lista de puntos clave con contexto
        """
        puntos_clave = []

        for game in pointbypoint:
            set_num = game.get("set_number")
            game_num = game.get("number_game")
            puntos = game.get("points", [])

            for punto in puntos:
                # Break point
                if punto.get("break_point"):
                    puntos_clave.append(
                        {
                            "tipo": "break_point",
                            "set": set_num,
                            "juego": game_num,
                            "punto": punto.get("number_point"),
                            "marcador": punto.get("score"),
                            "descripcion": f"Break point en {set_num}, juego {game_num}",
                        }
                    )

                # Set point
                if punto.get("set_point"):
                    puntos_clave.append(
                        {
                            "tipo": "set_point",
                            "set": set_num,
                            "juego": game_num,
                            "punto": punto.get("number_point"),
                            "marcador": punto.get("score"),
                            "descripcion": f"Set point en {set_num}, juego {game_num}",
                        }
                    )

                # Match point
                if punto.get("match_point"):
                    puntos_clave.append(
                        {
                            "tipo": "match_point",
                            "set": set_num,
                            "juego": game_num,
                            "punto": punto.get("number_point"),
                            "marcador": punto.get("score"),
                            "descripcion": f"Match point en {set_num}, juego {game_num}",
                        }
                    )

        return puntos_clave

    def generar_timeline(self, pointbypoint: List[Dict]) -> List[Dict]:
        """
        Genera timeline del partido juego por juego

        Args:
            pointbypoint: Datos punto por punto

        Returns:
            Timeline con marcador acumulado
        """
        timeline = []
        marcador_sets = {"jugador1": 0, "jugador2": 0}
        marcador_juegos = {"jugador1": 0, "jugador2": 0}

        current_set = None

        for game in pointbypoint:
            set_num = game.get("set_number")
            ganador = game.get("serve_winner")

            # Nuevo set
            if set_num != current_set:
                if current_set is not None:
                    # Determinar ganador del set anterior
                    if marcador_juegos["jugador1"] > marcador_juegos["jugador2"]:
                        marcador_sets["jugador1"] += 1
                    else:
                        marcador_sets["jugador2"] += 1

                current_set = set_num
                marcador_juegos = {"jugador1": 0, "jugador2": 0}

            # Actualizar juegos
            if ganador == "First Player":
                marcador_juegos["jugador1"] += 1
            else:
                marcador_juegos["jugador2"] += 1

            timeline.append(
                {
                    "set": set_num,
                    "juego": game.get("number_game"),
                    "servidor": game.get("player_served"),
                    "ganador": ganador,
                    "marcador_juegos": f"{marcador_juegos['jugador1']}-{marcador_juegos['jugador2']}",
                    "marcador_sets": f"{marcador_sets['jugador1']}-{marcador_sets['jugador2']}",
                    "fue_break": game.get("player_served") != ganador,
                }
            )

        return timeline

    def calcular_momentum(self, timeline: List[Dict]) -> List[Dict]:
        """
        Calcula el momentum del partido (quién domina en cada momento)

        Args:
            timeline: Timeline del partido

        Returns:
            Lista con momentum por juego (-100 a +100)
        """
        momentum = []
        ventana = 5  # Últimos 5 juegos

        for i, juego in enumerate(timeline):
            # Tomar últimos N juegos
            inicio = max(0, i - ventana + 1)
            juegos_recientes = timeline[inicio : i + 1]

            # Contar juegos ganados
            ganados_j1 = sum(1 for j in juegos_recientes if j["ganador"] == "First Player")
            ganados_j2 = len(juegos_recientes) - ganados_j1

            # Calcular momentum (-100 a +100)
            # +100 = Jugador 1 domina, -100 = Jugador 2 domina
            if len(juegos_recientes) > 0:
                momentum_valor = ((ganados_j1 - ganados_j2) / len(juegos_recientes)) * 100
            else:
                momentum_valor = 0

            momentum.append(
                {
                    "juego": i + 1,
                    "set": juego["set"],
                    "momentum": round(momentum_valor, 1),
                    "dominando": (
                        "jugador1" if momentum_valor > 20 else "jugador2" if momentum_valor < -20 else "equilibrado"
                    ),
                }
            )

        return momentum

    def generar_resumen_completo(
        self, scores: List[Dict], pointbypoint: List[Dict], event_data: Dict
    ) -> Dict:
        """
        Genera resumen completo del partido con todas las estadísticas

        Args:
            scores: Marcadores por set
            pointbypoint: Datos punto por punto
            event_data: Datos del evento de la API

        Returns:
            Dict con resumen completo
        """
        # Estadísticas básicas
        stats_basicas = self.calcular_estadisticas_basicas(scores)

        # Estadísticas avanzadas
        stats_avanzadas = {}
        if pointbypoint:
            stats_avanzadas = self.calcular_estadisticas_avanzadas(pointbypoint, scores)

        # Timeline y momentum
        timeline = []
        momentum = []
        if pointbypoint:
            timeline = self.generar_timeline(pointbypoint)
            momentum = self.calcular_momentum(timeline)

        # Puntos clave
        puntos_clave = []
        if pointbypoint:
            puntos_clave = self.extraer_puntos_clave(pointbypoint)

        return {
            "basicas": stats_basicas,
            "avanzadas": stats_avanzadas,
            "timeline": timeline,
            "momentum": momentum,
            "puntos_clave": puntos_clave,
            "duracion_estimada": self._estimar_duracion(len(timeline)) if timeline else None,
        }

    def _estimar_duracion(self, total_juegos: int) -> str:
        """
        Estima la duración del partido basándose en el número de juegos

        Args:
            total_juegos: Total de juegos jugados

        Returns:
            Duración estimada en formato "Xh Ymin"
        """
        # Promedio: ~4 minutos por juego
        minutos_totales = total_juegos * 4

        horas = minutos_totales // 60
        minutos = minutos_totales % 60

        if horas > 0:
            return f"{horas}h {minutos}min"
        else:
            return f"{minutos}min"
