"""
Sistema de Reportes Automáticos Periódicos
Fase 4: Tennis ML Predictor - Mejoras
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Añadir path para imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tracking.database_setup import TennisDatabase
from src.tracking.dashboard_generator import DashboardGenerator


class ReportePeriodico:
    """
    Genera reportes automáticos por periodo (semanal/mensual)
    """

    def __init__(self, db_path="apuestas_tracker.db"):
        self.db = TennisDatabase(db_path)
        self.dashboard_gen = DashboardGenerator(db_path)

    def generar_reporte_semanal(self, output_dir="resultados/reportes"):
        """
        Genera reporte de la última semana
        """
        print("=" * 60)
        print("📅 GENERANDO REPORTE SEMANAL")
        print("=" * 60)

        fecha_inicio = datetime.now() - timedelta(days=7)

        df = self.db.obtener_predicciones()
        df["fecha_prediccion"] = pd.to_datetime(df["fecha_prediccion"])
        df_semana = df[df["fecha_prediccion"] >= fecha_inicio]

        # Calcular métricas semanales
        metricas = self._calcular_metricas_periodo(df_semana)

        if metricas is None:
            print("⚠️  No hay datos suficientes para el reporte semanal")
            return None

        # Crear directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generar reporte HTML
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = f"{output_dir}/reporte_semanal_{timestamp}.html"

        html_content = self._crear_reporte_html(metricas, "Semanal", df_semana)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\n✅ Reporte semanal generado: {output_path}")
        self._mostrar_resumen(metricas, "Semanal")

        return metricas

    def generar_reporte_mensual(self, output_dir="resultados/reportes"):
        """
        Genera reporte del último mes
        """
        print("=" * 60)
        print("📅 GENERANDO REPORTE MENSUAL")
        print("=" * 60)

        fecha_inicio = datetime.now() - timedelta(days=30)

        df = self.db.obtener_predicciones()
        df["fecha_prediccion"] = pd.to_datetime(df["fecha_prediccion"])
        df_mes = df[df["fecha_prediccion"] >= fecha_inicio]

        # Calcular métricas mensuales
        metricas = self._calcular_metricas_periodo(df_mes)

        if metricas is None:
            print("⚠️  No hay datos suficientes para el reporte mensual")
            return None

        # Crear directorio si no existe
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generar reporte HTML
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = f"{output_dir}/reporte_mensual_{timestamp}.html"

        html_content = self._crear_reporte_html(metricas, "Mensual", df_mes)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\n✅ Reporte mensual generado: {output_path}")
        self._mostrar_resumen(metricas, "Mensual")

        return metricas

    def comparar_periodos(self, dias_actual=7, dias_anterior=7):
        """
        Compara dos periodos y muestra tendencias
        """
        print("=" * 60)
        print("📊 COMPARACIÓN DE PERIODOS")
        print("=" * 60)

        # Periodo actual
        fecha_inicio_actual = datetime.now() - timedelta(days=dias_actual)
        df = self.db.obtener_predicciones()
        df["fecha_prediccion"] = pd.to_datetime(df["fecha_prediccion"])
        df_actual = df[df["fecha_prediccion"] >= fecha_inicio_actual]

        # Periodo anterior
        fecha_fin_anterior = fecha_inicio_actual
        fecha_inicio_anterior = fecha_fin_anterior - timedelta(days=dias_anterior)
        df_anterior = df[
            (df["fecha_prediccion"] >= fecha_inicio_anterior)
            & (df["fecha_prediccion"] < fecha_fin_anterior)
        ]

        metricas_actual = self._calcular_metricas_periodo(df_actual)
        metricas_anterior = self._calcular_metricas_periodo(df_anterior)

        if metricas_actual is None or metricas_anterior is None:
            print("⚠️  No hay datos suficientes para comparar")
            return

        # Mostrar comparación
        print(f"\n📈 PERIODO ACTUAL (últimos {dias_actual} días):")
        self._mostrar_resumen(metricas_actual, "")

        print(f"\n📉 PERIODO ANTERIOR ({dias_anterior} días previos):")
        self._mostrar_resumen(metricas_anterior, "")

        print("\n🔄 TENDENCIAS:")
        self._mostrar_tendencias(metricas_actual, metricas_anterior)

    def _calcular_metricas_periodo(self, df):
        """Calcula métricas para un periodo específico"""
        df_apuestas = df[df["decision"] == "APOSTAR"]
        df_completadas = df_apuestas[df_apuestas["resultado_real"].notna()]

        if len(df_completadas) == 0:
            return None

        total_apostado = df_completadas["apuesta_cantidad"].sum()
        ganancia_neta = df_completadas["ganancia"].sum()
        roi = (ganancia_neta / total_apostado) * 100 if total_apostado > 0 else 0

        ganadas = (df_completadas["resultado_real"] == 1).sum()
        win_rate = (ganadas / len(df_completadas)) * 100

        return {
            "total_predicciones": len(df),
            "total_apuestas": len(df_completadas),
            "ganadas": int(ganadas),
            "perdidas": len(df_completadas) - int(ganadas),
            "win_rate": win_rate,
            "total_apostado": total_apostado,
            "ganancia_neta": ganancia_neta,
            "roi": roi,
            "ev_promedio": df_completadas["ev"].mean() * 100,
        }

    def _mostrar_resumen(self, metricas, periodo):
        """Muestra resumen de métricas en consola"""
        if periodo:
            print(f"\n📊 RESUMEN {periodo.upper()}:")
        print(f"   Total predicciones: {metricas['total_predicciones']}")
        print(f"   Apuestas realizadas: {metricas['total_apuestas']}")
        print(f"   Ganadas: {metricas['ganadas']}")
        print(f"   Perdidas: {metricas['perdidas']}")
        print(f"   Win Rate: {metricas['win_rate']:.1f}%")
        print(f"   Total apostado: {metricas['total_apostado']:.2f}€")
        print(f"   Ganancia neta: {metricas['ganancia_neta']:+.2f}€")
        print(f"   ROI: {metricas['roi']:+.2f}%")
        print(f"   EV promedio: +{metricas['ev_promedio']:.2f}%")

    def _mostrar_tendencias(self, actual, anterior):
        """Muestra tendencias entre dos periodos"""

        def calcular_cambio(actual_val, anterior_val):
            if anterior_val == 0:
                return 0
            return ((actual_val - anterior_val) / abs(anterior_val)) * 100

        cambio_roi = calcular_cambio(actual["roi"], anterior["roi"])
        cambio_wr = calcular_cambio(actual["win_rate"], anterior["win_rate"])
        cambio_ev = calcular_cambio(actual["ev_promedio"], anterior["ev_promedio"])

        def emoji_tendencia(cambio):
            if cambio > 5:
                return "📈 ⬆️"
            elif cambio < -5:
                return "📉 ⬇️"
            else:
                return "➡️"

        print(f"   ROI: {emoji_tendencia(cambio_roi)} {cambio_roi:+.1f}%")
        print(f"   Win Rate: {emoji_tendencia(cambio_wr)} {cambio_wr:+.1f}%")
        print(f"   EV Promedio: {emoji_tendencia(cambio_ev)} {cambio_ev:+.1f}%")

    def _crear_reporte_html(self, metricas, periodo, df_periodo):
        """Crea HTML del reporte"""

        # Análisis por superficie del periodo
        df_apuestas = df_periodo[df_periodo["decision"] == "APOSTAR"]
        df_completadas = df_apuestas[df_apuestas["resultado_real"].notna()]

        analisis_superficie = ""
        if len(df_completadas) > 0 and not df_completadas["superficie"].isna().all():
            superficie_stats = (
                df_completadas.groupby("superficie")
                .agg({"resultado_real": ["count", "sum"], "ganancia": "sum"})
                .reset_index()
            )

            if len(superficie_stats) > 0:
                analisis_superficie = "<h2>📊 Análisis por Superficie</h2><table>"
                analisis_superficie += "<tr><th>Superficie</th><th>Apuestas</th><th>Ganadas</th><th>Win Rate</th><th>Ganancia</th></tr>"

                for _, row in superficie_stats.iterrows():
                    superficie = row["superficie"]
                    total = row[("resultado_real", "count")]
                    ganadas = row[("resultado_real", "sum")]
                    wr = (ganadas / total * 100) if total > 0 else 0
                    ganancia = row[("ganancia", "sum")]

                    analisis_superficie += f"<tr><td>{superficie}</td><td>{total}</td><td>{ganadas}</td><td>{wr:.1f}%</td><td>{ganancia:+.2f}€</td></tr>"

                analisis_superficie += "</table>"

        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte {periodo} - Tennis ML</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .metric-value.positive {{
            color: #2ecc71;
        }}
        .metric-value.negative {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Reporte {periodo}</h1>
        <p>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Predicciones</div>
            <div class="metric-value">{metricas['total_predicciones']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Apuestas Realizadas</div>
            <div class="metric-value">{metricas['total_apuestas']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{metricas['win_rate']:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ROI</div>
            <div class="metric-value {'positive' if metricas['roi'] > 0 else 'negative'}">
                {metricas['roi']:+.2f}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Apostado</div>
            <div class="metric-value">{metricas['total_apostado']:.2f}€</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Ganancia Neta</div>
            <div class="metric-value {'positive' if metricas['ganancia_neta'] > 0 else 'negative'}">
                {metricas['ganancia_neta']:+.2f}€
            </div>
        </div>
    </div>
    
    {analisis_superficie}
    
    <div class="footer">
        <p>Reporte generado automáticamente | Tennis ML Tracking System</p>
    </div>
</body>
</html>
        """

        return html


# Ejemplo de uso
if __name__ == "__main__":
    reporte = ReportePeriodico("apuestas_tracker.db")

    # Generar reporte semanal
    reporte.generar_reporte_semanal()

    # Generar reporte mensual
    reporte.generar_reporte_mensual()

    # Comparar periodos
    reporte.comparar_periodos(dias_actual=7, dias_anterior=7)
