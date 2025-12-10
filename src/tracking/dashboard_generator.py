"""
Generador de Dashboard Interactivo
Fase 4: Tennis ML Predictor
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Añadir path para imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tracking.database_setup import TennisDatabase
from datetime import datetime


class DashboardGenerator:
    """
    Genera dashboard HTML interactivo con visualizaciones
    """
    
    def __init__(self, db_path="apuestas_tracker.db"):
        self.db = TennisDatabase(db_path)
    
    def generar_dashboard_completo(self, output_path="resultados/dashboard.html"):
        """
        Genera dashboard HTML completo con todas las visualizaciones
        """
        
        print("=" * 60)
        print("📊 GENERANDO DASHBOARD")
        print("=" * 60)
        
        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Obtener datos
        df_todas = self.db.obtener_predicciones()
        df_apuestas = df_todas[df_todas['decision'] == 'APOSTAR']
        df_completadas = df_apuestas[df_apuestas['resultado_real'].notna()]
        
        # Crear figuras
        fig_curva_ganancias = self._crear_curva_ganancias(df_completadas)
        fig_win_rate = self._crear_distribucion_win_rate(df_completadas)
        fig_ev_distribucion = self._crear_distribucion_ev(df_apuestas)
        fig_performance_superficie = self._crear_performance_superficie(df_completadas)
        fig_ev_vs_resultado = self._crear_ev_vs_resultado(df_completadas)
        
        # Calcular métricas resumen
        metricas = self.db.calcular_metricas()
        
        # Generar HTML
        html_content = self._crear_html_dashboard(
            metricas=metricas,
            fig_curva=fig_curva_ganancias,
            fig_win_rate=fig_win_rate,
            fig_ev=fig_ev_distribucion,
            fig_superficie=fig_performance_superficie,
            fig_ev_resultado=fig_ev_vs_resultado,
            df_apuestas=df_completadas
        )
        
        # Guardar
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ Dashboard generado: {output_path}")
        print(f"💡 Abre el archivo en tu navegador para visualizar")
    
    def _crear_curva_ganancias(self, df):
        """Curva de ganancias acumuladas"""
        
        if len(df) == 0:
            return go.Figure().add_annotation(
                text="No hay datos disponibles",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        df = df.sort_values('fecha_prediccion')
        df['ganancia_acumulada'] = df['ganancia'].cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(df) + 1)),
            y=df['ganancia_acumulada'],
            mode='lines',
            name='Ganancia Acumulada',
            line=dict(color='#2ecc71', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.1)'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title='Curva de Ganancias Acumuladas',
            xaxis_title='Número de Apuesta',
            yaxis_title='Ganancia (€)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _crear_distribucion_win_rate(self, df):
        """Distribución de resultados"""
        
        if len(df) == 0:
            return go.Figure().add_annotation(
                text="No hay datos disponibles",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        ganadas = (df['resultado_real'] == 1).sum()
        perdidas = (df['resultado_real'] == 0).sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Ganadas', 'Perdidas'],
            values=[ganadas, perdidas],
            hole=0.4,
            marker_colors=['#2ecc71', '#e74c3c'],
            textinfo='label+percent',
            textfont_size=14
        )])
        
        fig.update_layout(
            title=f'Win Rate: {ganadas/(ganadas+perdidas)*100:.1f}%',
            height=400
        )
        
        return fig
    
    def _crear_distribucion_ev(self, df):
        """Distribución de Expected Value"""
        
        if len(df) == 0:
            return go.Figure().add_annotation(
                text="No hay datos disponibles",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['ev'] * 100,
            nbinsx=20,
            marker_color='#3498db',
            name='EV Distribution'
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_vline(x=df['ev'].mean() * 100, line_dash="solid", 
                     line_color="green", opacity=0.7,
                     annotation_text=f"Media: {df['ev'].mean()*100:.1f}%")
        
        fig.update_layout(
            title='Distribución de Expected Value',
            xaxis_title='EV (%)',
            yaxis_title='Frecuencia',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _crear_performance_superficie(self, df):
        """Performance por superficie"""
        
        if len(df) == 0 or df['superficie'].isna().all():
            return go.Figure().add_annotation(
                text="No hay datos disponibles",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Agrupar por superficie
        superficie_stats = df.groupby('superficie').agg({
            'resultado_real': ['count', 'sum', 'mean'],
            'ganancia': 'sum',
            'apuesta_cantidad': 'sum'
        }).reset_index()
        
        superficie_stats.columns = ['superficie', 'total', 'ganadas', 'win_rate', 'ganancia', 'apostado']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Win Rate por Superficie', 'ROI por Superficie'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Win Rate
        fig.add_trace(
            go.Bar(
                x=superficie_stats['superficie'],
                y=superficie_stats['win_rate'] * 100,
                name='Win Rate',
                marker_color='#3498db',
                text=superficie_stats['win_rate'].apply(lambda x: f'{x*100:.1f}%'),
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # ROI
        roi_por_superficie = (superficie_stats['ganancia'] / superficie_stats['apostado']) * 100
        fig.add_trace(
            go.Bar(
                x=superficie_stats['superficie'],
                y=roi_por_superficie,
                name='ROI',
                marker_color='#2ecc71',
                text=roi_por_superficie.apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_yaxes(title_text="Win Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="ROI (%)", row=1, col=2)
        fig.update_layout(height=400, template='plotly_white', showlegend=False)
        
        return fig
    
    def _crear_ev_vs_resultado(self, df):
        """EV vs Resultado real"""
        
        if len(df) == 0:
            return go.Figure().add_annotation(
                text="No hay datos disponibles",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        # Scatter plot
        colors = ['#2ecc71' if x == 1 else '#e74c3c' for x in df['resultado_real']]
        
        fig.add_trace(go.Scatter(
            x=df['ev'] * 100,
            y=df['ganancia'],
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            text=df['jugador_nombre'] + ' vs ' + df['oponente_nombre'],
            hovertemplate='<b>%{text}</b><br>EV: %{x:.2f}%<br>Ganancia: %{y:.2f}€<extra></extra>'
        ))
        
        fig.update_layout(
            title='Expected Value vs Resultado Real',
            xaxis_title='EV (%)',
            yaxis_title='Ganancia (€)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _crear_html_dashboard(self, metricas, fig_curva, fig_win_rate, 
                              fig_ev, fig_superficie, fig_ev_resultado, df_apuestas):
        """Genera HTML completo"""
        
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Tennis ML</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
            font-size: 36px;
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
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
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
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .table-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
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
        <h1>🎾 Tennis ML Dashboard</h1>
        <p>Análisis de Predicciones y Apuestas | Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    
    <div class="metrics-grid">
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
        <div class="metric-card">
            <div class="metric-label">ROI</div>
            <div class="metric-value {'positive' if metricas['roi'] > 0 else 'negative'}">
                {metricas['roi']:+.1f}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{metricas['win_rate']:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Apuestas</div>
            <div class="metric-value">{metricas['total_apuestas']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">EV Promedio</div>
            <div class="metric-value positive">+{metricas['ev_promedio']:.2f}%</div>
        </div>
    </div>
    
    <div class="chart-container">
        {fig_curva.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
        <div class="chart-container">
            {fig_win_rate.to_html(full_html=False, include_plotlyjs=False)}
        </div>
        <div class="chart-container">
            {fig_ev.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </div>
    
    <div class="chart-container">
        {fig_superficie.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    
    <div class="chart-container">
        {fig_ev_resultado.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    
    <div class="table-container">
        <h2>📋 Últimas 10 Apuestas</h2>
        {self._crear_tabla_ultimas_apuestas(df_apuestas)}
    </div>
    
    <div class="footer">
        <p>Dashboard generado automáticamente | Tennis ML Tracking System v1.0</p>
    </div>
</body>
</html>
        """
        
        return html
    
    def _crear_tabla_ultimas_apuestas(self, df):
        """Crea tabla HTML con últimas apuestas"""
        
        if len(df) == 0:
            return "<p>No hay apuestas registradas aún.</p>"
        
        df_ultimas = df.sort_values('fecha_prediccion', ascending=False).head(10)
        
        html = "<table><thead><tr>"
        html += "<th>Fecha</th><th>Partido</th><th>Superficie</th>"
        html += "<th>EV</th><th>Cuota</th><th>Apuesta</th><th>Resultado</th><th>Ganancia</th>"
        html += "</tr></thead><tbody>"
        
        for _, row in df_ultimas.iterrows():
            resultado_icon = "✅" if row['resultado_real'] == 1 else "❌" if row['resultado_real'] == 0 else "⏳"
            ganancia_color = 'green' if row.get('ganancia', 0) > 0 else 'red'
            
            html += "<tr>"
            html += f"<td>{row['fecha_partido']}</td>"
            html += f"<td>{row['jugador_nombre']} vs {row['oponente_nombre']}</td>"
            html += f"<td>{row['superficie']}</td>"
            html += f"<td>{row['ev']*100:+.1f}%</td>"
            html += f"<td>@{row['cuota']:.2f}</td>"
            html += f"<td>{row['apuesta_cantidad']:.0f}€</td>"
            html += f"<td>{resultado_icon}</td>"
            html += f"<td style='color: {ganancia_color}; font-weight: bold;'>{row.get('ganancia', 0):+.2f}€</td>"
            html += "</tr>"
        
        html += "</tbody></table>"
        
        return html


# Ejecutar
if __name__ == "__main__":
    generator = DashboardGenerator("apuestas_tracker.db")
    generator.generar_dashboard_completo("resultados/dashboard.html")
    
    print("\n✅ Dashboard generado!")
    print("💡 Abre resultados/dashboard.html en tu navegador")
