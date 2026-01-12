"""
AlertSystem - Sistema de alertas para oportunidades de apuestas

Este módulo maneja:
- Detección de oportunidades de alto EV
- Alertas por consola
- Alertas por email (opcional)
- Priorización de oportunidades
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

from src.config import Config as BookmakerConfig


class AlertSystem:
    """
    Sistema de alertas para oportunidades de apuestas

    Características:
    - Alertas por consola (siempre)
    - Alertas por email (opcional)
    - Filtrado por umbral de EV
    - Formato HTML para emails
    """

    def __init__(self, email_enabled=None):
        """
        Inicializa el sistema de alertas

        Args:
            email_enabled: Si True, envía emails. Si None, usa config
        """
        self.email_enabled = (
            email_enabled if email_enabled is not None else BookmakerConfig.EMAIL_ENABLED
        )

        if self.email_enabled:
            # Validar configuración de email
            if not BookmakerConfig.EMAIL_ADDRESS or not BookmakerConfig.EMAIL_PASSWORD:
                print("⚠️  Email habilitado pero credenciales no configuradas")
                print("   Las alertas solo se mostrarán en consola")
                self.email_enabled = False

        print(f"✅ AlertSystem inicializado")
        print(
            f"   Email: {'✅ Habilitado' if self.email_enabled else '❌ Deshabilitado (solo consola)'}"
        )

    def verificar_oportunidades(self, resultados_prediccion, umbral_ev=None):
        """
        Verifica si hay oportunidades que alertar

        Args:
            resultados_prediccion: list de dict con predicciones
                Cada dict debe tener: jugador, cuota, prob, ev, bookmaker
            umbral_ev: EV mínimo para alertar (si None, usa config)

        Returns:
            list: Oportunidades detectadas
        """
        umbral = umbral_ev if umbral_ev is not None else BookmakerConfig.EV_THRESHOLD_ALERT

        # Filtrar oportunidades sobre el umbral
        oportunidades = [r for r in resultados_prediccion if r.get("ev", 0) > umbral]

        # Ordenar por EV descendente
        oportunidades.sort(key=lambda x: x.get("ev", 0), reverse=True)

        if len(oportunidades) > 0:
            print(f"\n🚨 {len(oportunidades)} OPORTUNIDAD(ES) DETECTADA(S)!")
            self._mostrar_consola(oportunidades)

            if self.email_enabled:
                self.enviar_alerta(oportunidades)
        else:
            print(f"\n✅ No hay oportunidades con EV > {umbral*100:.1f}% en este momento")

        return oportunidades

    def enviar_alerta(self, oportunidades):
        """
        Envía alerta por email

        Args:
            oportunidades: Lista de oportunidades a notificar
        """
        if not self.email_enabled:
            return

        # Crear mensaje
        asunto = f"🚨 {len(oportunidades)} Oportunidad(es) de Apuesta - Tennis ML"
        cuerpo = self._crear_mensaje_email(oportunidades)

        try:
            # Crear mensaje MIME
            msg = MIMEMultipart()
            msg["From"] = BookmakerConfig.EMAIL_ADDRESS
            msg["To"] = BookmakerConfig.EMAIL_RECIPIENT
            msg["Subject"] = asunto

            msg.attach(MIMEText(cuerpo, "html"))

            # Enviar email
            server = smtplib.SMTP(BookmakerConfig.SMTP_SERVER, BookmakerConfig.SMTP_PORT)
            server.starttls()
            server.login(BookmakerConfig.EMAIL_ADDRESS, BookmakerConfig.EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()

            print(f"✅ Alerta enviada por email a {BookmakerConfig.EMAIL_RECIPIENT}")

        except Exception as e:
            print(f"❌ Error enviando email: {e}")
            print(f"   Las alertas se muestran en consola")

    def _mostrar_consola(self, oportunidades):
        """
        Muestra oportunidades en consola

        Args:
            oportunidades: Lista de oportunidades
        """
        print("\n" + "=" * 60)
        print("🎯 OPORTUNIDADES DE APUESTA")
        print("=" * 60)

        for i, op in enumerate(oportunidades, 1):
            print(f"\n🏆 OPORTUNIDAD #{i}:")
            print(f"   Partido: {op.get('jugador', 'N/A')} vs {op.get('oponente', 'N/A')}")
            print(f"   Apostar a: {op.get('jugador', 'N/A')}")
            print(f"   Bookmaker: {op.get('bookmaker', 'N/A')}")
            print(f"   Cuota: @{op.get('cuota', 0):.2f}")
            print(f"   Probabilidad modelo: {op.get('prob', 0)*100:.1f}%")
            print(f"   Expected Value: {op.get('ev', 0)*100:+.2f}%")

            # Información adicional si está disponible
            if "saving_vs_promedio" in op:
                print(f"   Saving vs promedio: {op['saving_vs_promedio']*100:+.2f}%")

            if "apuesta_cantidad" in op:
                print(f"   Apuesta sugerida: {op['apuesta_cantidad']:.2f}€")

        print("\n" + "=" * 60)

    def _crear_mensaje_email(self, oportunidades):
        """
        Crea HTML para email

        Args:
            oportunidades: Lista de oportunidades

        Returns:
            str: HTML del email
        """
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .container { max-width: 600px; margin: 0 auto; }
                .header { background-color: #2ecc71; color: white; padding: 20px; text-align: center; }
                .opportunity { border: 2px solid #3498db; padding: 15px; margin: 15px 0; border-radius: 5px; }
                .high-ev { border-color: #e74c3c; background-color: #ffe6e6; }
                .label { font-weight: bold; color: #2c3e50; }
                .value { color: #34495e; }
                .ev-positive { color: #27ae60; font-weight: bold; }
                .footer { text-align: center; padding: 20px; color: #7f8c8d; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>🎾 Oportunidades de Apuesta Detectadas</h2>
                    <p>Tennis ML Predictor</p>
                </div>
        """

        for i, op in enumerate(oportunidades, 1):
            # Determinar si es alta prioridad (EV > 10%)
            high_priority = op.get("ev", 0) > 0.10
            class_name = "opportunity high-ev" if high_priority else "opportunity"

            html += f"""
                <div class="{class_name}">
                    <h3>🏆 Oportunidad #{i}</h3>
                    <p><span class="label">Partido:</span> <span class="value">{op.get('jugador', 'N/A')} vs {op.get('oponente', 'N/A')}</span></p>
                    <p><span class="label">Apostar a:</span> <span class="value">{op.get('jugador', 'N/A')}</span></p>
                    <p><span class="label">Bookmaker:</span> <span class="value">{op.get('bookmaker', 'N/A')}</span></p>
                    <p><span class="label">Cuota:</span> <span class="value">@{op.get('cuota', 0):.2f}</span></p>
                    <p><span class="label">Probabilidad:</span> <span class="value">{op.get('prob', 0)*100:.1f}%</span></p>
                    <p><span class="label">Expected Value:</span> <span class="ev-positive">+{op.get('ev', 0)*100:.2f}%</span></p>
            """

            if "saving_vs_promedio" in op:
                html += f"""
                    <p><span class="label">Saving vs promedio:</span> <span class="value">+{op['saving_vs_promedio']*100:.2f}%</span></p>
                """

            if "apuesta_cantidad" in op:
                html += f"""
                    <p><span class="label">Apuesta sugerida:</span> <span class="value">{op['apuesta_cantidad']:.2f}€</span></p>
                """

            html += "</div>"

        # Footer
        html += f"""
                <div class="footer">
                    <p>Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Tennis ML Predictor - Phase 6</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def enviar_alerta_simple(self, mensaje, asunto="Tennis ML - Alerta"):
        """
        Envía una alerta simple por email

        Args:
            mensaje: Mensaje a enviar
            asunto: Asunto del email
        """
        if not self.email_enabled:
            print(f"\n📧 {asunto}")
            print(f"   {mensaje}")
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = BookmakerConfig.EMAIL_ADDRESS
            msg["To"] = BookmakerConfig.EMAIL_RECIPIENT
            msg["Subject"] = asunto

            msg.attach(MIMEText(mensaje, "plain"))

            server = smtplib.SMTP(BookmakerConfig.SMTP_SERVER, BookmakerConfig.SMTP_PORT)
            server.starttls()
            server.login(BookmakerConfig.EMAIL_ADDRESS, BookmakerConfig.EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()

            print(f"✅ Alerta enviada: {asunto}")

        except Exception as e:
            print(f"❌ Error enviando email: {e}")


# Ejemplo de uso
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚨 ALERT SYSTEM - DEMO")
    print("=" * 60)

    # Crear sistema de alertas
    alert = AlertSystem(email_enabled=False)  # Solo consola para demo

    # Oportunidades de ejemplo
    oportunidades_ejemplo = [
        {
            "jugador": "Carlos Alcaraz",
            "oponente": "Jannik Sinner",
            "cuota": 2.10,
            "bookmaker": "Pinnacle",
            "prob": 0.58,
            "ev": 0.078,
            "saving_vs_promedio": 0.018,
            "apuesta_cantidad": 45.50,
        },
        {
            "jugador": "Novak Djokovic",
            "oponente": "Daniil Medvedev",
            "cuota": 1.95,
            "bookmaker": "Bet365",
            "prob": 0.62,
            "ev": 0.109,
            "saving_vs_promedio": 0.025,
            "apuesta_cantidad": 62.30,
        },
    ]

    # Verificar oportunidades
    alert.verificar_oportunidades(oportunidades_ejemplo, umbral_ev=0.05)

    print(f"\n✅ Demo completado!")
