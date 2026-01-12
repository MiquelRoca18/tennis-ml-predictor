"""
DEPRECATED: Este archivo se mantiene solo para compatibilidad hacia atrás.

La funcionalidad de Kelly Criterion ahora está integrada directamente en TrackingSystem.

USO NUEVO (recomendado):
    from src.tracking.tracking_system import TrackingSystem

    sistema = TrackingSystem(
        modelo_path="modelos/random_forest_calibrado.pkl",
        db_path="apuestas_tracker.db",
        bankroll_actual=1000,
        usar_kelly=True,
        kelly_fraccion=0.25
    )

USO ANTIGUO (todavía funciona):
    from src.tracking.tracking_system_kelly import TrackingSystemKelly

    sistema = TrackingSystemKelly(
        modelo_path="modelos/random_forest_calibrado.pkl",
        db_path="apuestas_tracker.db",
        bankroll_actual=1000,
        usar_kelly=True,
        kelly_fraccion=0.25
    )
"""

import warnings
from src.tracking.tracking_system import TrackingSystem


class TrackingSystemKelly(TrackingSystem):
    """
    DEPRECATED: Usar TrackingSystem con usar_kelly=True en su lugar.

    Esta clase se mantiene solo para compatibilidad hacia atrás.
    """

    def __init__(
        self,
        modelo_path,
        db_path="apuestas_tracker.db",
        bankroll_actual=1000,
        usar_kelly=True,
        kelly_fraccion=0.25,
    ):
        """
        DEPRECATED: Usar TrackingSystem directamente.

        Args:
            modelo_path: Path al modelo calibrado
            db_path: Path a la base de datos
            bankroll_actual: Capital actual disponible
            usar_kelly: Si True, usa Kelly; si False, usa flat betting
            kelly_fraccion: Fracción de Kelly a usar
        """
        warnings.warn(
            "TrackingSystemKelly está deprecated. "
            "Usar TrackingSystem con usar_kelly=True en su lugar.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Simplemente llamar al constructor de TrackingSystem con Kelly activado
        super().__init__(
            modelo_path=modelo_path,
            db_path=db_path,
            bankroll_actual=bankroll_actual,
            usar_kelly=usar_kelly,
            kelly_fraccion=kelly_fraccion,
        )

    def generar_reporte_kelly(self):
        """
        DEPRECATED: Usar generar_reporte() en su lugar.

        El método generar_reporte() ahora incluye automáticamente
        las métricas de Kelly cuando usar_kelly=True.
        """
        warnings.warn(
            "generar_reporte_kelly() está deprecated. " "Usar generar_reporte() en su lugar.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.generar_reporte()


# Ejemplo de uso (deprecated)
if __name__ == "__main__":
    print("\n⚠️  ADVERTENCIA: Este archivo está deprecated")
    print("Usar TrackingSystem con usar_kelly=True en su lugar\n")

    # Esto todavía funciona pero mostrará un warning
    sistema = TrackingSystemKelly(
        modelo_path="modelos/random_forest_calibrado.pkl",
        db_path="apuestas_tracker_kelly.db",
        bankroll_actual=1000,
        usar_kelly=True,
        kelly_fraccion=0.25,
    )

    print("\n✅ Sistema inicializado (usando wrapper deprecated)")
    sistema.generar_reporte()
