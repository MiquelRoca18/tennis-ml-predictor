"""
Utilidades Compartidas
=====================

Funciones comunes usadas en todo el proyecto.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Any, Callable
import sys
import joblib  # Añadido para cargar modelos correctamente


def load_model(model_path: str) -> Any:
    """
    Carga modelo de forma segura usando joblib
    
    Args:
        model_path: Ruta al archivo del modelo
        
    Returns:
        Modelo cargado
        
    Raises:
        FileNotFoundError: Si el modelo no existe
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    # Usar joblib para cargar (compatible con cómo se guardan los modelos)
    return joblib.load(model_path)


def load_data(data_path: str, features: Optional[list] = None) -> Tuple:
    """
    Carga datos de forma segura
    
    Args:
        data_path: Ruta al archivo CSV
        features: Lista de features a extraer (opcional)
        
    Returns:
        Si features especificado: (X, y, df)
        Si no: df
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Datos no encontrados: {data_path}")
    
    df = pd.read_csv(data_path)
    
    if features:
        X = df[features]
        # Intentar encontrar la columna target
        y = df.get('target', df.get('winner', df.get('resultado')))
        return X, y, df
    
    return df


def print_header(title: str, emoji: str = "📊", width: int = 60):
    """
    Imprime header consistente
    
    Args:
        title: Título a mostrar
        emoji: Emoji opcional
        width: Ancho del header
    """
    print("\n" + "=" * width)
    print(f"{emoji} {title}")
    print("=" * width)


def print_section(title: str, width: int = 60):
    """
    Imprime sección con líneas
    
    Args:
        title: Título de la sección
        width: Ancho de la sección
    """
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def print_metric(name: str, value: Any, unit: str = "", decimals: int = 2):
    """
    Imprime métrica formateada
    
    Args:
        name: Nombre de la métrica
        value: Valor de la métrica
        unit: Unidad (opcional)
        decimals: Decimales a mostrar
    """
    if isinstance(value, (int, float)):
        if decimals == 0:
            print(f"   {name}: {value:.0f}{unit}")
        else:
            print(f"   {name}: {value:.{decimals}f}{unit}")
    else:
        print(f"   {name}: {value}{unit}")


def safe_execute(func: Callable, error_msg: str = "Error", verbose: bool = True) -> bool:
    """
    Ejecuta función con manejo de errores
    
    Args:
        func: Función a ejecutar
        error_msg: Mensaje de error personalizado
        verbose: Si True, imprime traceback
        
    Returns:
        True si éxito, False si error
    """
    try:
        func()
        return True
    except Exception as e:
        print(f"❌ {error_msg}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Formatea valor como porcentaje
    
    Args:
        value: Valor entre 0 y 1
        decimals: Decimales a mostrar
        
    Returns:
        String formateado (ej: "75.5%")
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, symbol: str = "€", decimals: int = 2) -> str:
    """
    Formatea valor como moneda
    
    Args:
        value: Valor numérico
        symbol: Símbolo de moneda
        decimals: Decimales a mostrar
        
    Returns:
        String formateado (ej: "1,234.56€")
    """
    return f"{value:,.{decimals}f}{symbol}"


def create_directory(path: str, verbose: bool = False) -> Path:
    """
    Crea directorio si no existe
    
    Args:
        path: Ruta del directorio
        verbose: Si True, imprime mensaje
        
    Returns:
        Path object del directorio
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"✅ Directorio creado/verificado: {dir_path}")
    
    return dir_path


def get_project_root() -> Path:
    """
    Obtiene la raíz del proyecto
    
    Returns:
        Path a la raíz del proyecto
    """
    # Asume que este archivo está en src/utils/
    return Path(__file__).parent.parent.parent


def add_project_to_path():
    """Añade la raíz del proyecto al sys.path"""
    project_root = str(get_project_root())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def print_summary(results: dict, title: str = "RESUMEN"):
    """
    Imprime resumen de resultados
    
    Args:
        results: Dict con {nombre: bool_exito}
        title: Título del resumen
    """
    print_header(title, "📋")
    
    for nombre, exito in results.items():
        estado = "✅ PASÓ" if exito else "❌ FALLÓ"
        print(f"{estado} - {nombre}")
    
    total_exitosas = sum(results.values())
    total = len(results)
    
    print(f"\n📊 Total: {total_exitosas}/{total} exitosas")
    
    if total_exitosas == total:
        print("\n🎉 ¡Todo completado exitosamente!")
    else:
        print(f"\n⚠️  {total - total_exitosas} fallaron")
    
    return total_exitosas == total


# Constantes útiles
EMOJI_SUCCESS = "✅"
EMOJI_ERROR = "❌"
EMOJI_WARNING = "⚠️"
EMOJI_INFO = "ℹ️"
EMOJI_ROCKET = "🚀"
EMOJI_CHART = "📊"
EMOJI_MONEY = "💰"
EMOJI_TROPHY = "🏆"
