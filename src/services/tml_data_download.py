"""
Descarga de CSVs desde TML-Database (GitHub) para ELO y predicciones.
Usado por POST /admin/refresh-elo-data y por el webhook de GitHub.
"""

import urllib.request
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import date

logger = logging.getLogger(__name__)

TML_BASE_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
DATA_RAW = Path("datos/raw")


def download_tml_csvs(years: List[int]) -> Tuple[List[str], List[str]]:
    """
    Descarga los CSV de TML-Database para los años indicados y los guarda en datos/raw/.

    Args:
        years: Lista de años (ej: [2025, 2026]). Se descargan como {año}.csv.

    Returns:
        (downloaded, errors): listas de años descargados correctamente y de errores.
    """
    current_year = date.today().year
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    downloaded: List[str] = []
    errors: List[str] = []

    for yr in years:
        if yr < 2018 or yr > current_year + 1:
            continue
        url = f"{TML_BASE_URL}/{yr}.csv"
        dest = DATA_RAW / f"{yr}.csv"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TennisML-Predictor/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            downloaded.append(str(yr))
            logger.info(f"✅ TML-Database CSV descargado: {yr}.csv")
        except Exception as e:
            errors.append(f"{yr}: {e}")
            logger.warning(f"⚠️ No se pudo descargar {yr}.csv: {e}")

    return downloaded, errors


def extract_years_from_csv_filenames(filenames: List[str]) -> List[int]:
    """
    Extrae años de nombres de archivo como '2025.csv', '2026.csv'.

    Args:
        filenames: Lista de nombres (ej: ["2025.csv", "2026.csv"] del webhook).

    Returns:
        Lista de años válidos (2018..año_actual+1).
    """
    current_year = date.today().year
    years: List[int] = []
    for name in filenames:
        if not name.endswith(".csv"):
            continue
        base = name[:-4]
        if base.isdigit():
            yr = int(base)
            if 2018 <= yr <= current_year + 1:
                years.append(yr)
    return sorted(set(years))


def remove_old_year_csvs() -> List[str]:
    """
    Elimina de datos/raw/ los CSV de años que ya no se usan en la temporada actual.

    En producción solo usamos (año_actual - 1) y año_actual para ELO.
    Ejemplo: en temporada 2027 se usan 2026 y 2027; se elimina 2025.csv (y cualquier año anterior).

    Returns:
        Lista de nombres de archivos eliminados (ej: ["2025.csv"]).
    """
    current_year = date.today().year
    min_year_to_keep = current_year - 1  # Por debajo de este año se elimina
    removed: List[str] = []
    if not DATA_RAW.exists():
        return removed
    for path in DATA_RAW.iterdir():
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        stem = path.stem
        if not stem.isdigit():
            continue
        yr = int(stem)
        if yr < min_year_to_keep:
            try:
                path.unlink()
                removed.append(path.name)
                logger.info(f"🗑️ Eliminado CSV de temporada antigua: {path.name} (año actual {current_year})")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo eliminar {path.name}: {e}")
    return removed


def refresh_elo_data_daily() -> dict:
    """
    Rutina diaria: descarga CSVs de la temporada actual (año anterior + año actual),
    elimina CSVs de años ya no usados, y devuelve estado para que el caller resetee el predictor.

    Returns:
        dict con downloaded, removed, errors (el caller debe llamar a reset_fgs y reset_predictor).
    """
    current_year = date.today().year
    years_to_download = [current_year - 1, current_year]
    removed = remove_old_year_csvs()
    downloaded, errors = download_tml_csvs(years_to_download)
    return {
        "downloaded": downloaded,
        "removed": removed,
        "errors": errors,
        "years_downloaded": years_to_download,
    }
