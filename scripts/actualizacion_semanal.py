"""
Actualización Semanal Automática del Modelo
============================================

Este script:
1. Descarga datos actualizados de TML GitHub
2. Mantiene ventana temporal fija (elimina semana más antigua)
3. Re-genera dataset con nuevo formato
4. Re-entrena modelo con datos actualizados
5. Guarda modelo actualizado

Uso:
    python scripts/actualizacion_semanal.py
    
Programar en cron (cada domingo a las 2 AM):
    0 2 * * 0 cd /path/to/tennis-ml-predictor && python scripts/actualizacion_semanal.py
"""

import pandas as pd
import requests
from pathlib import Path
import logging
from datetime import datetime, timedelta
import sys
import subprocess

# Añadir path para imports
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.features.feature_engineer_completo import CompleteFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/actualizacion_semanal.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ActualizadorSemanal:
    """
    Gestiona la actualización semanal automática del modelo
    """
    
    def __init__(self, ventana_años=4):
        """
        Args:
            ventana_años: Años de datos a mantener (default: 4 años)
        """
        self.ventana_años = ventana_años
        self.datos_dir = Path("datos/raw")
        self.processed_dir = Path("datos/processed")
        self.modelos_dir = Path("modelos")
        
        # URLs de TML GitHub
        self.tml_base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
        
        logger.info("🔄 Actualizador Semanal inicializado")
        logger.info(f"   Ventana temporal: {ventana_años} años")
    
    def descargar_datos_tml(self):
        """
        Descarga datos actualizados de TML GitHub
        """
        logger.info("\n📥 Descargando datos de TML GitHub...")
        
        año_actual = datetime.now().year
        años_a_descargar = range(año_actual - self.ventana_años, año_actual + 1)
        
        archivos_descargados = []
        
        for año in años_a_descargar:
            filename = f"atp_matches_{año}.csv"
            url = f"{self.tml_base_url}/{filename}"
            output_path = self.datos_dir / f"atp_matches_{año}_tml.csv"
            
            try:
                logger.info(f"   Descargando {filename}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Guardar archivo
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(response.content)
                
                archivos_descargados.append(output_path)
                logger.info(f"   ✅ {filename} descargado")
                
            except Exception as e:
                logger.error(f"   ❌ Error descargando {filename}: {e}")
        
        logger.info(f"\n✅ Descargados {len(archivos_descargados)} archivos")
        return archivos_descargados
    
    def combinar_datos(self):
        """
        Combina datos de múltiples años y aplica ventana temporal
        """
        logger.info("\n🔗 Combinando datos...")
        
        año_actual = datetime.now().year
        fecha_limite = datetime.now() - timedelta(days=self.ventana_años * 365)
        
        # Cargar todos los archivos
        dfs = []
        for año in range(año_actual - self.ventana_años, año_actual + 1):
            filepath = self.datos_dir / f"atp_matches_{año}_tml.csv"
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
                dfs.append(df)
                logger.info(f"   ✅ Cargado {año}: {len(df)} partidos")
        
        # Combinar
        df_combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"\n📊 Total partidos antes de filtrar: {len(df_combined):,}")
        
        # Aplicar ventana temporal (mantener últimos N años)
        df_filtered = df_combined[df_combined['tourney_date'] >= fecha_limite].copy()
        logger.info(f"📊 Partidos después de filtrar (últimos {self.ventana_años} años): {len(df_filtered):,}")
        
        # Guardar dataset combinado
        output_path = self.processed_dir / "atp_matches_clean.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(output_path, index=False)
        
        logger.info(f"💾 Dataset combinado guardado: {output_path}")
        logger.info(f"   Rango: {df_filtered['tourney_date'].min()} - {df_filtered['tourney_date'].max()}")
        
        return df_filtered
    
    def regenerar_dataset(self):
        """
        Re-genera dataset con features completas
        """
        logger.info("\n🔧 Re-generando dataset con features...")
        
        # Cargar datos limpios
        df = pd.read_csv(self.processed_dir / "atp_matches_clean.csv")
        df['tourney_date'] = pd.to_datetime(df['tourney_date'])
        
        logger.info(f"   Partidos a procesar: {len(df):,}")
        
        # Crear feature engineer
        engineer = CompleteFeatureEngineer(df)
        
        # Procesar dataset completo
        df_features = engineer.procesar_dataset_completo(
            save_path=str(self.processed_dir / "dataset_features_fase3_completas.csv")
        )
        
        logger.info(f"✅ Dataset regenerado: {len(df_features):,} partidos, {len(df_features.columns)-2} features")
        
        return df_features
    
    def reentrenar_modelo(self):
        """
        Re-entrena modelo con datos actualizados
        """
        logger.info("\n🎓 Re-entrenando modelo...")
        
        # Ejecutar pipeline de optimización
        script_path = Path("scripts/internal/run_fase3_optimization.py")
        
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hora máximo
            )
            
            if result.returncode == 0:
                logger.info("✅ Modelo re-entrenado exitosamente")
                logger.info(f"\n{result.stdout[-500:]}")  # Últimas 500 líneas
            else:
                logger.error(f"❌ Error re-entrenando modelo: {result.stderr}")
                raise Exception("Error en re-entrenamiento")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout re-entrenando modelo (>1 hora)")
            raise
    
    def ejecutar_actualizacion_completa(self):
        """
        Ejecuta el proceso completo de actualización semanal
        """
        logger.info("="*70)
        logger.info("🚀 ACTUALIZACIÓN SEMANAL AUTOMÁTICA")
        logger.info("="*70)
        logger.info(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        try:
            # 1. Descargar datos actualizados
            self.descargar_datos_tml()
            
            # 2. Combinar y filtrar datos (ventana temporal)
            self.combinar_datos()
            
            # 3. Re-generar dataset con features
            self.regenerar_dataset()
            
            # 4. Re-entrenar modelo
            self.reentrenar_modelo()
            
            logger.info("\n" + "="*70)
            logger.info("✅ ACTUALIZACIÓN COMPLETADA EXITOSAMENTE")
            logger.info("="*70)
            logger.info(f"Modelo actualizado: {self.modelos_dir / 'random_forest_calibrado.pkl'}")
            logger.info(f"Próxima actualización: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ ERROR EN ACTUALIZACIÓN: {e}")
            logger.exception("Detalles del error:")
            return False


def main():
    """
    Función principal
    """
    # Crear directorio de logs
    Path("logs").mkdir(exist_ok=True)
    
    # Ejecutar actualización
    actualizador = ActualizadorSemanal(ventana_años=4)
    exito = actualizador.ejecutar_actualizacion_completa()
    
    # Exit code
    sys.exit(0 if exito else 1)


if __name__ == "__main__":
    main()
