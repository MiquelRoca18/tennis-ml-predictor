"""
Procesador de Cuotas de Tennis-Data.co.uk
==========================================

Este script procesa archivos Excel/CSV de Tennis-Data.co.uk
y los convierte al formato necesario para el backtesting.

Fuente: http://www.tennis-data.co.uk/
Archivos: 2022.xlsx, 2023.xlsx, 2024.xlsx

INSTRUCCIONES:
1. Visita http://www.tennis-data.co.uk/
2. Descarga los archivos Excel de los años que quieras (2022-2024)
3. Guárdalos en: datos/odds_historicas/
4. Ejecuta este script: python scripts/internal/process_tennis_data_odds.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TennisDataProcessor:
    """
    Procesa archivos de Tennis-Data.co.uk
    """
    
    def __init__(self, input_dir="datos/odds_historicas", output_dir="datos/odds_historicas"):
        """
        Inicializa el procesador
        
        Args:
            input_dir: Directorio con archivos Excel descargados
            output_dir: Directorio para guardar CSV procesado
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ TennisDataProcessor inicializado")
        logger.info(f"📁 Input: {self.input_dir}")
        logger.info(f"📁 Output: {self.output_dir}")
    
    def load_excel_file(self, filepath):
        """
        Carga un archivo Excel de Tennis-Data.co.uk
        
        Args:
            filepath: Path al archivo Excel
            
        Returns:
            DataFrame con datos cargados
        """
        try:
            logger.info(f"📥 Cargando {filepath.name}...")
            
            # Intentar leer como Excel
            try:
                df = pd.read_excel(filepath)
            except:
                # Si falla, intentar como CSV
                df = pd.read_csv(filepath)
            
            logger.info(f"  ✅ {len(df)} partidos cargados")
            return df
            
        except Exception as e:
            logger.error(f"  ❌ Error al cargar {filepath.name}: {e}")
            return None
    
    def process_tennis_data(self, df):
        """
        Procesa DataFrame de Tennis-Data.co.uk al formato necesario
        
        Args:
            df: DataFrame con datos de Tennis-Data.co.uk
            
        Returns:
            DataFrame procesado
        """
        logger.info("🔄 Procesando datos...")
        
        # Crear DataFrame de salida
        records = []
        
        for idx, row in df.iterrows():
            try:
                # Información básica
                fecha = pd.to_datetime(row.get('Date'))
                ganador = row.get('Winner')
                perdedor = row.get('Loser')
                superficie = row.get('Court', 'Unknown')
                
                # Cuotas - Preferencia: Pinnacle > Bet365 > Exchange
                # Pinnacle es considerado el bookmaker más sharp
                cuota_ganador = None
                cuota_perdedor = None
                bookmaker_usado = None
                
                # Intentar Pinnacle primero
                if pd.notna(row.get('PSW')) and pd.notna(row.get('PSL')):
                    cuota_ganador = float(row['PSW'])
                    cuota_perdedor = float(row['PSL'])
                    bookmaker_usado = 'Pinnacle'
                
                # Si no hay Pinnacle, intentar Bet365
                elif pd.notna(row.get('B365W')) and pd.notna(row.get('B365L')):
                    cuota_ganador = float(row['B365W'])
                    cuota_perdedor = float(row['B365L'])
                    bookmaker_usado = 'Bet365'
                
                # Si no hay Bet365, intentar Exchange
                elif pd.notna(row.get('EXW')) and pd.notna(row.get('EXL')):
                    cuota_ganador = float(row['EXW'])
                    cuota_perdedor = float(row['EXL'])
                    bookmaker_usado = 'Exchange'
                
                # Si no hay cuotas, saltar
                if cuota_ganador is None or cuota_perdedor is None:
                    continue
                
                # Validar cuotas (deben ser >= 1.01)
                if cuota_ganador < 1.01 or cuota_perdedor < 1.01:
                    continue
                
                # Crear registro
                record = {
                    'fecha': fecha,
                    'jugador_1': ganador,
                    'jugador_2': perdedor,
                    'cuota_jugador_1': cuota_ganador,
                    'cuota_jugador_2': cuota_perdedor,
                    'superficie': superficie,
                    'bookmaker': bookmaker_usado,
                    'torneo': row.get('Tournament', 'Unknown'),
                    'serie': row.get('Series', 'Unknown'),
                    'ronda': row.get('Round', 'Unknown'),
                    'ganador_rank': row.get('WRank', np.nan),
                    'perdedor_rank': row.get('LRank', np.nan)
                }
                
                records.append(record)
                
            except Exception as e:
                logger.warning(f"  ⚠️  Error procesando fila {idx}: {e}")
                continue
        
        if records:
            df_processed = pd.DataFrame(records)
            df_processed = df_processed.sort_values('fecha').reset_index(drop=True)
            
            logger.info(f"  ✅ {len(df_processed)} partidos procesados con cuotas")
            
            # Estadísticas
            logger.info(f"\n📊 ESTADÍSTICAS:")
            logger.info(f"  Periodo: {df_processed['fecha'].min().date()} a {df_processed['fecha'].max().date()}")
            logger.info(f"  Cuota promedio ganador: {df_processed['cuota_jugador_1'].mean():.2f}")
            logger.info(f"  Cuota promedio perdedor: {df_processed['cuota_jugador_2'].mean():.2f}")
            
            # Distribución por bookmaker
            bookmaker_counts = df_processed['bookmaker'].value_counts()
            logger.info(f"\n  Distribución por bookmaker:")
            for bm, count in bookmaker_counts.items():
                logger.info(f"    {bm}: {count} partidos ({count/len(df_processed)*100:.1f}%)")
            
            # Distribución por superficie
            superficie_counts = df_processed['superficie'].value_counts()
            logger.info(f"\n  Distribución por superficie:")
            for sup, count in superficie_counts.items():
                logger.info(f"    {sup}: {count} partidos ({count/len(df_processed)*100:.1f}%)")
            
            return df_processed
        else:
            logger.warning("  ⚠️  No se pudieron procesar partidos")
            return pd.DataFrame()
    
    def process_multiple_years(self, years=[2022, 2023, 2024]):
        """
        Procesa archivos de múltiples años
        
        Args:
            years: Lista de años a procesar
            
        Returns:
            DataFrame combinado con todos los años
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🎾 PROCESANDO DATOS DE TENNIS-DATA.CO.UK")
        logger.info(f"{'='*70}")
        logger.info(f"Años a procesar: {years}")
        
        all_dfs = []
        
        for year in years:
            # Buscar archivo para este año
            excel_file = self.input_dir / f"{year}.xlsx"
            csv_file = self.input_dir / f"{year}.csv"
            
            filepath = None
            if excel_file.exists():
                filepath = excel_file
            elif csv_file.exists():
                filepath = csv_file
            else:
                logger.warning(f"  ⚠️  No se encontró archivo para {year}")
                logger.warning(f"     Esperado: {excel_file} o {csv_file}")
                continue
            
            # Cargar y procesar
            df = self.load_excel_file(filepath)
            if df is not None:
                df_processed = self.process_tennis_data(df)
                if not df_processed.empty:
                    all_dfs.append(df_processed)
        
        # Combinar todos los años
        if all_dfs:
            df_final = pd.concat(all_dfs, ignore_index=True)
            df_final = df_final.sort_values('fecha').reset_index(drop=True)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ PROCESAMIENTO COMPLETADO")
            logger.info(f"{'='*70}")
            logger.info(f"Total partidos: {len(df_final)}")
            logger.info(f"Periodo: {df_final['fecha'].min().date()} a {df_final['fecha'].max().date()}")
            logger.info(f"Años incluidos: {sorted(df_final['fecha'].dt.year.unique())}")
            
            return df_final
        else:
            logger.error("\n❌ No se pudieron procesar archivos")
            return pd.DataFrame()
    
    def save_processed_data(self, df, filename="tennis_odds_processed.csv"):
        """
        Guarda datos procesados
        
        Args:
            df: DataFrame con datos procesados
            filename: Nombre del archivo de salida
        """
        if df.empty:
            logger.error("❌ No hay datos para guardar")
            return None
        
        output_file = self.output_dir / filename
        df.to_csv(output_file, index=False)
        
        logger.info(f"\n💾 Datos guardados en: {output_file}")
        logger.info(f"   Total partidos: {len(df)}")
        
        return output_file


def main():
    """
    Función principal
    """
    logger.info("\n" + "="*70)
    logger.info("🎾 PROCESADOR DE CUOTAS DE TENNIS-DATA.CO.UK")
    logger.info("="*70)
    
    # Verificar si existen archivos
    input_dir = Path("datos/odds_historicas")
    
    if not input_dir.exists():
        logger.error(f"\n❌ Directorio no encontrado: {input_dir}")
        logger.info("\nPor favor:")
        logger.info("  1. Crea el directorio: mkdir -p datos/odds_historicas")
        logger.info("  2. Descarga archivos de http://www.tennis-data.co.uk/")
        logger.info("  3. Guarda los archivos Excel en datos/odds_historicas/")
        return
    
    # Buscar archivos disponibles
    excel_files = list(input_dir.glob("*.xlsx"))
    csv_files = list(input_dir.glob("*.csv"))
    all_files = excel_files + csv_files
    
    if not all_files:
        logger.error(f"\n❌ No se encontraron archivos en {input_dir}")
        logger.info("\n📥 INSTRUCCIONES DE DESCARGA:")
        logger.info("  1. Visita: http://www.tennis-data.co.uk/")
        logger.info("  2. Descarga los archivos de los años que quieras:")
        logger.info("     - 2024.xlsx")
        logger.info("     - 2023.xlsx")
        logger.info("     - 2022.xlsx")
        logger.info(f"  3. Guárdalos en: {input_dir.absolute()}")
        logger.info("  4. Ejecuta este script de nuevo")
        return
    
    logger.info(f"\n📁 Archivos encontrados:")
    for f in all_files:
        logger.info(f"  - {f.name}")
    
    # Crear procesador
    processor = TennisDataProcessor()
    
    # Detectar años disponibles
    years = []
    for f in all_files:
        try:
            year = int(f.stem)
            if 2000 <= year <= 2030:
                years.append(year)
        except:
            pass
    
    years = sorted(set(years))
    
    if not years:
        logger.error("\n❌ No se detectaron años válidos en los archivos")
        return
    
    logger.info(f"\n📅 Años detectados: {years}")
    
    # Procesar datos
    df_processed = processor.process_multiple_years(years)
    
    if not df_processed.empty:
        # Guardar
        output_file = processor.save_processed_data(
            df_processed,
            filename=f"tennis_odds_{min(years)}_{max(years)}.csv"
        )
        
        logger.info(f"\n✅ Proceso completado exitosamente")
        logger.info(f"\n🚀 PRÓXIMO PASO:")
        logger.info(f"   Ejecuta el backtesting con cuotas reales:")
        logger.info(f"   python scripts/backtesting_real_odds.py")
    else:
        logger.error("\n❌ No se pudieron procesar los datos")


if __name__ == "__main__":
    main()
