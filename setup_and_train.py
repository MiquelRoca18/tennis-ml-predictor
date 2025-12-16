#!/usr/bin/env python3
"""
Pipeline Maestro para Tennis ML Predictor
Versión FIXED - Corrección en verify_dependencies
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import argparse


class TennisPipeline:
    def __init__(self):
        self.start_time = time.time()
        self.steps_completed = []
        self.steps_failed = []
    
    def print_header(self, text):
        """Imprime encabezado formateado"""
        print("\n" + "=" * 80)
        print(f"🎾 {text}")
        print("=" * 80 + "\n")
    
    def print_step(self, step_num, total_steps, description):
        """Imprime paso actual"""
        print("\n" + "=" * 60)
        print(f"📍 PASO {step_num}/{total_steps}: {description}")
        print("=" * 60 + "\n")
    
    def run_command(self, command, description, required=True):
        """Ejecuta un comando del sistema"""
        print(f"🔄 Ejecutando: {command}")
        print(f"   Descripción: {description}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=False,
                text=True
            )
            
            print(f"✅ {description} - Completado\n")
            self.steps_completed.append(description)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error en: {description}")
            self.steps_failed.append(description)
            
            if required:
                print(f"Error crítico ejecutando: {command}")
                print(f"Código de salida: {e.returncode}")
                return False
            else:
                print(f"⚠️  Error no crítico, continuando...")
                return True
    
    def verify_dependencies(self):
        """Verifica que las dependencias estén instaladas - VERSIÓN CORREGIDA"""
        self.print_step(1, 10, "Verificando Dependencias")
        
        print("📦 Verificando paquetes de Python...")
        
        # Mapeo correcto: nombre_display -> nombre_importación
        required_packages = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scikit-learn': 'sklearn',  # ← FIX: sklearn es el nombre de importación
            'xgboost': 'xgboost',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'joblib': 'joblib',
            'requests': 'requests'
        }
        
        missing = []
        for display_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"  ✅ {display_name}")
            except ImportError:
                print(f"  ❌ {display_name} - NO INSTALADO")
                missing.append(display_name)
        
        if missing:
            print(f"\n⚠️  Faltan paquetes: {', '.join(missing)}")
            print("Ejecuta: pip install -r requirements.txt")
            return False
        
        print("\n✅ Todas las dependencias están instaladas")
        self.steps_completed.append("Verificar dependencias")
        return True
    
    def create_directories(self):
        """Crea estructura de carpetas"""
        self.print_step(2, 10, "Creando Estructura de Carpetas")
        
        directories = [
            'datos/raw',
            'datos/processed',
            'datos/tml_database',
            'modelos',
            'resultados/calibracion',
            'resultados/backtesting',
            'resultados/walk_forward',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {directory}")
        
        print("\n✅ Estructura de carpetas creada")
        self.steps_completed.append("Crear carpetas")
        return True
    
    def download_data(self):
        """Descarga datos de TML Database"""
        self.print_step(3, 10, "Descargando Datos")
        
        return self.run_command(
            "python src/data/tml_data_downloader.py",
            "Descargar datos de TML Database",
            required=True
        )
    
    def process_data(self):
        """Procesa datos raw"""
        self.print_step(4, 10, "Procesando Datos")
        
        return self.run_command(
            "python src/data/data_processor.py",
            "Procesar datos raw",
            required=True
        )
    
    def feature_engineering(self):
        """Genera features completas"""
        self.print_step(5, 10, "Feature Engineering (149 features)")
        
        print("⏱️  Este paso puede tardar 3-5 minutos...")
        
        return self.run_command(
            "python run_feature_engineering_fase3.py",
            "Generar 149 features avanzadas",
            required=True
        )
    
    def train_and_optimize(self):
        """Entrena modelos y optimiza hiperparámetros"""
        self.print_step(6, 10, "Entrenamiento y Optimización")
        
        print("⏱️  Este paso puede tardar 10-15 minutos...")
        
        return self.run_command(
            "python run_fase3_optimization.py",
            "Entrenar modelos y optimizar hiperparámetros",
            required=True
        )
    
    def validate_calibration(self):
        """Valida calibración del modelo"""
        self.print_step(7, 10, "Validación de Calibración")
        
        return self.run_command(
            "python validacion_calibracion.py",
            "Validar calibración del modelo",
            required=True
        )
    
    def run_backtesting(self):
        """Ejecuta backtesting"""
        self.print_step(8, 10, "Backtesting")
        
        print("⏱️  Este paso puede tardar 5 minutos...")
        
        return self.run_command(
            "python backtesting_fase2.py",
            "Ejecutar backtesting con datos históricos",
            required=True
        )
    
    def walk_forward_validation(self):
        """Ejecuta Walk-Forward Validation"""
        self.print_step(9, 10, "Walk-Forward Validation")
        
        print("⏱️  Este paso puede tardar 10 minutos...")
        
        return self.run_command(
            "python validacion_final_fase3.py",
            "Validación temporal con folds",
            required=True
        )
    
    def generate_reports(self):
        """Genera reportes finales"""
        self.print_step(10, 10, "Generando Reportes")
        
        return self.run_command(
            "python generar_reporte_fase2.py",
            "Generar reporte HTML interactivo",
            required=False
        )
    
    def run_full_pipeline(self):
        """Ejecuta el pipeline completo"""
        self.print_header("PIPELINE COMPLETO - INICIO")
        
        print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("⏱️  Tiempo estimado: 30-40 minutos")
        print("\n📋 Pasos a ejecutar:")
        print("  1. Verificar dependencias")
        print("  2. Crear estructura de carpetas")
        print("  3. Descargar datos (TML Database)")
        print("  4. Procesar datos")
        print("  5. Feature engineering (149 features)")
        print("  6. Entrenar y optimizar modelos")
        print("  7. Validar calibración")
        print("  8. Ejecutar backtesting")
        print("  9. Walk-Forward Validation")
        print("  10. Generar reportes")
        
        input("\n👉 Presiona ENTER para comenzar...")
        
        # Ejecutar pasos
        steps = [
            self.verify_dependencies,
            self.create_directories,
            self.download_data,
            self.process_data,
            self.feature_engineering,
            self.train_and_optimize,
            self.validate_calibration,
            self.run_backtesting,
            self.walk_forward_validation,
            self.generate_reports
        ]
        
        for step in steps:
            if not step():
                print("\n❌ Pipeline interrumpido por error crítico")
                self.print_summary()
                return False
        
        self.print_summary()
        return True
    
    def run_train_only(self):
        """Solo entrena (asume que ya hay datos)"""
        self.print_header("ENTRENAMIENTO - Solo Entrenar Modelo")
        
        print("📋 Pasos a ejecutar:")
        print("  1. Verificar dependencias")
        print("  2. Feature engineering")
        print("  3. Entrenar y optimizar")
        print("  4. Validar calibración")
        print("  5. Walk-Forward Validation")
        
        input("\n👉 Presiona ENTER para comenzar...")
        
        steps = [
            self.verify_dependencies,
            self.feature_engineering,
            self.train_and_optimize,
            self.validate_calibration,
            self.walk_forward_validation
        ]
        
        for step in steps:
            if not step():
                print("\n❌ Entrenamiento interrumpido por error")
                self.print_summary()
                return False
        
        self.print_summary()
        return True
    
    def run_validate_only(self):
        """Solo valida (asume que ya hay modelo)"""
        self.print_header("VALIDACIÓN - Solo Validar Modelo Existente")
        
        print("📋 Pasos a ejecutar:")
        print("  1. Verificar dependencias")
        print("  2. Validar calibración")
        print("  3. Backtesting")
        print("  4. Walk-Forward Validation")
        print("  5. Generar reportes")
        
        input("\n👉 Presiona ENTER para comenzar...")
        
        steps = [
            self.verify_dependencies,
            self.validate_calibration,
            self.run_backtesting,
            self.walk_forward_validation,
            self.generate_reports
        ]
        
        for step in steps:
            if not step():
                print("\n❌ Validación interrumpida por error")
                self.print_summary()
                return False
        
        self.print_summary()
        return True
    
    def print_summary(self):
        """Imprime resumen final"""
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        self.print_header("RESUMEN FINAL")
        
        print(f"⏱️  Tiempo total: {minutes} minutos, {seconds} segundos")
        print(f"✅ Pasos completados: {len(self.steps_completed)}")
        print(f"❌ Pasos fallidos: {len(self.steps_failed)}")
        
        if self.steps_completed:
            print("\n✅ Completados:")
            for step in self.steps_completed:
                print(f"  ✅ {step}")
        
        if self.steps_failed:
            print("\n❌ Fallidos:")
            for step in self.steps_failed:
                print(f"  ❌ {step}")
        
        if not self.steps_failed:
            print("\n" + "=" * 80)
            print("🎉 ¡PIPELINE COMPLETADO EXITOSAMENTE!")
            print("=" * 80)
            print("\n📁 Archivos generados:")
            print("  - modelos/random_forest_calibrado.pkl")
            print("  - resultados/REPORTE_FASE_2.html")
            print("  - resultados/walk_forward/REPORTE_VALIDACION_FINAL.txt")
            print("\n🚀 El modelo está listo para usar!")
            print("Ver README.md para instrucciones de uso")
        else:
            print("\n⚠️  Pipeline completado con errores")
            print("Revisa los mensajes de error arriba")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Pipeline maestro para Tennis ML Predictor (FIXED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  
  # Pipeline completo (primera vez)
  python setup_and_train_fixed.py --full
  
  # Solo entrenar (si ya tienes datos)
  python setup_and_train_fixed.py --train-only
  
  # Solo validar (si ya tienes modelo)
  python setup_and_train_fixed.py --validate-only
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--full', action='store_true',
                      help='Pipeline completo (datos + entrenamiento + validación)')
    group.add_argument('--train-only', action='store_true',
                      help='Solo entrenar modelo (asume que ya hay datos)')
    group.add_argument('--validate-only', action='store_true',
                      help='Solo validar modelo existente')
    
    args = parser.parse_args()
    
    # Crear pipeline
    pipeline = TennisPipeline()
    
    # Ejecutar según opción
    if args.full:
        success = pipeline.run_full_pipeline()
    elif args.train_only:
        success = pipeline.run_train_only()
    elif args.validate_only:
        success = pipeline.run_validate_only()
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()