"""
Sistema de reentrenamiento automático del modelo
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib
import logging
from sklearn.metrics import accuracy_score, brier_score_loss
import shutil
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import Config
from src.automation.data_updater import DataUpdater

logger = logging.getLogger(__name__)


class ModelRetrainer:
    """
    Sistema de reentrenamiento automático del modelo
    """

    def __init__(self, data_path=None, model_dir="modelos", backup_dir=None):

        self.data_path = Path(data_path or Config.DATA_PATH)
        self.model_dir = Path(model_dir)
        self.backup_dir = Path(backup_dir or Config.MODEL_BACKUP_DIR)

        self.model_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)

        self.current_model_path = Path(Config.MODEL_PATH)

    def deberia_reentrenar(self, estrategia=None):
        """
        Determina si es momento de reentrenar

        Estrategias:
        - 'diario': Reentrena cada día (muy intensivo)
        - 'semanal': Reentrena cada semana (RECOMENDADO)
        - 'mensual': Reentrena cada mes
        - 'threshold': Reentrena cuando hay N partidos nuevos
        """
        if estrategia is None:
            estrategia = Config.RETRAIN_STRATEGY

        # Verificar fecha último reentrenamiento
        if not self.current_model_path.exists():
            logger.info("🆕 No hay modelo en producción - reentrenamiento necesario")
            return True

        # Obtener fecha del modelo actual
        model_time = datetime.fromtimestamp(self.current_model_path.stat().st_mtime)
        days_since_retrain = (datetime.now() - model_time).days

        logger.info(f"📅 Días desde último reentrenamiento: {days_since_retrain}")

        if estrategia == "diario":
            return days_since_retrain >= 1
        elif estrategia == "semanal":
            return days_since_retrain >= 7
        elif estrategia == "mensual":
            return days_since_retrain >= 30
        elif estrategia == "threshold":
            # Contar partidos nuevos desde último reentrenamiento
            nuevos_partidos = self._contar_partidos_nuevos(model_time)
            logger.info(f"📊 Partidos nuevos: {nuevos_partidos}")
            threshold = Config.RETRAIN_THRESHOLD_MATCHES
            return nuevos_partidos >= threshold

        return False

    def _contar_partidos_nuevos(self, since_date):
        """
        Cuenta partidos nuevos desde una fecha
        """
        try:
            df = pd.read_csv(self.data_path)

            if "fecha" not in df.columns:
                return 0

            df["fecha"] = pd.to_datetime(df["fecha"])
            nuevos = df[df["fecha"] >= since_date]
            return len(nuevos)
        except:
            return 0

    def reentrenar_modelo(self):
        """
        Pipeline completo de reentrenamiento
        """
        logger.info("=" * 60)
        logger.info("🔄 INICIANDO REENTRENAMIENTO DEL MODELO")
        logger.info("=" * 60)

        try:
            # 1. Actualizar datos primero
            logger.info("\n📥 Paso 1: Actualizando datos...")
            if not self._actualizar_datos():
                logger.error("❌ Error actualizando datos")
                return False

            # 2. Cargar y preparar datos
            logger.info("\n📊 Paso 2: Cargando datos...")
            X_train, y_train, X_val, y_val, X_test, y_test = self._preparar_datos()

            if X_train is None:
                logger.error("❌ Error preparando datos")
                return False

            logger.info(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

            # 3. Entrenar nuevo modelo
            logger.info("\n🤖 Paso 3: Entrenando nuevo modelo...")
            nuevo_modelo = self._entrenar_nuevo_modelo(X_train, y_train, X_val, y_val)

            # 4. Evaluar nuevo modelo
            logger.info("\n📊 Paso 4: Evaluando nuevo modelo...")
            metricas_nuevo = self._evaluar_modelo(nuevo_modelo, X_test, y_test)

            # 5. Comparar con modelo actual (si existe)
            logger.info("\n⚖️  Paso 5: Comparando con modelo actual...")
            deberia_actualizar = self._comparar_con_actual(
                nuevo_modelo, metricas_nuevo, X_test, y_test
            )

            if deberia_actualizar:
                # 6. Backup del modelo actual
                logger.info("\n💾 Paso 6: Haciendo backup del modelo actual...")
                self._hacer_backup_modelo()

                # 7. Actualizar modelo en producción
                logger.info("\n🚀 Paso 7: Desplegando nuevo modelo...")
                self._desplegar_nuevo_modelo(nuevo_modelo, metricas_nuevo)

                logger.info("\n" + "=" * 60)
                logger.info("✅ REENTRENAMIENTO COMPLETADO EXITOSAMENTE")
                logger.info("=" * 60)
                return True
            else:
                logger.info("\n" + "=" * 60)
                logger.info("⚠️  NUEVO MODELO NO SUPERA AL ACTUAL")
                logger.info("   Manteniendo modelo en producción")
                logger.info("=" * 60)
                return False

        except Exception as e:
            logger.error(f"\n❌ ERROR EN REENTRENAMIENTO: {str(e)}", exc_info=True)
            return False

    def _actualizar_datos(self):
        """
        Actualiza datos antes de reentrenar
        """
        try:
            updater = DataUpdater(str(self.data_path))
            df_actualizado = updater.actualizar_si_necesario(force=True)

            if df_actualizado is not None:
                logger.info("✅ Datos actualizados correctamente")
                return True
            else:
                logger.warning("⚠️  No se pudieron actualizar datos")
                return True  # Continuar con datos existentes
        except Exception as e:
            logger.error(f"❌ Error actualizando datos: {e}")
            return False

    def _preparar_datos(self):
        """
        Prepara datos para entrenamiento
        """
        try:
            df = pd.read_csv(self.data_path)

            # Verificar columnas necesarias
            if "resultado" not in df.columns:
                logger.error("❌ Columna 'resultado' no encontrada")
                return None, None, None, None, None, None

            # Ordenar por fecha si existe
            if "fecha" in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"])
                df = df.sort_values("fecha").reset_index(drop=True)

            # Features (excluir resultado y fecha)
            exclude_cols = ["resultado", "fecha"]
            feature_cols = [col for col in df.columns if col not in exclude_cols]

            # Split temporal: 60% train, 20% val, 20% test
            n = len(df)
            train_end = int(n * 0.6)
            val_end = int(n * 0.8)

            X_train = df.iloc[:train_end][feature_cols]
            y_train = df.iloc[:train_end]["resultado"]

            X_val = df.iloc[train_end:val_end][feature_cols]
            y_val = df.iloc[train_end:val_end]["resultado"]

            X_test = df.iloc[val_end:][feature_cols]
            y_test = df.iloc[val_end:]["resultado"]

            return X_train, y_train, X_val, y_val, X_test, y_test

        except Exception as e:
            logger.error(f"Error preparando datos: {e}")
            return None, None, None, None, None, None

    def _entrenar_nuevo_modelo(self, X_train, y_train, X_val, y_val):
        """
        Entrena el nuevo modelo
        """
        from xgboost import XGBClassifier
        from sklearn.calibration import CalibratedClassifierCV

        # Usar configuración óptima encontrada en Fase 3
        modelo_base = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

        # Calibrar con cross-validation
        logger.info("   Entrenando y calibrando modelo...")
        modelo_calibrado = CalibratedClassifierCV(modelo_base, method="sigmoid", cv=5)

        # Entrenar en train+val para tener más datos
        import pandas as pd

        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        modelo_calibrado.fit(X_train_val, y_train_val)

        logger.info("   ✅ Modelo entrenado y calibrado")

        return modelo_calibrado

    def _evaluar_modelo(self, modelo, X_test, y_test):
        """
        Evalúa métricas del modelo
        """
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)

        metricas = {
            "accuracy": accuracy,
            "brier_score": brier,
            "fecha_evaluacion": datetime.now().isoformat(),
        }

        logger.info(f"   Accuracy:     {accuracy*100:.2f}%")
        logger.info(f"   Brier Score:  {brier:.4f}")

        return metricas

    def _comparar_con_actual(self, nuevo_modelo, metricas_nuevo, X_test, y_test):
        """
        Compara nuevo modelo con el actual
        """
        if not self.current_model_path.exists():
            logger.info("   No hay modelo actual - aceptando nuevo modelo")
            return True

        # Cargar modelo actual
        try:
            modelo_actual = joblib.load(self.current_model_path)

            # Evaluar modelo actual
            y_pred_actual = modelo_actual.predict(X_test)
            y_prob_actual = modelo_actual.predict_proba(X_test)[:, 1]

            acc_actual = accuracy_score(y_test, y_pred_actual)
            brier_actual = brier_score_loss(y_test, y_prob_actual)

            logger.info(f"\n   📊 COMPARACIÓN:")
            logger.info(f"   Modelo Actual:")
            logger.info(f"      Accuracy:     {acc_actual*100:.2f}%")
            logger.info(f"      Brier Score:  {brier_actual:.4f}")
            logger.info(f"   Modelo Nuevo:")
            logger.info(f"      Accuracy:     {metricas_nuevo['accuracy']*100:.2f}%")
            logger.info(f"      Brier Score:  {metricas_nuevo['brier_score']:.4f}")

            # Criterios de decisión
            mejora_accuracy = metricas_nuevo["accuracy"] > acc_actual
            mejora_brier = metricas_nuevo["brier_score"] < brier_actual

            # Aceptar si mejora en AMBAS métricas o si mejora significativamente en una
            if mejora_accuracy and mejora_brier:
                logger.info("\n   ✅ Nuevo modelo MEJORA en ambas métricas")
                return True
            elif mejora_brier and (brier_actual - metricas_nuevo["brier_score"]) > 0.01:
                logger.info("\n   ✅ Nuevo modelo MEJORA significativamente en calibración")
                return True
            else:
                logger.info("\n   ⚠️  Nuevo modelo NO supera al actual")
                return False

        except Exception as e:
            logger.error(f"   Error comparando modelos: {e}")
            return False

    def _hacer_backup_modelo(self):
        """
        Hace backup del modelo actual
        """
        if self.current_model_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"modelo_production_{timestamp}.pkl"

            shutil.copy2(self.current_model_path, backup_path)
            logger.info(f"   ✅ Backup guardado: {backup_path}")

    def _desplegar_nuevo_modelo(self, modelo, metricas):
        """
        Despliega nuevo modelo en producción
        """
        # Guardar modelo
        joblib.dump(modelo, self.current_model_path)
        logger.info(f"   ✅ Modelo guardado: {self.current_model_path}")

        # Guardar métricas
        metricas_path = self.model_dir / "production_model_metrics.json"
        import json

        with open(metricas_path, "w") as f:
            json.dump(metricas, f, indent=2)

        logger.info(f"   ✅ Métricas guardadas: {metricas_path}")


# Script principal
def main():
    """
    Script para ejecutar reentrenamiento
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/retraining.log"), logging.StreamHandler()],
    )

    retrainer = ModelRetrainer()

    # Verificar si es momento de reentrenar
    if retrainer.deberia_reentrenar():
        logger.info("🔄 Iniciando reentrenamiento...")
        success = retrainer.reentrenar_modelo()
        exit(0 if success else 1)
    else:
        logger.info("✅ No es necesario reentrenar aún")
        exit(0)


if __name__ == "__main__":
    main()
