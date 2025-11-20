"""
Modelo de Predicción de Precios para Computadores
==================================================

Este módulo carga el dataset procesado y entrena un modelo de machine learning
para predecir precios de préstamo basándose en las características del computador.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from typing import Dict, Any, Optional
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModeloPrecioComputador:
    """Clase para entrenar y usar modelos de predicción de precios."""
    
    def __init__(self):
        self.modelo = None
        self.caracteristicas = None
        self.modelo_entrenado = False
        self.archivo_modelo = 'modelo_precio_computador.pkl'
        self.archivo_dataset = 'dataset_computadores_entrenamiento_LISTO.csv'
    
    def cargar_dataset(self) -> Optional[pd.DataFrame]:
        """Carga el dataset de entrenamiento."""
        try:
            if not os.path.exists(self.archivo_dataset):
                logger.error(f"❌ No se encontró el archivo: {self.archivo_dataset}")
                return None
            
            df = pd.read_csv(self.archivo_dataset)
            logger.info(f"✅ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error al cargar dataset: {str(e)}")
            return None
    
    def preparar_datos(self, df: pd.DataFrame) -> tuple:
        """Prepara los datos para entrenamiento."""
        try:
            # Separar características y target
            X = df.drop('precio_prestamo', axis=1)
            y = df['precio_prestamo']
            
            # Guardar nombres de características
            self.caracteristicas = X.columns.tolist()
            
            logger.info(f"✅ Datos preparados - Características: {len(self.caracteristicas)}")
            logger.info(f"Características: {self.caracteristicas}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error al preparar datos: {str(e)}")
            return None, None
    
    def entrenar_modelo(self, X: pd.DataFrame, y: pd.Series, tipo_modelo: str = 'random_forest') -> bool:
        """Entrena el modelo de predicción."""
        try:
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Seleccionar y entrenar modelo
            if tipo_modelo == 'random_forest':
                self.modelo = RandomForestRegressor(n_estimators=100, random_state=42)
            elif tipo_modelo == 'linear_regression':
                self.modelo = LinearRegression()
            else:
                logger.error(f"❌ Tipo de modelo no válido: {tipo_modelo}")
                return False
            
            logger.info(f"🔄 Entrenando modelo {tipo_modelo}...")
            self.modelo.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = self.modelo.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"✅ Modelo entrenado exitosamente")
            logger.info(f"📊 Métricas del modelo:")
            logger.info(f"  - MAE (Error Absoluto Medio): ${mae:,.0f}")
            logger.info(f"  - RMSE (Raíz del Error Cuadrático Medio): ${rmse:,.0f}")
            logger.info(f"  - R² (Coeficiente de Determinación): {r2:.3f}")
            
            self.modelo_entrenado = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al entrenar modelo: {str(e)}")
            return False
    
    def guardar_modelo(self) -> bool:
        """Guarda el modelo entrenado."""
        try:
            if not self.modelo_entrenado:
                logger.error("❌ No hay modelo entrenado para guardar")
                return False
            
            modelo_data = {
                'modelo': self.modelo,
                'caracteristicas': self.caracteristicas
            }
            
            joblib.dump(modelo_data, self.archivo_modelo)
            logger.info(f"✅ Modelo guardado en: {self.archivo_modelo}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al guardar modelo: {str(e)}")
            return False
    
    def cargar_modelo(self) -> bool:
        """Carga un modelo previamente entrenado."""
        try:
            if not os.path.exists(self.archivo_modelo):
                logger.error(f"❌ No se encontró el archivo: {self.archivo_modelo}")
                return False
            
            modelo_data = joblib.load(self.archivo_modelo)
            self.modelo = modelo_data['modelo']
            self.caracteristicas = modelo_data['caracteristicas']
            self.modelo_entrenado = True
            
            logger.info(f"✅ Modelo cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {str(e)}")
            return False
    
    def predecir_precio(self, caracteristicas: Dict[str, Any]) -> Optional[float]:
        """Predice el precio de un computador dadas sus características."""
        try:
            if not self.modelo_entrenado:
                logger.error("❌ No hay modelo cargado o entrenado")
                return None
            
            # Crear array de características en el orden correcto
            valores = []
            for feature in self.caracteristicas:
                if feature not in caracteristicas:
                    logger.error(f"❌ Falta la característica: {feature}")
                    return None
                valores.append(caracteristicas[feature])
            
            # Convertir a array de numpy
            X = np.array([valores])
            
            # Predecir
            precio_predicho = self.modelo.predict(X)[0]
            
            logger.info(f"✅ Predicción realizada: ${precio_predicho:,.0f}")
            return precio_predicho
            
        except Exception as e:
            logger.error(f"❌ Error al predecir precio: {str(e)}")
            return None
    
    def mostrar_importancia_caracteristicas(self) -> None:
        """Muestra la importancia de las características (solo para Random Forest)."""
        try:
            if not self.modelo_entrenado:
                logger.error("❌ No hay modelo cargado")
                return
            
            if hasattr(self.modelo, 'feature_importances_'):
                importancias = self.modelo.feature_importances_
                
                print(f"\n📊 IMPORTANCIA DE CARACTERÍSTICAS:")
                print("-" * 40)
                
                # Crear lista de tuplas (característica, importancia)
                feature_importance = list(zip(self.caracteristicas, importancias))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                for feature, importance in feature_importance:
                    print(f"{feature:25s}: {importance:.3f}")
                    
            else:
                logger.info("ℹ️  Este modelo no proporciona importancia de características")
                
        except Exception as e:
            logger.error(f"❌ Error al mostrar importancias: {str(e)}")
    
    def entrenar_y_guardar(self, tipo_modelo: str = 'random_forest') -> bool:
        """Pipeline completo: entrenar y guardar modelo."""
        logger.info("🔄 Iniciando pipeline de entrenamiento...")
        
        # Cargar dataset
        df = self.cargar_dataset()
        if df is None:
            return False
        
        # Preparar datos
        X, y = self.preparar_datos(df)
        if X is None or y is None:
            return False
        
        # Entrenar modelo
        if not self.entrenar_modelo(X, y, tipo_modelo):
            return False
        
        # Guardar modelo
        if not self.guardar_modelo():
            return False
        
        # Mostrar importancia de características
        self.mostrar_importancia_caracteristicas()
        
        logger.info("✅ Pipeline de entrenamiento completado")
        return True

# =================================================================
# FUNCIÓN AUXILIAR PARA CONVERTIR DATOS
# =================================================================

def convertir_datos_entrada(marca: str, tipo_disco: str, capacidad_disco_gb: float,
                           ram_gb: float, procesador: str, tiene_grafica: bool) -> Dict[str, Any]:
    """Convierte los datos de entrada al formato necesario para el modelo."""
    
    # Importar funciones del avaluador
    from avaluador import limpiar_marca, extraer_generacion_procesador, calcular_score_procesador
    
    # Procesar datos
    marca_limpia = limpiar_marca(marca)
    marca_score = MARCA_SCORES.get(marca_limpia, MARCA_SCORES['GENERICO'])['reventa']
    
    es_ssd = 1 if tipo_disco.upper() == 'SSD' else 0
    
    generacion_procesador = extraer_generacion_procesador(procesador)
    procesador_score = calcular_score_procesador(procesador)
    
    tiene_grafica_num = 1 if tiene_grafica else 0
    
    return {
        'marca_score': marca_score,
        'es_ssd': es_ssd,
        'capacidad_disco_gb': capacidad_disco_gb,
        'ram_gb': ram_gb,
        'generacion_procesador': generacion_procesador,
        'procesador_score': procesador_score,
        'tiene_grafica': tiene_grafica_num
    }

# =================================================================
# EJEMPLO DE USO
# =================================================================

if __name__ == "__main__":
    # Crear instancia del modelo
    modelo = ModeloPrecioComputador()
    
    # Entrenar modelo si no existe
    if not os.path.exists('modelo_precio_computador.pkl'):
        print("🔄 No se encontró modelo guardado. Entrenando nuevo modelo...")
        exito = modelo.entrenar_y_guardar('random_forest')
        
        if not exito:
            print("❌ Error al entrenar el modelo")
            exit(1)
    else:
        # Cargar modelo existente
        print("📂 Cargando modelo existente...")
        if not modelo.cargar_modelo():
            print("❌ Error al cargar el modelo")
            exit(1)
    
    # Ejemplo de predicción
    print("\n🧪 Ejemplo de predicción:")
    
    # Datos de ejemplo
    datos_ejemplo = {
        'marca_score': 5,      # Apple
        'es_ssd': 1,           # SSD
        'capacidad_disco_gb': 512,
        'ram_gb': 8,
        'generacion_procesador': 11,
        'procesador_score': 5,
        'tiene_grafica': 1
    }
    
    precio_predicho = modelo.predecir_precio(datos_ejemplo)
    
    if precio_predicho:
        print(f"💰 Precio predicho: ${precio_predicho:,.0f}")
    
    # Mostrar importancia de características
    modelo.mostrar_importancia_caracteristicas()