"""
Avaluador de Computadores para Préstamos
========================================

Este script procesa datos históricos de computadores para generar un dataset
listo para entrenar modelos de machine learning en Azure ML.

Autor: Tu nombre
Fecha: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =================================================================
# CONFIGURACIÓN Y CONSTANTES
# =================================================================

# Scores de reventa por marca
MARCA_SCORES = {
    'APPLE': {'reventa': 5}, 'DELL': {'reventa': 5},
    'LENOVO': {'reventa': 4}, 'HP': {'reventa': 4},
    'ASUS': {'reventa': 3}, 'ACER': {'reventa': 2},
    'SAMSUNG': {'reventa': 3}, 'SONY': {'reventa': 3}, 
    'VICTUS': {'reventa': 4}, 'KOORUI': {'reventa': 1}, 
    'WINDOWS': {'reventa': 1}, 'GENERICO': {'reventa': 2}, 
    'LG': {'reventa': 3}, 'MSI': {'reventa': 4}, 
    'TOSHIBA': {'reventa': 2}
}

# Características para el modelo
FEATURES = [
    'marca_score', 'es_ssd', 'capacidad_disco_gb', 'ram_gb', 
    'generacion_procesador', 'procesador_score', 'tiene_grafica'
]

TARGET = 'precio_prestamo'

# =================================================================
# FUNCIONES DE PROCESAMIENTO
# =================================================================

def limpiar_marca(marca: str) -> str:
    """
    Limpia y estandariza el nombre de la marca.
    
    Args:
        marca: Nombre de la marca a limpiar
        
    Returns:
        Marca estandarizada en mayúsculas
    """
    if pd.isna(marca):
        return 'GENERICO'
    return str(marca).upper().strip()

def extraer_generacion_procesador(procesador: str) -> int:
    """
    Extrae la generación del procesador del nombre.
    
    Args:
        procesador: Nombre del procesador
        
    Returns:
        Número de generación (0-13)
    """
    if pd.isna(procesador):
        return 0
    
    procesador = str(procesador).upper()
    
    # Búsqueda de generaciones específicas
    if '13TH' in procesador:
        return 13
    elif '12TH' in procesador or '12 TH' in procesador:
        return 12
    elif '11TH' in procesador or '11 TH' in procesador:
        return 11
    elif '10TH' in procesador or '10 TH' in procesador:
        return 10
    elif '8TH' in procesador or '8 TH' in procesador:
        return 8
    elif '7TH' in procesador or '7 TG' in procesador:
        return 7
    elif '6TH' in procesador:
        return 6
    elif '3TH' in procesador:
        return 3
    elif '2TH' in procesador:
        return 2
    
    # Detección por series
    if 'RYZEN 7' in procesador or 'CORE I7' in procesador:
        return 7
    if 'RYZEN 5' in procesador or 'CORE I5' in procesador:
        return 5
    if 'RYZEN 3' in procesador or 'CORE I3' in procesador:
        return 3
    
    # Procesadores de gama baja
    if any(proc in procesador for proc in ['PENTIUM', 'CELERON', 'ATHLON', 'AMD A']):
        return 1
    
    return 0

def calcular_score_procesador(procesador: str) -> int:
    """
    Calcula el score del procesador basado en su gama.
    
    Args:
        procesador: Nombre del procesador
        
    Returns:
        Score del procesador (1-5)
    """
    if pd.isna(procesador):
        return 1
    
    procesador = str(procesador).upper()
    
    if 'I9' in procesador or 'RYZEN 9' in procesador:
        return 5
    elif 'I7' in procesador or 'RYZEN 7' in procesador:
        return 5
    elif 'I5' in procesador or 'RYZEN 5' in procesador:
        return 4
    elif 'I3' in procesador or 'RYZEN 3' in procesador:
        return 3
    elif any(proc in procesador for proc in ['PENTIUM', 'CELERON', 'ATHLON', 'A-SERIES']):
        return 1
    else:
        return 2

def limpiar_datos_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y estandariza las columnas numéricas.
    
    Args:
        df: DataFrame a limpiar
        
    Returns:
        DataFrame con datos numéricos limpios
    """
    df_copy = df.copy()
    
    # Convertir a tipos numéricos
    df_copy['capacidad_disco_gb'] = pd.to_numeric(df_copy['capacidad_disco_gb'], errors='coerce')
    df_copy['ram_gb'] = pd.to_numeric(df_copy['ram_gb'], errors='coerce')
    
    # Imputar valores faltantes con la mediana
    df_copy['capacidad_disco_gb'] = df_copy['capacidad_disco_gb'].fillna(df_copy['capacidad_disco_gb'].median())
    df_copy['ram_gb'] = df_copy['ram_gb'].fillna(df_copy['ram_gb'].median())
    
    # Convertir tipo de disco a binario
    df_copy['es_ssd'] = (df_copy['tipo_disco'] == 'SSD').astype(int)
    
    return df_copy

def validar_estructura_datos(data: Dict[str, List[Any]]) -> bool:
    """
    Valida que todas las columnas tengan la misma longitud.
    
    Args:
        data: Diccionario con los datos
        
    Returns:
        True si la estructura es válida
        
    Raises:
        ValueError: Si las longitudes no coinciden
    """
    longitudes = {k: len(v) for k, v in data.items()}
    logger.info(f"Longitudes por columna: {longitudes}")
    
    longitud_referencia = list(longitudes.values())[0]
    
    for columna, longitud in longitudes.items():
        if longitud != longitud_referencia:
            raise ValueError(f"La columna '{columna}' tiene {longitud} elementos, pero se esperaban {longitud_referencia}")
    
    logger.info(f"✅ Estructura válida: {longitud_referencia} registros por columna")
    return True

# =================================================================
# DATOS HISTÓRICOS
# =================================================================

def obtener_datos_historicos() -> Dict[str, List[Any]]:
    """
    Retorna los datos históricos de computadores.
    
    Returns:
        Diccionario con los datos históricos
    """
    return {
        'marca': [
            'ACER', 'GENERICO', 'GENERICO', 'GENERICO', 'GENERICO', 'GENERICO', 'GENERICO',
            'GENERICO', 'GENERICO', 'GENERICO', 'GENERICO', 'GENERICO', 'APPLE', 'GENERICO',
            'ASUS', 'ASUS', 'ASUS', 'ASUS', 'ASUS', 'ASUS', 'ASUS', 'ASUS', 'ASUS', 'ASUS',
            'ASUS', 'ASUS', 'ASUS', 'DELL', 'DELL', 'DELL', 'DELL', 'DELL', 'DELL', 'DELL',
            'DELL', 'DELL', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP',
            'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP', 'HP',
            'HP', 'HP', 'HP', 'HP', 'KOORUI', 'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO',
            'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO',
            'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO',
            'LENOVO', 'LENOVO', 'LENOVO', 'LENOVO', 'LG', 'LG', 'LG', 'MSI', 'MSI',
            'SAMSUNG', 'SAMSUNG', 'SAMSUNG'
        ],
        'tipo_disco': [
            'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'SSD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'SSD', 'HDD',
            'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'HDD',
            'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD',
            'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD',
            'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'SSD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD',
            'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD', 'HDD',
            'HDD', 'HDD', 'HDD', 'HDD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'SSD', 'HDD'
        ],
        'capacidad_disco_gb': [
            500, 500, 500, 500, 500, 256, 500, 500, 500, 500, 500, 500, 256, 500,
            512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 500,
            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
            500, 500, 500, 500, 500, 500, 500, 500, 512, 500, 500, 500, 500, 500,
            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
            500, 500, 500, 500, 256, 256, 256, 512, 512, 512, 512, 512, 500
        ],
        'ram_gb': [
            4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 8, 4,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 8, 8, 8, 16, 16, 8, 8, 8, 4
        ],
        'procesador': [
            'INTEL CORE I3-8130U', 'INTEL CELERON N4000', 'INTEL PENTIUM N5000', 'INTEL CORE I3-6006U', 'INTEL CELERON N3350',
            'INTEL CORE I5-1035G1', 'INTEL CELERON N3060', 'INTEL CORE I3-5005U', 'INTEL CELERON N3160', 'INTEL PENTIUM N3710',
            'INTEL CORE I3-4030U', 'INTEL CELERON N2840', 'APPLE M1', 'INTEL CELERON N2940',
            'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7',
            'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7',
            'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7', 'INTEL CORE I3-7130U',
            'INTEL CELERON N4000', 'INTEL PENTIUM N5000', 'INTEL CORE I3-6006U', 'INTEL CELERON N3350', 'INTEL CORE I5-1035G1',
            'INTEL CELERON N3060', 'INTEL CORE I3-5005U', 'INTEL CELERON N3160', 'INTEL PENTIUM N3710', 'INTEL CORE I3-4030U',
            'INTEL CELERON N2840', 'INTEL CORE I3-8130U', 'INTEL CELERON N4000', 'INTEL PENTIUM N5000', 'INTEL CORE I3-6006U',
            'INTEL CELERON N3350', 'INTEL CORE I5-1035G1', 'INTEL CELERON N3060', 'INTEL CORE I3-5005U', 'INTEL CELERON N3160',
            'INTEL PENTIUM N3710', 'INTEL CORE I3-4030U', 'INTEL CELERON N2840', 'INTEL CORE I3-8130U', 'INTEL CELERON N4000',
            'INTEL PENTIUM N5000', 'INTEL CORE I3-6006U', 'INTEL CELERON N3350', 'INTEL CORE I5-1035G1', 'INTEL CELERON N3060',
            'INTEL CORE I3-5005U', 'INTEL CELERON N3160', 'INTEL PENTIUM N3710', 'INTEL CORE I3-4030U', 'INTEL CELERON N2840',
            'AMD RYZEN 5 5500U', 'INTEL CORE I3-7130U', 'INTEL CELERON N4000', 'INTEL PENTIUM N5000', 'INTEL CORE I3-6006U',
            'INTEL CELERON N3350', 'INTEL CORE I5-1035G1', 'INTEL CELERON N3060', 'INTEL CORE I3-5005U', 'INTEL CELERON N3160',
            'INTEL PENTIUM N3710', 'INTEL CORE I3-4030U', 'INTEL CELERON N2840', 'INTEL CORE I3-8130U', 'INTEL CELERON N4000',
            'INTEL PENTIUM N5000', 'INTEL CORE I3-6006U', 'INTEL CELERON N3350', 'INTEL CORE I5-1035G1', 'INTEL CELERON N3060',
            'INTEL CORE I3-5005U', 'INTEL CELERON N3160', 'INTEL PENTIUM N3710', 'INTEL CORE I3-4030U', 'INTEL CELERON N2840',
            'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7', 'INTEL CORE I7-1165G7', 'AMD RYZEN 7 5800H', 'AMD RYZEN 7 5800H',
            'INTEL CORE I7-1185G7', 'INTEL CORE I7-1185G7', 'INTEL CORE I7-1185G7', 'INTEL CELERON N2840'
        ],
        'tiene_grafica': [
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0
        ],
        'precio_prestamo': [
            150000, 150000, 150000, 150000, 150000, 450000, 150000, 150000, 150000, 150000, 150000, 150000, 600000, 150000,
            800000, 800000, 800000, 800000, 800000, 800000, 800000, 800000, 800000, 800000, 800000, 800000, 800000, 150000,
            150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000,
            150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000,
            150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 850000, 150000, 150000, 150000, 150000, 150000,
            150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000,
            150000, 150000, 150000, 150000, 500000, 500000, 500000, 950000, 950000, 700000, 700000, 700000, 150000
        ]
    }

# =================================================================
# PROCESAMIENTO PRINCIPAL
# =================================================================

def procesar_dataset(data: Dict[str, List[Any]]) -> pd.DataFrame:
    """
    Procesa el dataset completo aplicando todas las transformaciones.
    
    Args:
        data: Datos históricos sin procesar
        
    Returns:
        DataFrame procesado y listo para Azure ML
    """
    logger.info("🔄 Iniciando procesamiento del dataset...")
    
    # Validar estructura de datos
    validar_estructura_datos(data)
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    logger.info(f"📊 Dataset creado: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    # Procesar marca
    df['marca'] = df['marca'].apply(limpiar_marca)
    df['marca_score'] = df['marca'].apply(lambda x: MARCA_SCORES.get(x, MARCA_SCORES['GENERICO'])['reventa'])
    logger.info("✅ Marcas procesadas")
    
    # Procesar procesador
    df['generacion_procesador'] = df['procesador'].apply(extraer_generacion_procesador)
    df['procesador_score'] = df['procesador'].apply(calcular_score_procesador)
    logger.info("✅ Procesadores procesados")
    
    # Limpiar datos numéricos
    df = limpiar_datos_numericos(df)
    logger.info("✅ Datos numéricos limpiados")
    
    # Seleccionar columnas finales
    df_final = df[FEATURES + [TARGET]].copy()
    
    logger.info(f"✅ Dataset final creado: {df_final.shape[0]} filas × {df_final.shape[1]} columnas")
    logger.info(f"📋 Columnas: {list(df_final.columns)}")
    
    return df_final

def guardar_dataset(df: pd.DataFrame, nombre_archivo: str = 'dataset_computadores_entrenamiento_LISTO.csv') -> None:
    """
    Guarda el dataset procesado en un archivo CSV.
    
    Args:
        df: DataFrame a guardar
        nombre_archivo: Nombre del archivo de salida
    """
    try:
        df.to_csv(nombre_archivo, index=False, encoding='utf-8')
        logger.info(f"💾 Dataset guardado exitosamente: '{nombre_archivo}'")
        logger.info(f"📈 Total de registros: {len(df)}")
    except Exception as e:
        logger.error(f"❌ Error al guardar el dataset: {str(e)}")
        raise

def mostrar_resumen_dataset(df: pd.DataFrame) -> None:
    """
    Muestra un resumen estadístico del dataset.
    
    Args:
        df: DataFrame a analizar
    """
    print("\n" + "="*60)
    print("📊 RESUMEN DEL DATASET")
    print("="*60)
    
    print(f"\nTotal de registros: {len(df)}")
    print(f"Total de columnas: {len(df.columns)}")
    
    print(f"\nColumnas del dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nPrimeras 5 filas:")
    print(df.head())
    
    print(f"\nEstadísticas descriptivas:")
    print(df.describe())

# =================================================================
# FUNCIÓN PRINCIPAL
# =================================================================

def main():
    """
    Función principal del script.
    """
    logger.info("🚀 Iniciando Avaluador de Computadores para Azure ML")
    
    try:
        # Obtener datos históricos
        data = obtener_datos_historicos()
        
        # Procesar dataset
        df_procesado = procesar_dataset(data)
        
        # Guardar dataset
        guardar_dataset(df_procesado)
        
        # Mostrar resumen
        mostrar_resumen_dataset(df_procesado)
        
        logger.info("✅ Proceso completado exitosamente")
        logger.info("🎯 El dataset está listo para ser subido a Azure Machine Learning")
        
    except Exception as e:
        logger.error(f"❌ Error en el proceso: {str(e)}")
        raise

# =================================================================
# EJECUCIÓN DEL SCRIPT
# =================================================================

if __name__ == "__main__":
    main()