"""
Configuración del Sistema Avaluador de Computadores
====================================================

Este archivo contiene toda la configuración del sistema.
"""

import os
from typing import Optional
from typing import Dict, Any

# Carga sencilla de variables desde .env (sin dependencias externas)
def _load_dotenv_from_root():
    try:
        root = os.path.dirname(__file__)
        env_path = os.path.join(root, ".env")
        if not os.path.isfile(env_path):
            return
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Eliminar comillas si están presentes
                if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                    val = val[1:-1]
                # No sobrescribir si ya viene de entorno
                os.environ.setdefault(key, val)
    except Exception:
        # Silencioso: si falla la carga, continuamos con variables de entorno existentes
        pass

_load_dotenv_from_root()

# --- Azure Vision (Computer Vision) ---
# Credenciales vía variables de entorno para evitar exponer claves.
#   AZURE_VISION_ENDPOINT = https://<tu-recurso>.cognitiveservices.azure.com/
#   AZURE_VISION_KEY      = <clave>
def _clean_env(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    # Limpia espacios y comillas/backticks accidentales
    value = value.strip()
    # Elimina backticks y comillas simples/dobles en extremos
    value = value.strip("`")
    value = value.strip("'")
    value = value.strip('"')
    # Elimina ángulos si la clave se pegó como <KEY>
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    return value

AZURE_VISION_ENDPOINT = _clean_env(os.getenv("AZURE_VISION_ENDPOINT"))
AZURE_VISION_KEY = _clean_env(os.getenv("AZURE_VISION_KEY"))

# Configuración general
NOMBRE_SISTEMA = "Sistema Avaluador de Computadores"
VERSION = "2.0.0"
AUTOR = "Sistema Integrado"

# Configuración de archivos
ARCHIVOS = {
    'dataset_entrenamiento': 'dataset_computadores_entrenamiento_LISTO.csv',
    'modelo_ml': 'modelo_precio_computador.pkl',
    'evaluaciones_tradicionales': 'evaluaciones_computadores.json',
    'evaluaciones_completas': 'evaluaciones_completas.json',
    'log_sistema': 'avaluador_log.txt'
}

# Configuración de precios base (en dólares)
PRECIOS_BASE = {
    'laptop': {
        'bajo': 200,      # Uso básico
        'medio': 500,     # Uso intermedio
        'alto': 1000      # Uso profesional/gaming
    },
    'desktop': {
        'bajo': 150,      # Uso básico
        'medio': 400,     # Uso intermedio
        'alto': 800       # Uso profesional/gaming
    }
}

# Factores de ajuste
FACTORES_AJUSTE = {
    'condicion': {
        'excelente': 1.2,
        'buena': 1.0,
        'regular': 0.8,
        'mala': 0.6
    },
    'antiguedad': {
        '0-1': 1.0,       # Años
        '2-3': 0.9,
        '4-5': 0.7,
        '6+': 0.5
    },
    'componentes': {
        'ssd': 1.3,       # Multiplicador por tener SSD
        'grafica_dedicada': 1.4,  # Multiplicador por tarjeta gráfica
        'ram_alta': 1.2   # Más de 8GB RAM
    }
}

# Reglas de mercado para compraventa / préstamo
# Se aplica un factor al precio estimado para ofrecer el valor de préstamo.
# Ejemplo: 0.3 significa 30% del valor estimado del producto usado.
MARKET_RULES = {
    # 0.44 => ~10% aumento respecto a un factor previo de 0.40
    'factor_compraventa': 0.44,
    'min_prestamo': 100000,
    # Si True, también aplica el factor al precio de ML cuando se compare/recomiende.
    'aplicar_factor_ml': True,
}

# Puntuaciones de marcas (0-100)
MARCAS_PUNTUACION = {
    'apple': 95,
    'dell': 85,
    'hp': 80,
    'lenovo': 82,
    'asus': 78,
    'acer': 75,
    'msi': 83,
    'razer': 88,
    'samsung': 84,
    'lg': 79,
    'toshiba': 73,
    'sony': 81,
    'microsoft': 89,
    'huawei': 76,
    'xiaomi': 74
}

# Configuración del modelo ML
MODELO_ML = {
    'tipo_modelo': 'random_forest',  # 'random_forest' o 'linear_regression'
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,  # Para Random Forest
    'max_depth': 10,      # Para Random Forest
    'min_samples_split': 2  # Para Random Forest
}

# Características del modelo
CARACTERISTICAS_MODELO = [
    'marca_score',
    'es_ssd',
    'capacidad_disco_gb',
    'ram_gb',
    'generacion_procesador',
    'procesador_score',
    'tiene_grafica'
]

# Configuración de logging
LOGGING = {
    'nivel': 'INFO',
    'formato': '%(asctime)s - %(levelname)s - %(message)s',
    'archivo': 'avaluador.log'
}

# Mensajes del sistema
MENSAJES = {
    'bienvenida': """
    🏪 SISTEMA AVALUADOR DE COMPUTADORES
    ===================================
    
    Este sistema le ayudará a evaluar el precio de un computador
    basándose en sus características técnicas y condición actual.
    
    El sistema combina:
    ✅ Evaluación tradicional basada en reglas
    🤖 Predicción con Machine Learning (si disponible)
    📊 Comparación y análisis de resultados
    """,
    
    'instrucciones': """
    📋 INSTRUCCIONES:
    
    1. Responda las preguntas sobre el computador
    2. El sistema calculará el precio con múltiples métodos
    3. Podrá guardar y comparar evaluaciones
    4. Genere reportes de sus evaluaciones
    
    Presione ENTER para continuar...
    """
}

def verificar_configuracion() -> bool:
    """Verifica que la configuración sea válida."""
    try:
        # Verificar que existan archivos necesarios
        for archivo in ARCHIVOS.values():
            if archivo.endswith('.csv') or archivo.endswith('.pkl'):
                # Estos archivos se pueden crear, no es error si no existen
                continue
        
        # Verificar que las puntuaciones estén en rango válido
        for marca, puntuacion in MARCAS_PUNTUACION.items():
            if not 0 <= puntuacion <= 100:
                print(f"⚠️  Puntuación inválida para {marca}: {puntuacion}")
                return False
        
        # Verificar que los factores sean positivos
        for factor, valores in FACTORES_AJUSTE.items():
            for key, valor in valores.items():
                if valor <= 0:
                    print(f"⚠️  Factor inválido en {factor}.{key}: {valor}")
                    return False
        
        print("✅ Configuración válida")
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def obtener_configuracion() -> Dict[str, Any]:
    """Retorna la configuración completa."""
    return {
        'nombre': NOMBRE_SISTEMA,
        'version': VERSION,
        'archivos': ARCHIVOS,
        'precios_base': PRECIOS_BASE,
        'factores_ajuste': FACTORES_AJUSTE,
        'marcas_puntuacion': MARCAS_PUNTUACION,
        'modelo_ml': MODELO_ML,
        'caracteristicas_modelo': CARACTERISTICAS_MODELO,
        'logging': LOGGING,
        'mensajes': MENSAJES
    }

if __name__ == "__main__":
    verificar_configuracion()