"""
Script rápido para entrenar el modelo de precios y guardar el archivo .pkl
Uso:
    python entrenar_ml.py

Requisitos:
    - dataset_computadores_entrenamiento_LISTO.csv en el mismo directorio
    - pandas, numpy, scikit-learn, joblib
"""

import sys
import os

def main() -> None:
    try:
        # Importar clase del modelo
        from modelo_prediccion import ModeloPrecioComputador

        print("🚀 Iniciando entrenamiento de modelo ML...")
        modelo = ModeloPrecioComputador()

        # Verificar dataset
        if not os.path.exists(modelo.archivo_dataset):
            print(f"❌ No se encontró el dataset: {modelo.archivo_dataset}")
            print("   Asegúrese de subir 'dataset_computadores_entrenamiento_LISTO.csv' al mismo directorio.")
            sys.exit(1)

        # Entrenar y guardar
        exito = modelo.entrenar_y_guardar('random_forest')
        if not exito:
            print("❌ Falló el entrenamiento del modelo.")
            sys.exit(1)

        # Confirmar archivo .pkl
        if os.path.exists(modelo.archivo_modelo):
            print(f"✅ Modelo guardado: {modelo.archivo_modelo}")
        else:
            print("⚠️  El archivo .pkl no se encontró tras el entrenamiento.")
            print("   Verifique permisos de escritura y ruta de trabajo actual.")

    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        print("   Instale dependencias: pip install pandas numpy scikit-learn joblib")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()