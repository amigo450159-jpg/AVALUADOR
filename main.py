#!/usr/bin/env python3
"""
Sistema Avaluador de Computadores - Script Principal
==================================================

Este es el punto de entrada principal del sistema.
Ejecute este archivo para iniciar el avaluador.
"""

import sys
import os
from typing import Optional

def verificar_python() -> bool:
    """Verifica que la versión de Python sea compatible."""
    if sys.version_info < (3, 7):
        print("❌ Este sistema requiere Python 3.7 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    return True

def verificar_dependencias() -> bool:
    """Verifica que todas las dependencias estén instaladas."""
    dependencias = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('joblib', 'joblib')
    ]
    
    faltantes = []
    
    for modulo, paquete in dependencias:
        try:
            __import__(modulo)
        except ImportError:
            faltantes.append(paquete)
    
    if faltantes:
        print("❌ Faltan las siguientes dependencias:")
        for paquete in faltantes:
            print(f"   - {paquete}")
        print("\nPara instalarlas, ejecute:")
        print(f"   pip install {' '.join(faltantes)}")
        return False
    
    return True

def mostrar_ayuda():
    """Muestra información de ayuda."""
    print("""
🏪 SISTEMA AVALUADOR DE COMPUTADORES
====================================

USO:
    python main.py [opciones]

OPCIONES:
    -h, --help          Muestra esta ayuda
    -t, --tradicional   Solo modo tradicional (sin ML)
    -m, --ml           Solo modo machine learning
    -r, --reporte      Genera reporte de evaluaciones
    
EJEMPLOS:
    python main.py              # Inicia el sistema completo
    python main.py -t           # Solo evaluación tradicional
    python main.py -r           # Genera reporte
    
ARCHIVOS IMPORTANTES:
    main.py                     # Este archivo - punto de entrada
    sistema_completo.py         # Sistema integrado completo
    avaluador_interactivo.py   # Evaluación tradicional con preguntas
    modelo_prediccion.py        # Modelo de machine learning
    config.py                   # Configuración del sistema
    avaluador.py               # Procesamiento de datos original
    
DATASET:
    dataset_computadores_entrenamiento_LISTO.csv  # Datos para entrenar ML
    
EVALUACIONES:
    evaluaciones_computadores.json     # Evaluaciones tradicionales
    evaluaciones_completas.json        # Evaluaciones con ML
""")

def main():
    """Función principal."""
    import argparse
    
    # Configurar argumentos
    parser = argparse.ArgumentParser(description='Sistema Avaluador de Computadores', add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help='Muestra ayuda')
    parser.add_argument('-t', '--tradicional', action='store_true', help='Solo modo tradicional')
    parser.add_argument('-m', '--ml', action='store_true', help='Solo modo machine learning')
    parser.add_argument('-r', '--reporte', action='store_true', help='Genera reporte')
    
    args = parser.parse_args()
    
    # Mostrar ayuda si se solicitó
    if args.help:
        mostrar_ayuda()
        return
    
    print("🚀 Iniciando Sistema Avaluador de Computadores...")
    
    # Verificar requisitos
    if not verificar_python():
        return
    
    if not verificar_dependencias():
        return
    
    try:
        # Importar el sistema completo
        from sistema_completo import SistemaAvaluadorCompleto
        
        # Crear instancia del sistema
        sistema = SistemaAvaluadorCompleto()
        
        # Manejar opciones específicas
        if args.reporte:
            sistema.generar_reporte_completo()
            return
        
        if args.ml:
            print("🤖 Modo Machine Learning (en desarrollo)")
            # Aquí iría la lógica específica de ML
            return
        
        if args.tradicional:
            print("📋 Modo Evaluación Tradicional")
            # Inicializar solo el avaluador tradicional
            from avaluador_interactivo import AvaluadorComputador
            avaluador = AvaluadorComputador()
            resultado = avaluador.evaluar_computador()
            if resultado:
                avaluador.mostrar_resultado(resultado)
                guardar = input("\n¿Desea guardar esta evaluación? (s/n): ").strip().lower()
                if guardar == 's':
                    avaluador.guardar_evaluacion(resultado)
            return
        
        # Modo completo por defecto
        print("✅ Sistema completo iniciado")
        sistema.inicializar_modelo_ml()
        sistema.menu_principal()
        
    except KeyboardInterrupt:
        print("\n\n👋 Sistema interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("Por favor, verifique que todos los archivos estén presentes:")
        print("   - sistema_completo.py")
        print("   - avaluador_interactivo.py") 
        print("   - modelo_prediccion.py")
        print("   - config.py")
        print("   - avaluador.py (original)")

if __name__ == "__main__":
    main()