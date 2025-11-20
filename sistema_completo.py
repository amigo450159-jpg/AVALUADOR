"""
Sistema Completo de Avaluador de Computadores
=============================================

Sistema integrado que combina:
1. Evaluación interactiva con preguntas
2. Predicción basada en machine learning
3. Gestión de evaluaciones guardadas
4. Reportes y estadísticas
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Agregar el directorio actual al path para importar módulos locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from avaluador_interactivo import AvaluadorComputador
    from modelo_prediccion import ModeloPrecioComputador, convertir_datos_entrada
    from config import ARCHIVOS, MARCAS_PUNTUACION, PRECIOS_BASE, FACTORES_AJUSTE, MODELO_ML, MARKET_RULES
except ImportError as e:
    print(f"❌ Error al importar módulos: {e}")
    print("Asegúrese de que existan todos los archivos necesarios.")
    sys.exit(1)

class SistemaAvaluadorCompleto:
    """Sistema completo que integra evaluación interactiva y ML."""
    
    def __init__(self):
        self.avaluador = AvaluadorComputador()
        self.modelo_ml = ModeloPrecioComputador()
        self.usar_ml = False
        self._verificar_requisitos()
    
    def _verificar_requisitos(self) -> None:
        """Verifica que todos los requisitos estén instalados."""
        try:
            import sklearn
            import joblib
        except ImportError:
            print("❌ Faltan dependencias:")
            print("   pip install scikit-learn joblib")
            sys.exit(1)
    
    def inicializar_modelo_ml(self) -> bool:
        """Inicializa el modelo de machine learning."""
        print("\n🤖 CONFIGURACIÓN DE MACHINE LEARNING")
        print("-" * 40)
        
        # Verificar si existe el dataset
        dataset_file = 'dataset_computadores_entrenamiento_LISTO.csv'
        if not os.path.exists(dataset_file):
            print(f"⚠️  No se encontró el dataset: {dataset_file}")
            print("   El modelo ML no estará disponible.")
            print("   Use el avaluador tradicional en su lugar.")
            return False
        
        # Verificar si existe modelo guardado
        modelo_file = 'modelo_precio_computador.pkl'
        if os.path.exists(modelo_file):
            print("📂 Cargando modelo de ML existente...")
            if self.modelo_ml.cargar_modelo():
                print("✅ Modelo ML cargado exitosamente.")
                self.usar_ml = True
                return True
        
        # Entrenar nuevo modelo
        print("🔄 Entrenando nuevo modelo de ML...")
        print("   Esto puede tomar un momento...")
        
        if self.modelo_ml.entrenar_y_guardar('random_forest'):
            print("✅ Modelo ML entrenado y guardado.")
            self.usar_ml = True
            return True
        else:
            print("❌ Error al entrenar el modelo ML.")
            print("   Use el avaluador tradicional en su lugar.")
            return False
    
    def evaluar_con_ml(self, datos_computador: Dict[str, Any]) -> Optional[float]:
        """Evalúa el precio usando machine learning."""
        if not self.usar_ml:
            return None
        
        try:
            # Convertir datos al formato necesario
            datos_ml = {
                'marca_score': datos_computador['marca_score'],
                'es_ssd': datos_computador['es_ssd'],
                'capacidad_disco_gb': datos_computador['capacidad_disco_gb'],
                'ram_gb': datos_computador['ram_gb'],
                'generacion_procesador': datos_computador['generacion_procesador'],
                'procesador_score': datos_computador['procesador_score'],
                'tiene_grafica': datos_computador['tiene_grafica']
            }
            
            # Predecir precio
            precio_ml = self.modelo_ml.predecir_precio(datos_ml)
            # Aplicar regla de compraventa si así se configuró
            if precio_ml is not None and MARKET_RULES.get('aplicar_factor_ml', True):
                factor = MARKET_RULES.get('factor_compraventa', 1.0)
                minimo = MARKET_RULES.get('min_prestamo', 100000)
                precio_ml = max(minimo, (precio_ml or 0) * factor)
            return precio_ml
            
        except Exception as e:
            print(f"⚠️  Error en predicción ML: {e}")
            return None
    
    def comparar_metodos(self, datos_computador: Dict[str, Any], 
                        precio_tradicional: float) -> Dict[str, Any]:
        """Compara los métodos de evaluación."""
        precio_ml = self.evaluar_con_ml(datos_computador)
        
        resultado = {
            'precio_tradicional': precio_tradicional,
            'precio_ml': precio_ml,
            'diferencia': None,
            'metodo_recomendado': 'tradicional',
            'razon': 'Método tradicional'
        }
        
        if precio_ml is not None:
            diferencia = abs(precio_tradicional - precio_ml)
            resultado['diferencia'] = diferencia
            resultado['diferencia_porcentaje'] = (diferencia / precio_tradicional) * 100
            
            # Decidir método recomendado
            if diferencia < precio_tradicional * 0.15:  # Diferencia menor al 15%
                resultado['metodo_recomendado'] = 'promedio'
                resultado['razon'] = 'Ambos métodos coinciden'
                resultado['precio_recomendado'] = (precio_tradicional + precio_ml) / 2
            elif precio_ml > precio_tradicional:
                resultado['metodo_recomendado'] = 'ml'
                resultado['razon'] = 'ML detecta valor adicional'
                resultado['precio_recomendado'] = precio_ml
            else:
                resultado['metodo_recomendado'] = 'tradicional'
                resultado['razon'] = 'Método tradicional más conservador'
                resultado['precio_recomendado'] = precio_tradicional
        
        return resultado
    
    def evaluar_computador_completo(self) -> Optional[Dict[str, Any]]:
        """Evalúa un computador con ambos métodos."""
        print(f"\n🖥️  EVALUACIÓN COMPLETA DE COMPUTADOR")
        print("="*60)
        
        # Usar el avaluador interactivo para recopilar datos
        resultado_tradicional = self.avaluador.evaluar_computador()
        
        if not resultado_tradicional:
            return None
        
        datos_computador = resultado_tradicional['datos_computador']
        precio_tradicional = resultado_tradicional['precio_final']
        
        # Comparar con ML si está disponible
        comparacion = self.comparar_metodos(datos_computador, precio_tradicional)
        
        # Crear resultado completo
        resultado_completo = {
            'datos_computador': datos_computador,
            'resultado_tradicional': resultado_tradicional,
            'comparacion_metodos': comparacion,
            'fecha_evaluacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'modelo_ml_disponible': self.usar_ml
        }
        
        # Mostrar resultados
        self.mostrar_resultado_completo(resultado_completo)
        
        return resultado_completo
    
    def mostrar_resultado_completo(self, resultado: Dict[str, Any]) -> None:
        """Muestra el resultado completo de la evaluación."""
        print(f"\n📊 RESULTADO COMPLETO DE LA EVALUACIÓN")
        print("="*60)
        
        # Mostrar datos del computador
        self.avaluador.mostrar_resultado(resultado['resultado_tradicional'])
        
        # Mostrar comparación si ML está disponible
        if resultado['modelo_ml_disponible']:
            comparacion = resultado['comparacion_metodos']
            
            print(f"\n🤖 ANÁLISIS CON MACHINE LEARNING")
            print("-" * 40)
            print(f"Precio tradicional: ${comparacion['precio_tradicional']:,.0f}")
            
            if comparacion['precio_ml'] is not None:
                print(f"Precio ML: ${comparacion['precio_ml']:,.0f}")
                print(f"Diferencia: ${comparacion['diferencia']:,.0f} ({comparacion['diferencia_porcentaje']:.1f}%)")
                print(f"\n✅ MÉTODO RECOMENDADO: {comparacion['metodo_recomendado'].upper()}")
                print(f"   Razón: {comparacion['razon']}")
                print(f"   💰 Precio final recomendado: ${comparacion['precio_recomendado']:,.0f}")
            else:
                print("⚠️  No se pudo obtener predicción del modelo ML")
        else:
            print(f"\n⚠️  Modelo ML no disponible - Usando método tradicional")
        
        print(f"\n📅 Fecha de evaluación: {resultado['fecha_evaluacion']}")
    
    def guardar_evaluacion_completa(self, resultado: Dict[str, Any]) -> None:
        """Guarda la evaluación completa."""
        try:
            archivo = 'evaluaciones_completas.json'
            
            # Cargar evaluaciones anteriores
            if os.path.exists(archivo):
                with open(archivo, 'r', encoding='utf-8') as f:
                    evaluaciones = json.load(f)
            else:
                evaluaciones = []
            
            # Añadir nueva evaluación
            evaluaciones.append(resultado)
            
            # Guardar
            with open(archivo, 'w', encoding='utf-8') as f:
                json.dump(evaluaciones, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Evaluación completa guardada en {archivo}")
            
        except Exception as e:
            print(f"❌ Error al guardar evaluación: {e}")
    
    def generar_reporte_completo(self) -> None:
        """Genera un reporte completo de todas las evaluaciones."""
        try:
            archivo_tradicional = 'evaluaciones_computadores.json'
            archivo_completo = 'evaluaciones_completas.json'
            
            print(f"\n📊 REPORTE COMPLETO DE EVALUACIONES")
            print("="*60)
            
            # Reporte de evaluaciones tradicionales
            if os.path.exists(archivo_tradicional):
                import json
                with open(archivo_tradicional, 'r', encoding='utf-8') as f:
                    tradicionales = json.load(f)
                
                print(f"\n📋 Evaluaciones tradicionales: {len(tradicionales)}")
                if tradicionales:
                    precios = [eval['precio_final'] for eval in tradicionales]
                    print(f"   Precio promedio: ${np.mean(precios):,.0f}")
                    print(f"   Rango: ${min(precios):,.0f} - ${max(precios):,.0f}")
            
            # Reporte de evaluaciones completas
            if os.path.exists(archivo_completo):
                with open(archivo_completo, 'r', encoding='utf-8') as f:
                    completas = json.load(f)
                
                print(f"\n🔬 Evaluaciones completas (con ML): {len(completas)}")
                if completas:
                    precios_ml = []
                    for eval in completas:
                        if 'comparacion_metodos' in eval and eval['comparacion_metodos']['precio_ml']:
                            precios_ml.append(eval['comparacion_metodos']['precio_recomendado'])
                    
                    if precios_ml:
                        print(f"   Precio promedio (ML): ${np.mean(precios_ml):,.0f}")
                        print(f"   Rango (ML): ${min(precios_ml):,.0f} - ${max(precios_ml):,.0f}")
            
            # Estadísticas generales
            print(f"\n📈 ESTADÍSTICAS GENERALES:")
            print(f"   Total de evaluaciones: {len(tradicionales) if 'tradicionales' in locals() else 0}")
            print(f"   Modelo ML disponible: {'Sí' if self.usar_ml else 'No'}")
            
        except Exception as e:
            print(f"❌ Error al generar reporte: {e}")
    
    def menu_principal(self) -> None:
        """Muestra el menú principal del sistema."""
        print("\n" + "="*60)
        print("🏪 SISTEMA COMPLETO DE AVALUADOR DE COMPUTADORES")
        print("="*60)
        print("Este sistema combina evaluación tradicional y machine learning")
        print(f"Estado del modelo ML: {'✅ Disponible' if self.usar_ml else '❌ No disponible'}")
        
        while True:
            print(f"\n📋 MENÚ PRINCIPAL")
            print("-" * 30)
            print("1. 🖥️  Evaluar computador (completo)")
            print("2. 🔬 Solo evaluación tradicional")
            print("3. 🤖 Solo predicción ML (si disponible)")
            print("4. 📊 Ver reportes")
            print("5. ⚙️  Configurar modelo ML")
            print("6. ❌ Salir")
            
            opcion = input("\nSeleccione una opción: ").strip()
            
            if opcion == '1':
                resultado = self.evaluar_computador_completo()
                if resultado:
                    guardar = input("\n¿Desea guardar esta evaluación? (s/n): ").strip().lower()
                    if guardar == 's':
                        self.guardar_evaluacion_completa(resultado)
                        print("✅ Evaluación guardada exitosamente.")
            
            elif opcion == '2':
                # Solo evaluación tradicional
                resultado = self.avaluador.evaluar_computador()
                if resultado:
                    self.avaluador.mostrar_resultado(resultado)
                    guardar = input("\n¿Desea guardar esta evaluación? (s/n): ").strip().lower()
                    if guardar == 's':
                        self.avaluador.guardar_evaluacion(resultado)
            
            elif opcion == '3':
                # Solo predicción ML
                if not self.usar_ml:
                    print("⚠️  El modelo ML no está disponible.")
                    continue
                
                print("\n🤖 PREDICCIÓN CON MACHINE LEARNING")
                print("Ingrese las características del computador:")
                
                # Aquí iría la lógica para ingresar datos y predecir
                print("⚠️  Funcionalidad en desarrollo...")
            
            elif opcion == '4':
                self.generar_reporte_completo()
            
            elif opcion == '5':
                print("\n⚙️  CONFIGURACIÓN DE MODELO ML")
                print("1. Entrenar nuevo modelo")
                print("2. Recargar modelo existente")
                print("3. Ver importancia de características")
                
                subopcion = input("Seleccione una opción: ").strip()
                
                if subopcion == '1':
                    self.inicializar_modelo_ml()
                elif subopcion == '2':
                    if self.modelo_ml.cargar_modelo():
                        print("✅ Modelo recargado.")
                elif subopcion == '3':
                    if self.usar_ml:
                        self.modelo_ml.mostrar_importancia_caracteristicas()
                    else:
                        print("⚠️  Modelo ML no disponible.")
            
            elif opcion == '6':
                print("👋 ¡Gracias por usar el sistema avaluador!")
                break
            
            else:
                print("❌ Opción inválida. Por favor intente nuevamente.")

def main():
    """Función principal."""
    # Crear instancia del sistema
    sistema = SistemaAvaluadorCompleto()
    
    # Inicializar modelo ML si es posible
    print("🚀 Iniciando Sistema Avaluador Completo...")
    sistema.inicializar_modelo_ml()
    
    # Mostrar menú principal
    sistema.menu_principal()

if __name__ == "__main__":
    main()