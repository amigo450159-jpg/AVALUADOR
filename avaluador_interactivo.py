"""
Avaluador Interactivo de Computadores para Préstamos
====================================================

Sistema interactivo que evalúa el precio de préstamo de computadores
haciendo preguntas al usuario y utilizando machine learning.

Autor: Tu nombre
Fecha: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json
import os
from config import MARKET_RULES

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =================================================================
# CONFIGURACIÓN Y CONSTANTES
# =================================================================

# Scores de reventa por marca
MARCA_SCORES = {
    'APPLE': {'reventa': 5, 'nombre': 'Apple'}, 
    'DELL': {'reventa': 5, 'nombre': 'Dell'},
    'LENOVO': {'reventa': 4, 'nombre': 'Lenovo'}, 
    'HP': {'reventa': 4, 'nombre': 'HP'},
    'ASUS': {'reventa': 3, 'nombre': 'Asus'}, 
    'ACER': {'reventa': 2, 'nombre': 'Acer'},
    'SAMSUNG': {'reventa': 3, 'nombre': 'Samsung'}, 
    'SONY': {'reventa': 3, 'nombre': 'Sony'}, 
    'VICTUS': {'reventa': 4, 'nombre': 'Victus'}, 
    'KOORUI': {'reventa': 1, 'nombre': 'Koorui'}, 
    'WINDOWS': {'reventa': 1, 'nombre': 'Windows'}, 
    'GENERICO': {'reventa': 2, 'nombre': 'Genérico'}, 
    'LG': {'reventa': 3, 'nombre': 'LG'}, 
    'MSI': {'reventa': 4, 'nombre': 'MSI'}, 
    'TOSHIBA': {'reventa': 2, 'nombre': 'Toshiba'}
}

# Rangos de precios base por categoría
RANGOS_PRECIO = {
    'bajo': {'min': 100000, 'max': 300000, 'descripcion': 'Computador básico'},
    'medio': {'min': 300000, 'max': 700000, 'descripcion': 'Computador medio'},
    'alto': {'min': 700000, 'max': 1500000, 'descripcion': 'Computador alto rendimiento'}
}

# =================================================================
# FUNCIONES DE PROCESAMIENTO
# =================================================================

def limpiar_marca(marca: str) -> str:
    """Limpia y estandariza el nombre de la marca."""
    if pd.isna(marca):
        return 'GENERICO'
    marca_limpia = str(marca).upper().strip()
    # Buscar coincidencias parciales
    for key, value in MARCA_SCORES.items():
        if key in marca_limpia:
            return key
    return 'GENERICO'

def extraer_generacion_procesador(procesador: str) -> int:
    """Extrae la generación del procesador del nombre."""
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
    """Calcula el score del procesador basado en su gama."""
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

def calcular_antiguedad_factor(anio: int) -> float:
    """Calcula el factor de depreciación por antigüedad."""
    anio_actual = datetime.now().year
    antiguedad = anio_actual - anio
    
    if antiguedad <= 1:
        return 1.0  # Sin depreciación
    elif antiguedad <= 2:
        return 0.85  # 15% depreciación
    elif antiguedad <= 3:
        return 0.70  # 30% depreciación
    elif antiguedad <= 5:
        return 0.50  # 50% depreciación
    else:
        return 0.30  # 70% depreciación

# =================================================================
# CLASE AVALUADOR
# =================================================================

class AvaluadorComputador:
    """Clase principal para evaluar computadores."""
    
    def __init__(self):
        self.datos_computador = {}
        self.precio_base = 0
        self.precio_final = 0
        self.historial_evaluaciones = []
        
    def hacer_pregunta(self, pregunta: str, opciones: Optional[List[str]] = None, 
                      tipo: str = 'texto', min_val: Optional[float] = None, 
                      max_val: Optional[float] = None) -> str:
        """Hace una pregunta al usuario y valida la respuesta."""
        while True:
            try:
                print(f"\n❓ {pregunta}")
                
                if opciones:
                    print("Opciones:")
                    for i, opcion in enumerate(opciones, 1):
                        print(f"  {i}. {opcion}")
                    
                    respuesta = input("Seleccione una opción (número): ").strip()
                    if respuesta.isdigit() and 1 <= int(respuesta) <= len(opciones):
                        return opciones[int(respuesta) - 1]
                    else:
                        print("❌ Opción inválida. Por favor seleccione un número válido.")
                        continue
                
                respuesta = input("Su respuesta: ").strip()
                
                if tipo == 'numero':
                    respuesta_num = float(respuesta)
                    if min_val is not None and respuesta_num < min_val:
                        print(f"❌ El valor debe ser mayor o igual a {min_val}")
                        continue
                    if max_val is not None and respuesta_num > max_val:
                        print(f"❌ El valor debe ser menor o igual a {max_val}")
                        continue
                    return str(respuesta_num)
                
                if not respuesta:
                    print("❌ Por favor ingrese una respuesta válida.")
                    continue
                
                return respuesta
                
            except ValueError:
                print("❌ Entrada inválida. Por favor intente nuevamente.")
                continue
    
    def evaluar_marca(self, marca: str) -> Dict[str, Any]:
        """Evalúa la marca del computador."""
        marca_limpia = limpiar_marca(marca)
        marca_info = MARCA_SCORES.get(marca_limpia, MARCA_SCORES['GENERICO'])
        
        return {
            'marca_original': marca,
            'marca_estandarizada': marca_limpia,
            'marca_score': marca_info['reventa'],
            'nombre_comercial': marca_info['nombre']
        }
    
    def evaluar_procesador(self, procesador: str) -> Dict[str, Any]:
        """Evalúa el procesador del computador."""
        return {
            'procesador_original': procesador,
            'generacion_procesador': extraer_generacion_procesador(procesador),
            'procesador_score': calcular_score_procesador(procesador)
        }
    
    def calcular_precio_base(self, datos: Dict[str, Any]) -> float:
        """Calcula el precio base según las características."""
        # Precio base por marca
        precio_marca = datos['marca_score'] * 50000
        
        # Precio por procesador
        precio_procesador = datos['procesador_score'] * 75000
        
        # Precio por RAM
        precio_ram = datos['ram_gb'] * 25000
        
        # Precio por disco
        precio_disco = datos['capacidad_disco_gb'] * 100
        if datos['es_ssd']:
            precio_disco *= 1.5  # SSD vale 50% más
        
        # Precio por gráfica: solo gamer/dedicada de alto rendimiento
        precio_grafica = 100000 if (datos.get('grafica_gamer', datos.get('tiene_grafica', 0))) else 0
        
        # Precio base total
        precio_base = precio_marca + precio_procesador + precio_ram + precio_disco + precio_grafica
        
        return max(precio_base, 100000)  # Mínimo 100.000
    
    def ajustar_por_condicion(self, precio: float, condicion: str) -> float:
        """Ajusta el precio según la condición del computador."""
        factores = {
            'Excelente': 1.0,
            'Muy buena': 0.95,
            'Buena': 0.85,
            'Regular': 0.70,
            'Mala': 0.50
        }
        
        return precio * factores.get(condicion, 0.85)
    
    def ajustar_por_antiguedad(self, precio: float, anio: int) -> float:
        """Ajusta el precio por antigüedad."""
        factor = calcular_antiguedad_factor(anio)
        return precio * factor
    
    def evaluar_computador(self) -> Dict[str, Any]:
        """Realiza la evaluación completa del computador."""
        print("\n" + "="*60)
        print("🏪 AVALUADOR DE COMPUTADORES PARA PRÉSTAMOS")
        print("="*60)
        print("Por favor responda las siguientes preguntas sobre el computador:")
        
        # Recopilar información del computador
        print("\n📱 1. INFORMACIÓN BÁSICA")
        print("-" * 40)
        
        # Marca
        marcas_disponibles = [info['nombre'] for info in MARCA_SCORES.values()]
        marca = self.hacer_pregunta(
            "¿Cuál es la marca del computador?",
            opciones=marcas_disponibles
        )
        
        # Modelo
        modelo = self.hacer_pregunta("¿Cuál es el modelo del computador?")
        
        # Año
        anio = int(self.hacer_pregunta(
            "¿En qué año fue fabricado?",
            tipo='numero',
            min_val=2010,
            max_val=2024
        ))
        
        print("\n💾 2. ESPECIFICACIONES TÉCNICAS")
        print("-" * 40)
        
        # Tipo de disco
        tipo_disco = self.hacer_pregunta(
            "¿Qué tipo de disco duro tiene?",
            opciones=['HDD', 'SSD']
        )
        
        # Capacidad del disco
        capacidad_disco = float(self.hacer_pregunta(
            "¿Cuál es la capacidad del disco duro (en GB)?",
            tipo='numero',
            min_val=128,
            max_val=4000
        ))
        
        # RAM
        ram_gb = float(self.hacer_pregunta(
            "¿Cuánta memoria RAM tiene (en GB)?",
            tipo='numero',
            min_val=2,
            max_val=64
        ))
        
        # Procesador
        procesador = self.hacer_pregunta("¿Qué procesador tiene? (Ej: Intel Core i5, AMD Ryzen 5)")
        
        # Gráfica
        tiene_grafica_dedicada = self.hacer_pregunta(
            "¿Tiene tarjeta gráfica dedicada?",
            opciones=['Sí', 'No']
        ) == 'Sí'
        grafica_gamer = False
        if tiene_grafica_dedicada:
            grafica_gamer = self.hacer_pregunta(
                "¿La tarjeta es gamer/alto rendimiento (NVIDIA GTX/RTX, AMD RX)?",
                opciones=['Sí', 'No']
            ) == 'Sí'
        
        print("\n🔧 3. ESTADO DEL EQUIPO")
        print("-" * 40)
        
        # Condición
        condicion = self.hacer_pregunta(
            "¿En qué estado está el computador?",
            opciones=['Excelente', 'Muy buena', 'Buena', 'Regular', 'Mala']
        )
        
        # Funcionamiento
        funciona_correctamente = self.hacer_pregunta(
            "¿Funciona correctamente?",
            opciones=['Sí', 'No']
        ) == 'Sí'
        
        if not funciona_correctamente:
            print("⚠️  El computador debe estar en buen estado funcional para ser aceptado.")
            return None
        
        # Evaluar componentes
        eval_marca = self.evaluar_marca(marca)
        eval_procesador = self.evaluar_procesador(procesador)
        
        # Crear datos del computador
        datos_computador = {
            **eval_marca,
            'modelo': modelo,
            'anio': anio,
            'tipo_disco': tipo_disco,
            'capacidad_disco_gb': capacidad_disco,
            'ram_gb': ram_gb,
            'grafica_gamer': 1 if grafica_gamer else 0,
            # Para ML, 'tiene_grafica' solo cuenta si es gamer/alto rendimiento
            'tiene_grafica': 1 if grafica_gamer else 0,
            'condicion': condicion,
            'funciona_correctamente': funciona_correctamente
        }
        
        # Añadir información del procesador
        datos_computador.update(eval_procesador)
        
        # Convertir tipo de disco a binario
        datos_computador['es_ssd'] = 1 if tipo_disco == 'SSD' else 0
        
        # Calcular precio (valor usado estimado)
        precio_base = self.calcular_precio_base(datos_computador)
        precio_ajustado = self.ajustar_por_condicion(precio_base, condicion)
        precio_valor_usado = self.ajustar_por_antiguedad(precio_ajustado, anio)

        # Aplicar regla de compraventa: ofrecer préstamo como porcentaje del valor usado estimado
        factor = MARKET_RULES.get('factor_compraventa', 1.0)
        minimo = MARKET_RULES.get('min_prestamo', 100000)
        precio_final = max(minimo, precio_valor_usado * factor)
        
        # Determinar categoría
        if precio_final < 300000:
            categoria = 'bajo'
        elif precio_final < 700000:
            categoria = 'medio'
        else:
            categoria = 'alto'
        
        resultado = {
            'datos_computador': datos_computador,
            'precio_base': precio_base,
            'precio_ajustado_condicion': precio_ajustado,
            'precio_final': precio_final,
            'categoria': categoria,
            'fecha_evaluacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return resultado
    
    def mostrar_resultado(self, resultado: Dict[str, Any]) -> None:
        """Muestra el resultado de la evaluación."""
        print("\n" + "="*60)
        print("📊 RESULTADO DE LA EVALUACIÓN")
        print("="*60)
        
        datos = resultado['datos_computador']
        
        print(f"\n💻 INFORMACIÓN DEL COMPUTADOR:")
        print(f"  Marca: {datos['nombre_comercial']}")
        print(f"  Modelo: {datos['modelo']}")
        print(f"  Año: {datos['anio']}")
        print(f"  Condición: {datos['condicion']}")
        
        print(f"\n🔧 ESPECIFICACIONES:")
        print(f"  Procesador: {datos['procesador_original']}")
        print(f"  Generación: {datos['generacion_procesador']}")
        print(f"  Score Procesador: {datos['procesador_score']}/5")
        print(f"  RAM: {datos['ram_gb']} GB")
        print(f"  Disco: {datos['capacidad_disco_gb']} GB {'SSD' if datos['es_ssd'] else 'HDD'}")
        print(f"  Gráfica gamer/dedicada alta: {'Sí' if datos.get('grafica_gamer', datos.get('tiene_grafica', 0)) else 'No'}")
        
        print(f"\n💰 RESULTADO FINANCIERO:")
        print(f"  Precio base calculado (valor usado): ${resultado['precio_base']:,.0f}")
        print(f"  Ajuste por condición (valor usado): ${resultado['precio_ajustado_condicion']:,.0f}")
        print(f"  Regla compraventa aplicada: {int(MARKET_RULES.get('factor_compraventa', 1.0)*100)}% del valor usado")
        print(f"  💵 Precio de préstamo (final): ${resultado['precio_final']:,.0f}")
        
        print(f"\n🎯 RESULTADO FINAL:")
        print(f"  💵 Precio de préstamo sugerido: ${resultado['precio_final']:,.0f}")
        print(f"  📈 Categoría: {resultado['categoria'].upper()}")
        print(f"  📅 Fecha de evaluación: {resultado['fecha_evaluacion']}")
        
        print(f"\n📋 RECOMENDACIONES:")
        if resultado['precio_final'] < 200000:
            print("  ⚠️  Precio bajo - Revisar condiciones del préstamo")
        elif resultado['precio_final'] > 800000:
            print("  ✅ Equipo de alto valor - Buena garantía")
        else:
            print("  ✅ Precio estándar - Condiciones normales aplicables")
    
    def guardar_evaluacion(self, resultado: Dict[str, Any]) -> None:
        """Guarda la evaluación en un archivo."""
        try:
            archivo = 'evaluaciones_computadores.json'
            
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
            
            logger.info(f"✅ Evaluación guardada en {archivo}")
            
        except Exception as e:
            logger.error(f"❌ Error al guardar evaluación: {str(e)}")
    
    def generar_reporte(self) -> None:
        """Genera un reporte de todas las evaluaciones."""
        try:
            archivo = 'evaluaciones_computadores.json'
            
            if not os.path.exists(archivo):
                print("❌ No hay evaluaciones guardadas.")
                return
            
            with open(archivo, 'r', encoding='utf-8') as f:
                evaluaciones = json.load(f)
            
            print(f"\n📊 REPORTE DE EVALUACIONES")
            print("="*60)
            print(f"Total de evaluaciones: {len(evaluaciones)}")
            
            if evaluaciones:
                precios = [eval['precio_final'] for eval in evaluaciones]
                print(f"Precio promedio: ${np.mean(precios):,.0f}")
                print(f"Precio mínimo: ${min(precios):,.0f}")
                print(f"Precio máximo: ${max(precios):,.0f}")
                
                # Contar por categoría
                categorias = {}
                for eval in evaluaciones:
                    cat = eval['categoria']
                    categorias[cat] = categorias.get(cat, 0) + 1
                
                print(f"\nDistribución por categorías:")
                for cat, count in categorias.items():
                    print(f"  {cat.upper()}: {count} equipos")
            
        except Exception as e:
            logger.error(f"❌ Error al generar reporte: {str(e)}")

# =================================================================
# FUNCIÓN PRINCIPAL
# =================================================================

def main():
    """Función principal del sistema interactivo."""
    print("\n" + "="*60)
    print("🏪 SISTEMA AVALUADOR DE COMPUTADORES")
    print("="*60)
    print("Este sistema le ayudará a evaluar el precio de préstamo")
    print("de un computador basándose en sus características.")
    
    avaluador = AvaluadorComputador()
    
    while True:
        print(f"\n📋 MENÚ PRINCIPAL")
        print("-" * 30)
        print("1. 🖥️  Evaluar un computador")
        print("2. 📊 Ver reporte de evaluaciones")
        print("3. ❌ Salir")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == '1':
            resultado = avaluador.evaluar_computador()
            if resultado:
                avaluador.mostrar_resultado(resultado)
                
                # Preguntar si guardar
                guardar = input("\n¿Desea guardar esta evaluación? (s/n): ").strip().lower()
                if guardar == 's':
                    avaluador.guardar_evaluacion(resultado)
                    print("✅ Evaluación guardada exitosamente.")
            
        elif opcion == '2':
            avaluador.generar_reporte()
            
        elif opcion == '3':
            print("👋 ¡Gracias por usar el sistema avaluador!")
            break
            
        else:
            print("❌ Opción inválida. Por favor intente nuevamente.")

# =================================================================
# EJECUCIÓN DEL SCRIPT
# =================================================================

if __name__ == "__main__":
    main()