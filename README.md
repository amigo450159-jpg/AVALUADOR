# Sistema Avaluador de Computadores para Préstamos

## 📋 Descripción

Sistema completo que evalúa el precio de préstamo de computadores mediante:
- **Evaluación interactiva**: Preguntas guiadas al usuario
- **Machine Learning**: Predicción basada en datos históricos
- **Sistema completo**: Combinación de ambos métodos con análisis comparativo

## 🚀 Características

### ✅ Evaluación Tradicional
- Preguntas interactivas sobre características del computador
- Cálculo basado en reglas y factores de ajuste
- Guardado de evaluaciones en formato JSON
- Reportes de evaluaciones realizadas

### 🤖 Machine Learning
- Modelo predictivo entrenado con datos históricos
- Predicción de precios basada en características técnicas
- Comparación entre métodos tradicional y ML
- Análisis de importancia de características

### 📊 Sistema Completo
- Integración de ambos métodos
- Recomendación automática del mejor precio
- Análisis de diferencias entre métodos
- Reportes estadísticos completos

## 📁 Estructura del Proyecto

```
Proyecto_Avaluador_Azure/
├── main.py                          # Punto de entrada principal
├── sistema_completo.py              # Sistema integrado completo
├── avaluador_interactivo.py       # Evaluación tradicional con preguntas
├── modelo_prediccion.py            # Modelo de machine learning
├── config.py                        # Configuración del sistema
├── avaluador.py                     # Procesamiento de datos original
├── dataset_computadores_entrenamiento_LISTO.csv  # Datos de entrenamiento
├── evaluaciones_computadores.json   # Evaluaciones tradicionales guardadas
├── evaluaciones_completas.json      # Evaluaciones con ML guardadas
└── README.md                        # Este archivo
```

## 🔧 Instalación

### Requisitos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instalación de dependencias
```bash
pip install pandas numpy scikit-learn joblib
```

## 🎯 Uso del Sistema

### Opción 1: Sistema Completo (Recomendado)
```bash
python main.py
```
Este comando inicia el sistema completo con todas las funcionalidades.

### Opción 2: Modos Específicos
```bash
# Solo evaluación tradicional
python main.py -t

# Solo machine learning (si está disponible)
python main.py -m

# Generar reporte de evaluaciones
python main.py -r

# Mostrar ayuda
python main.py -h
```

## 📊 Flujo de Trabajo

### 1. Primera Ejecución
1. Ejecute `python main.py`
2. El sistema verificará las dependencias
3. Si existe `dataset_computadores_entrenamiento_LISTO.csv`, entrenará el modelo ML
4. Se mostrará el menú principal

### 2. Evaluación de un Computador
1. Seleccione "Evaluar computador (completo)"
2. Responda las preguntas sobre:
   - Tipo de computador (laptop/desktop)
   - Marca y modelo
   - Especificaciones técnicas (RAM, disco, procesador)
   - Condición física
   - Antigüedad
3. El sistema calculará el precio con ambos métodos
4. Se mostrará la comparación y recomendación
5. Podrá guardar la evaluación

### 3. Análisis de Resultados
- El sistema muestra precios calculados por ambos métodos
- Recomienda el precio final basándose en la diferencia
- Explica la razón de la recomendación
- Permite guardar y comparar evaluaciones

## 📈 Características Técnicas

### Datos de Entrada
- **Marca**: Apple, Dell, HP, Lenovo, Asus, Acer, etc.
- **Tipo**: Laptop o Desktop
- **RAM**: Cantidad de memoria en GB
- **Disco**: Capacidad y tipo (HDD/SSD)
- **Procesador**: Modelo y generación
- **Gráficos**: Tarjeta gráfica integrada/dedicada
- **Condición**: Excelente, buena, regular, mala
- **Antigüedad**: Años desde fabricación

### Factores de Ajuste
- **Marca**: Scores de 1-5 según reputación
- **Condición**: Multiplicadores de 0.6 a 1.2
- **Antigüedad**: Depreciación por años
- **Componentes**: Bonificadores por SSD, gráfica dedicada, RAM alta

### Modelo ML
- **Algoritmo**: Random Forest o Linear Regression
- **Características**: 7 atributos técnicos
- **Precisión**: Variable según calidad de datos
- **Actualización**: Se reentrena con nuevos datos

## 💾 Almacenamiento de Datos

### Evaluaciones Tradicionales
Archivo: `evaluaciones_computadores.json`
- Información del computador
- Precio calculado
- Fecha de evaluación

### Evaluaciones Completas
Archivo: `evaluaciones_completas.json`
- Datos del computador
- Resultados de ambos métodos
- Comparación y recomendación
- Fecha de evaluación

## 📊 Reportes

### Estadísticas Disponibles
- Número total de evaluaciones
- Precio promedio
- Rango de precios
- Comparación entre métodos
- Tendencias por marca/tipo

### Generación de Reportes
```bash
python main.py -r
```

## 🔍 Solución de Problemas

### Error: "Faltan dependencias"
```bash
pip install pandas numpy scikit-learn joblib
```

### Error: "Dataset no encontrado"
- Asegúrese de que `dataset_computadores_entrenamiento_LISTO.csv` exista
- El sistema puede funcionar sin ML, pero con funcionalidad limitada

### Error: "Modelo ML no disponible"
- El modelo se entrena automáticamente si hay datos
- Verifique que el archivo CSV tenga el formato correcto

### Error en la ejecución
- Verifique que todos los archivos `.py` estén presentes
- Asegúrese de usar Python 3.7+
- Revise los logs si están habilitados

## 🔧 Configuración

### Modificar Precios Base
Edite `config.py`:
```python
PRECIOS_BASE = {
    'laptop': {'bajo': 200, 'medio': 500, 'alto': 1000},
    'desktop': {'bajo': 150, 'medio': 400, 'alto': 800}
}
```

### Ajustar Factores
Edite `config.py`:
```python
FACTORES_AJUSTE = {
    'condicion': {'excelente': 1.2, 'buena': 1.0, 'regular': 0.8, 'mala': 0.6},
    'antiguedad': {'0-1': 1.0, '2-3': 0.9, '4-5': 0.7, '6+': 0.5}
}
```

### Configurar Modelo ML
Edite `config.py`:
```python
MODELO_ML = {
    'tipo_modelo': 'random_forest',
    'test_size': 0.2,
    'n_estimators': 100
}
```

## 🤝 Contribuciones

Para mejorar el sistema:
1. Entrene el modelo con más datos históricos
2. Ajuste los factores según su mercado local
3. Agregue nuevas características al modelo
4. Implemente nuevos algoritmos de ML

## 📞 Soporte

Si encuentra problemas:
1. Verifique esta documentación
2. Revise los archivos de log si existen
3. Asegúrese de tener todas las dependencias
4. Verifique el formato de los archivos de datos

## 📄 Licencia

Este sistema fue desarrollado para uso educativo y comercial.
Ajuste los parámetros según sus necesidades específicas.