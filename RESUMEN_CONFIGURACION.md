# Resumen de Configuración Operativa

Este documento resume únicamente la configuración y ajustes que quedaron funcionando en el proyecto.

## Entorno y arranque
- Sistema: Windows (ruta de trabajo `c:\Users\Cristina\OneDrive\Documents\Proyectos_python\Proyecto_Avaluador_Azure`)
- Arranque del servidor: `python -m uvicorn api_server:app --reload --port 8000 --log-level debug`
- Interfaz web: `http://127.0.0.1:8000/`

## Archivos clave
- `api_server.py`: servidor FastAPI y reglas de negocio.
- `vision_integration.py`: análisis de imágenes y cálculo de precio (usa Azure Vision si se definen credenciales).
- `modelo_precio_computador.pkl`: modelo ML cargado para predicción de precio (presente en la raíz del proyecto).
- `dataset_computadores_entrenamiento_LISTO.csv`: dataset base usado para entrenamiento.
- `web/index.html`: formulario y validaciones cliente.
- `config.py`: parámetros de mercado y características del modelo.

## Variables de entorno (si se usa Azure Vision)
- `AZURE_VISION_ENDPOINT`
- `AZURE_VISION_KEY`
Nota: en el servidor se loguea el `endpoint` y la longitud de la `key` para diagnóstico, sin exponerla.

## Políticas aplicadas (backend)
- Procesador:
  - Excluidos: `Pentium`, `Celeron`, `Atom`.
  - Mínimo aceptado: `Core i3` de **10ª generación** o superior. Si no se detecta la generación, se bloquea por falta de acreditación.
- Almacenamiento:
  - `HDD` no permitido. Se **requiere** `SSD`.
- Regla de mínimo económico:
  - Se calcula el precio con ML y se aplica `MARKET_RULES['factor_compraventa']` (por defecto `0.44`).
  - Se garantiza `MARKET_RULES['min_prestamo']` (por defecto `100000`).
  - Si el precio sin mínimo es inferior al mínimo, se marca **bloqueado**.

## Flujo de uso que funciona
1. Subir **al menos 3 imágenes** claras (pantalla, teclado, carcasa/bisagras).
2. Completar especificaciones: marca/modelo, procesador, RAM, capacidad y tipo de disco, gráfica gamer.
3. Enviar y recibir JSON con:
   - `precio_predicho` (numérico y formateado en el cliente).
   - `mensaje_cliente` (bloqueo o avalúo).
   - `advertencias` (motivos de política o entradas no reconocidas).

## Comandos útiles
- Listar paquetes instalados: `python -m pip list --format=columns`
- Exportar dependencias reproducibles: `python -m pip freeze > requirements.txt`
- Ver paquetes de primer nivel: `python -m pip list --not-required --format=columns`

## Dependencias clave (y para qué sirven)
- `fastapi`: framework web para el servidor REST.
- `uvicorn`: servidor ASGI para ejecutar FastAPI.
- `scikit-learn`: predicción de precios con el modelo entrenado.
- `pandas`/`numpy`: manejo y transformación de datos.
- `joblib`: carga/guardado del modelo `.pkl`.
- `python-dotenv`: lectura de variables desde `.env` (si se utiliza).
- `azure-cognitiveservices-vision-computervision`: OCR y señales de imagen (si se definen `AZURE_VISION_*`).

## Notas de validación
- Se agregó validación para que `precio_predicho` sea numérico y logs `[DEBUG] resultado.precio_predicho tipo=... valor=...`.
- El mensaje de **bloqueo** incluye advertencias explícitas del motivo (CPU y/o disco) además del mínimo económico.