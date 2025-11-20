"""
Servidor web (FastAPI) para integrar la evaluación con Azure Vision.

- Endpoint POST /avaluo: recibe imágenes (multipart) y especificaciones.
- Usa vision_integration.predecir_precio_con_imagenes para calcular el avalúo.
- Devuelve JSON en modo cliente, bloqueando si está por debajo del mínimo.
- Sirve una página HTML simple para subir fotos y ver el resultado.
"""

import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from vision_integration import predecir_precio_con_imagenes
from config import ARCHIVOS, AZURE_VISION_ENDPOINT, AZURE_VISION_KEY


app = FastAPI(title="Avaluador Azure Vision")

# Permitir CORS básico si quieres consumir desde otra página
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> FileResponse:
    """Devuelve la página HTML básica para probar el flujo."""
    return FileResponse(os.path.join("web", "index.html"))


def _infer_cpu_score(name: str) -> int:
    """Inferir puntuación del procesador desde texto libre (escala 1-9 aprox.)."""
    if not name:
        return 4
    s = name.lower()
    def has(*keys):
        return all(k in s for k in keys)
    # Marcas básicas
    if "celeron" in s:
        return 2
    if "athlon" in s:
        return 3
    if "pentium" in s:
        return 3
    if "silver" in s and "pentium" in s:
        return 3
    if "core i3" in s:
        return 4
    if "core i5" in s:
        return 6
    if "core i7" in s:
        return 8
    if "core i9" in s:
        return 9
    if "ryzen 3" in s:
        return 4
    if "ryzen 5" in s:
        return 6
    if "ryzen 7" in s:
        return 8
    if "ryzen 9" in s:
        return 9
    if "xeon" in s:
        return 7
    if "apple m1" in s:
        return 7
    if "apple m2" in s:
        return 8
    if "apple m3" in s:
        return 9
    return 4

def _cpu_is_known(name: str) -> bool:
    """Heurística: se considera conocido si coincide con familias comunes."""
    s = (name or "").lower()
    comunes = [
        "celeron", "pentium", "core i3", "core i5", "core i7", "core i9",
        "ryzen 3", "ryzen 5", "ryzen 7", "ryzen 9", "xeon",
        "apple m1", "apple m2", "apple m3"
    ]
    return any(k in s for k in comunes)

def _cpu_excluido_politica(name: str) -> tuple[bool, str]:
    """Regla de negocio: excluir familias no aceptadas y exigir mínimo i3 10ª gen.

    - Excluye explícitamente Pentium/Celeron/Atom.
    - Para Core i3, exige generación >= 10; si no se puede determinar, se bloquea por falta de acreditación.
    """
    s = (name or "").lower()
    if not s:
        return False, ""
    if "pentium" in s or "celeron" in s or "atom" in s:
        return True, "Procesador excluido por política (Pentium/Celeron/Atom). Mínimo Core i3 10ª gen en adelante."
    # Exigir mínimo i3 10ª gen
    try:
        gen = _infer_generation(name or "")
    except Exception:
        gen = None
    if ("core i3" in s) or (" i3" in s) or s.startswith("i3"):
        if gen is None:
            return True, "Core i3 sin generación acreditada. Requisito mínimo: i3 10ª gen en adelante."
        if isinstance(gen, int) and gen < 10:
            return True, "Core i3 anterior a 10ª gen no aceptado. Requisito mínimo: i3 10ª gen en adelante."
    return False, ""

def _disco_excluido_politica(es_ssd_val: int | bool) -> tuple[bool, str]:
    """Regla de negocio para almacenamiento: exigir SSD, excluir HDD.

    Si `es_ssd_val` es 0/False, se bloquea y se informa el motivo.
    """
    try:
        flag = bool(int(es_ssd_val))
    except Exception:
        flag = bool(es_ssd_val)
    if not flag:
        return True, "Disco HDD no permitido por política. Se requiere almacenamiento SSD."
    return False, ""


def _infer_generation(name: str) -> int:
    """Intentar inferir generación del procesador desde el texto.
    - Intel: i*-11xxx -> 11, i*-12xxx -> 12, etc.
    - Ryzen: 5600/4600 ~ 5/4 (aprox.). Si no se detecta, default 10.
    """
    import re
    if not name:
        return 10
    s = name.lower()
    # Intel Core i*-11, i*-12, i*-13...
    m = re.search(r"core\s*i[3579]\D*(1[1-5])", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Ryzen 5 5600U -> 5
    m = re.search(r"ryzen\s*[3579]?\D*([3-9])\D*5[0-9]{2,3}", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return 10


def _infer_brand(text: str) -> Optional[str]:
    brands = [
        "lenovo","hp","dell","asus","acer","msi","apple","huawei","samsung","microsoft","razer","alienware"
    ]
    s = (text or "").lower()
    for b in brands:
        if b in s:
            return b.capitalize()
    return None


@app.post("/avaluo")
async def avaluo(
    imagenes: List[UploadFile] = File(..., description="Sube al menos 1 imagen"),
    es_ssd: int = Form(1),
    capacidad_disco_gb: int = Form(256),
    ram_gb: int = Form(8),
    marca_modelo: Optional[str] = Form(None),
    procesador: Optional[str] = Form(None),
    tiene_grafica: int = Form(0),
    factor_mercado: Optional[float] = Form(None),
    objetivo_precio: Optional[float] = Form(None),
):
    """Recibe fotos y especificaciones y devuelve el avalúo en JSON para el cliente."""
    if not imagenes:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Debes subir al menos 1 imagen"})

    # Guardar imágenes temporalmente para pasarlas a vision_integration
    temp_paths: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="avaluo_")
    try:
        # Log básico de credenciales (sin exponer la key completa)
        try:
            print(f"[DEBUG] /avaluo usando Vision endpoint: {AZURE_VISION_ENDPOINT}; key_len: {len(AZURE_VISION_KEY or '')}")
        except Exception:
            pass

        for i, img in enumerate(imagenes):
            # Mantener extensión si existe
            _, ext = os.path.splitext(img.filename or "")
            ext = ext if ext else ".jpg"
            dest = os.path.join(tmpdir, f"img_{i}{ext}")
            with open(dest, "wb") as f:
                content = await img.read()
                f.write(content)
            temp_paths.append(dest)

        # Calcular internamente puntuación/generación si el cliente escribió el procesador
        cpu_score = _infer_cpu_score(procesador or "")
        cpu_gen = _infer_generation(procesador or "")

        # Mapear marca si el cliente escribió Marca/Modelo
        marca_detectada = _infer_brand(marca_modelo or "")
        marca_score_map = {
            "Lenovo": 5,
            "HP": 5,
            "Dell": 5,
            "Asus": 4,
            "Acer": 4,
            "MSI": 4,
            "Apple": 6,
            "Huawei": 4,
            "Samsung": 4,
            "Microsoft": 5,
            "Razer": 5,
            "Alienware": 6,
        }

        entrada_usuario = {
            "es_ssd": es_ssd,
            "capacidad_disco_gb": capacidad_disco_gb,
            "ram_gb": ram_gb,
            "generacion_procesador": cpu_gen,
            "procesador_score": cpu_score,
            "tiene_grafica": tiene_grafica,
            # Permitir que el usuario sobrescriba marca si la escribió
            "marca_score": marca_score_map.get(marca_detectada) if marca_detectada else None,
        }

        try:
            resultado = predecir_precio_con_imagenes(
                temp_paths,
                entrada_usuario,
                archivo_modelo=ARCHIVOS.get("modelo_ml", "modelo_precio_computador.pkl"),
                vision_endpoint=AZURE_VISION_ENDPOINT,
                vision_key=AZURE_VISION_KEY,
                factor_mercado=factor_mercado,
                objetivo_precio=objetivo_precio,
            )
        except Exception as e:
            # Responder con error claro para el frontend
            return JSONResponse(status_code=500, content={
                "ok": False,
                "error": str(e),
                "sugerencias": [
                    "Verifica AZURE_VISION_ENDPOINT y AZURE_VISION_KEY en el entorno",
                    "Asegúrate de tener instalado azure-cognitiveservices-vision-computervision",
                    "Confirma que el archivo del modelo existe (modelo_precio_computador.pkl)",
                ]
            })

        detalle = resultado.get("detalle", {})
        minimo = detalle.get("min_prestamo")
        sin_minimo = detalle.get("precio_mercado_sin_minimo")
        precio_final = resultado.get("precio_predicho")

        # Depuración del precio retornado por la integración
        try:
            print(f"[DEBUG] resultado.precio_predicho tipo={type(precio_final).__name__} valor={precio_final}")
        except Exception:
            pass

        # Validar que precio_final sea un número
        try:
            precio_final_num = float(precio_final)
        except (TypeError, ValueError):
            return JSONResponse(status_code=500, content={
                "ok": False,
                "error": "No se pudo calcular el precio (precio_predicho no numérico)",
                "sugerencias": [
                    "Verifica que las imágenes sean válidas (no vacías, formato JPG/PNG)",
                    "Confirma que el archivo del modelo existe y es accesible",
                    "Revisa las credenciales de Azure Vision y los logs para más detalle",
                ]
            })

        # Mensajes cliente
        # Bloqueo por mínimo económico
        try:
            bloqueado_minimo = bool(
                sin_minimo is not None and minimo is not None and float(sin_minimo) < float(minimo)
            )
        except (TypeError, ValueError):
            bloqueado_minimo = False

        # Bloqueo por política de hardware (procesador)
        bloqueado_cpu, motivo_cpu = _cpu_excluido_politica(procesador or "")

        # Bloqueo por política de almacenamiento (SSD obligatorio)
        bloqueado_disco, motivo_disco = _disco_excluido_politica(es_ssd)

        bloqueado = bool(bloqueado_minimo or bloqueado_cpu or bloqueado_disco)
        if bloqueado:
            mensaje = (
                "No es posible realizar el contrato, para mayor información comunicarse a la línea de atención o a una sede física"
            )
        else:
            mensaje = f"Tu avalúo del pc enviado es de ${int(round(precio_final_num)):,}. ¿Deseas continuar con el contrato?"

        # Advertencias para entradas que no se reconocen claramente
        advertencias = []
        if procesador and not _cpu_is_known(procesador):
            advertencias.append(
                "El procesador ingresado no coincide con familias comunes (Celeron, Pentium, Core i*, Ryzen *). Verifica el nombre."
            )
        if motivo_cpu:
            advertencias.append(motivo_cpu)
        if motivo_disco:
            advertencias.append(motivo_disco)
        if marca_modelo and _infer_brand(marca_modelo or "") is None:
            advertencias.append(
                "No se reconoce la marca en 'Marca / Modelo'. Ingresa HP, Dell, Lenovo, Asus, Acer, etc., o deja el campo vacío."
            )

        return {
            "ok": True,
            "precio_predicho": precio_final,
            "bloqueado_por_minimo": bloqueado,
            "mensaje_cliente": mensaje,
            "advertencias": advertencias,
            # Se devuelve un subconjunto de detalle útil
            "detalle": {
                "precio_base_ml": resultado.get("precio_base_ml"),
                "precio_post_danio": detalle.get("precio_post_danio"),
                "precio_mercado_sin_minimo": sin_minimo,
                "min_prestamo": minimo,
            },
        }
    except Exception as e:
        # Captura amplia para evitar caídas del servidor y devolver JSON
        try:
            err_msg = str(e)
        except Exception:
            err_msg = "Error desconocido en el servidor"
        return JSONResponse(status_code=500, content={
            "ok": False,
            "error": err_msg,
            "sugerencias": [
                "Verifica que subiste al menos 3 imágenes válidas",
                "Confirma AZURE_VISION_ENDPOINT y AZURE_VISION_KEY en el entorno",
                "Revisa los logs del servidor para el traceback",
            ]
        })
    finally:
        # Limpiar temporales
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass