import os
from typing import List, Dict, Optional, Tuple

try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
    from msrest.authentication import CognitiveServicesCredentials
    _AZURE_VISION_AVAILABLE = True
except Exception:
    # El SDK no está instalado; el módulo seguirá cargando para permitir mensajes claros.
    _AZURE_VISION_AVAILABLE = False


class AzureVisionClient:
    """
    Cliente sencillo para Azure Computer Vision: detecta marcas (logos) y extrae texto (OCR).
    - Usa variables de entorno `AZURE_VISION_ENDPOINT` y `AZURE_VISION_KEY`.
    - Si el SDK no está instalado, lanza un error instructivo al primer uso.
    """

    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None):
        def _clean(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            value = value.strip()
            value = value.strip("`")
            value = value.strip("'")
            value = value.strip('"')
            return value

        endpoint = _clean(endpoint or os.getenv("AZURE_VISION_ENDPOINT"))
        key = _clean(key or os.getenv("AZURE_VISION_KEY"))
        self.endpoint = endpoint
        self.key = key

        if not _AZURE_VISION_AVAILABLE:
            raise RuntimeError(
                "El SDK de Azure Vision no está instalado. Ejecuta: pip install azure-cognitiveservices-vision-computervision"
            )

        if not endpoint or not key:
            raise ValueError(
                "Faltan o son inválidas las credenciales de Azure Vision. Define AZURE_VISION_ENDPOINT y AZURE_VISION_KEY (sin backticks ni espacios)."
            )

        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    def analizar_imagen(self, image_path_or_url: str) -> Dict:
        """Analiza una imagen: detecta marcas, objetos y extrae texto.
        Acepta ruta local o URL.
        """
        is_url = image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://")

        visual_features = [
            VisualFeatureTypes.brands,
            VisualFeatureTypes.objects,
            VisualFeatureTypes.tags,
        ]

        if is_url:
            analysis = self.client.analyze_image(image_path_or_url, visual_features=visual_features)
            read_operation = self.client.read(image_path_or_url, raw=True)
        else:
            with open(image_path_or_url, "rb") as f:
                analysis = self.client.analyze_image_in_stream(f, visual_features=visual_features)
            with open(image_path_or_url, "rb") as f:
                read_operation = self.client.read_in_stream(f, raw=True)

        # OCR requiere consultar el resultado por Operation-Location y esperar a que termine
        operation_location = read_operation.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        import time
        max_tries = 15  # ~15 segundos máx.
        read_result = None
        for _ in range(max_tries):
            read_result = self.client.get_read_result(operation_id)
            status = getattr(read_result, "status", None)
            if status == "succeeded":
                break
            if status == "failed":
                break
            time.sleep(1.0)

        lines: List[str] = []
        if getattr(read_result, "status", None) == "succeeded":
            for page in read_result.analyze_result.read_results:
                for line in page.lines:
                    lines.append(line.text)

        brands = [b.name for b in (analysis.brands or [])]
        objects = [o.object_property for o in (analysis.objects or [])]
        tags = [t.name for t in (analysis.tags or [])]

        return {
            "brands": brands,
            "objects": objects,
            "tags": tags,
            "ocr_lines": lines,
        }


def inferir_marca(prefer_brands: List[str], ocr_lines: List[str]) -> Optional[str]:
    """Elige la mejor marca a partir de detección de logos y OCR."""
    known_brands = [
        "acer",
        "apple",
        "asus",
        "dell",
        "hp",
        "huawei",
        "lenovo",
        "microsoft",
        "msi",
        "samsung",
    ]

    # Si Vision detectó marcas, usa la primera
    for b in prefer_brands:
        if b:
            return b

    # Si no, intenta encontrar la marca en el texto OCR
    text_all = " ".join(ocr_lines).lower()
    # Sub-marca: HP Victus
    if "victus" in text_all:
        return "HP"
    for b in known_brands:
        if b in text_all:
            return b.capitalize()

    return None


def extraer_indicios_especificaciones(ocr_lines: List[str]) -> Dict[str, Optional[int]]:
    """
    Heurísticas simples para extraer RAM y almacenamiento desde el texto OCR de stickers.
    Devuelve posibles valores, o None si no se encuentran.
    """
    text = " ".join(ocr_lines).lower()

    def _buscar_num_unidad(patterns: List[Tuple[str, int]]) -> Optional[int]:
        for token, multiplier in patterns:
            if token in text:
                # Buscar el primer número antes del token
                import re
                m = re.search(r"(\d{1,3})\s*" + token, text)
                if m:
                    return int(m.group(1)) * multiplier
        return None

    ram_gb = _buscar_num_unidad([
        ("gb ram", 1),
        ("ram gb", 1),
        ("ram", 1),
    ])

    # SSD/HDD capacidad
    capacidad_gb = _buscar_num_unidad([
        ("gb ssd", 1),
        ("gb hdd", 1),
        ("gb", 1),
        ("tb", 1024),
    ])

    # Indicios de generación del procesador (e.g., i5-1135G7 => gen 11)
    generacion = None
    try:
        import re
        m = re.search(r"i[3579]-?(\d{2})(\d{2,3})", text)  # i7-1165G7, i5 1135G7
        if m:
            g = int(m.group(1))
            if 1 < g < 20:
                generacion = g
    except Exception:
        pass

    # Detección simple de GPU gamer en el texto (RTX/GTX/RX/Radeon/GeForce)
    gpu_model_detectado = None
    grafica_gamer_detectada = 0
    try:
        import re
        if any(k in text for k in ["rtx", "gtx", "geforce", "radeon", " rx ", " rx", "rx "]):
            grafica_gamer_detectada = 1
            # Intentar extraer modelo (p.ej., RTX 3050, GTX 1650, RX 6600)
            m = (
                re.search(r"rtx\s*(\d{3,4})", text) or
                re.search(r"gtx\s*(\d{3,4})", text) or
                re.search(r"rx\s*(\d{3,4})", text) or
                re.search(r"radeon\s*(rx\s*\d{3,4}|\d{3,4})", text) or
                re.search(r"geforce\s*(rtx|gtx)\s*(\d{3,4})", text)
            )
            if m:
                gpu_model_detectado = m.group(0).upper()
        # Penalizar caso integrado (no gamer)
        if any(k in text for k in ["intel hd", "intel uhd", "intel iris"]):
            grafica_gamer_detectada = 0
            if gpu_model_detectado is None:
                gpu_model_detectado = "Intel integrado"
    except Exception:
        pass

    return {
        "ram_gb": ram_gb,
        "capacidad_disco_gb": capacidad_gb,
        "generacion_procesador": generacion,
        "grafica_gamer_detectada": grafica_gamer_detectada,
        "gpu_model_detectado": gpu_model_detectado,
    }


def inferir_danios(tags: List[str], objects: List[str], ocr_lines: List[str]) -> Dict:
    """
    Heurísticas sencillas basadas en etiquetas y objetos detectados para inferir daños físicos.
    Aplica un factor de penalización acumulado y devuelve motivos.
    """
    # Usar principalmente OCR para evitar falsos positivos por etiquetas genéricas
    import re
    import unicodedata

    def _normalize(s: str) -> str:
        # Minúsculas y sin acentos para comparación robusta
        s = unicodedata.normalize('NFD', s)
        s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
        return s.lower()

    def contains_word(haystack: str, needle: str) -> bool:
        # Coincidencia por palabra completa (evita que 'dent' coincida con 'identificador')
        h = _normalize(haystack)
        n = _normalize(needle)
        # Permitir espacios como separadores flexibles
        n_pattern = re.escape(n).replace("\\ ", r"\\s+")
        return re.search(r"\b" + n_pattern + r"\b", h) is not None

    text_ocr = " ".join(ocr_lines or [])
    text_tags = " ".join(tags or [])
    text_objs = " ".join(objects or [])

    # Palabras clave de daños (inglés/español)
    keywords = {
        "pantalla_quebrada": ["crack", "cracked", "shattered", "screen crack", "pantalla rota", "pantalla quebrada"],
        # Para 'carcasa dañada' exigimos co-ocurrencia con términos de carcasa/case/chassis
        "carcasa_danada": ["broken", "chipped", "dent", "bent", "abollado", "carcasa rota", "carcasa dañada"],
        "rayones": ["scratch", "scratched", "rayado", "rayones"],
        "bisagra_rota": ["hinge", "hinge broken", "bisagra rota"],
        "teclado_incompleto": ["missing key", "missing keys", "teclas faltantes", "tecla faltante"],
        "manchas": ["stain", "stained", "smudge", "manchas"],
    }

    # Factores por daño (multiplicadores <= 1.0)
    factors = {
        "pantalla_quebrada": 0.60,
        "carcasa_danada": 0.85,
        "rayones": 0.90,
        "bisagra_rota": 0.80,
        "teclado_incompleto": 0.85,
        "manchas": 0.92,
    }

    detected = {}
    motivos = []
    evidencias = []
    factor_total = 1.0
    for key, kws in keywords.items():
        found = False
        origen = None
        matched_kw = None
        muestra = None
        # Primero, buscar en OCR (palabra completa)
        for kw in kws:
            if contains_word(text_ocr, kw):
                found = True
                origen = "ocr"
                matched_kw = kw
                # capturar una línea de muestra donde aparece el keyword
                try:
                    for line in ocr_lines or []:
                        if contains_word(line, kw):
                            muestra = line.strip()
                            break
                except Exception:
                    pass
                break
        # Si no se encontró en OCR, opcionalmente buscar en tags/objects (menos confiable) usando palabra completa
        if not found:
            for kw in kws:
                if contains_word(text_tags, kw):
                    found = True
                    origen = "tags"
                    matched_kw = kw
                    break
                if contains_word(text_objs, kw):
                    found = True
                    origen = "objects"
                    matched_kw = kw
                    break

        # Reglas de co-ocurrencia para evitar falsos positivos
        if found and key == "carcasa_danada":
            carcasa_terms = ["carcasa", "case", "chassis", "casing", "shell"]
            # True si encontramos explícitamente 'carcasa rota/dañada' o si hay daño + término de carcasa
            explicito = any(contains_word(text_ocr, t) for t in ["carcasa rota", "carcasa dañada"]) or \
                        any(contains_word(text_tags, t) for t in ["carcasa rota", "carcasa dañada"]) or \
                        any(contains_word(text_objs, t) for t in ["carcasa rota", "carcasa dañada"]) 
            coocurrencia = any(contains_word(text_ocr, t) for t in carcasa_terms) or \
                          any(contains_word(text_tags, t) for t in carcasa_terms) or \
                          any(contains_word(text_objs, t) for t in carcasa_terms)
            if not (explicito or coocurrencia):
                found = False

        if found and key == "pantalla_quebrada":
            pantalla_terms = ["pantalla", "screen"]
            tiene_superficie = any(contains_word(text_ocr, t) for t in pantalla_terms) or \
                               any(contains_word(text_tags, t) for t in pantalla_terms) or \
                               any(contains_word(text_objs, t) for t in pantalla_terms)
            if not tiene_superficie:
                found = False

        detected[key] = 1 if found else 0
        if found:
            factor_total *= factors.get(key, 1.0)
            motivos.append(key.replace("_", " "))
            evidencias.append({
                "tipo": key,
                "keyword": matched_kw,
                "origen": origen,
                "muestra": muestra,
            })

    # Limitar factor mínimo para evitar anulación total
    factor_total = max(factor_total, 0.50)

    return {
        "danios_detectados": detected,
        "factor_danio": factor_total,
        "motivos": motivos,
        "evidencias": evidencias,
    }


def construir_features_desde_vision(
    vision_result: Dict,
    entrada_usuario: Dict,
    marca_score_map: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    Combina resultados de Vision con datos que el usuario ingresa para formar el vector
    de características compatible con el modelo `ModeloPrecioComputador`.
    """
    # Valores por defecto y mapeos simples
    marca = inferir_marca(vision_result.get("brands", []), vision_result.get("ocr_lines", []))
    indicios = extraer_indicios_especificaciones(vision_result.get("ocr_lines", []))

    marca_score_map = marca_score_map or {
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
    }

    # Determinar flag de gráfica para ML
    grafica_flag = entrada_usuario.get("grafica_gamer")
    if grafica_flag is None:
        grafica_flag = entrada_usuario.get("tiene_grafica")
    if grafica_flag is None:
        grafica_flag = 1 if indicios.get("grafica_gamer_detectada") else 0

    features = {
        "marca_score": entrada_usuario.get("marca_score") or (marca_score_map.get(marca) if marca else 4),
        "es_ssd": entrada_usuario.get("es_ssd"),
        "capacidad_disco_gb": entrada_usuario.get("capacidad_disco_gb") or indicios.get("capacidad_disco_gb") or 256,
        "ram_gb": entrada_usuario.get("ram_gb") or indicios.get("ram_gb") or 8,
        "generacion_procesador": entrada_usuario.get("generacion_procesador") or indicios.get("generacion_procesador") or 10,
        "procesador_score": entrada_usuario.get("procesador_score") or 4,
        # Para ML: solo considerar como 'tiene_grafica' si es gamer/dedicada alta
        "tiene_grafica": 1 if bool(grafica_flag) else 0,
        "_marca_detectada": marca,
        "_ocr_texto": " ".join(vision_result.get("ocr_lines", [])),
        "_gpu_detectada": indicios.get("gpu_model_detectado"),
    }
    return features


def predecir_precio_con_imagenes(
    image_paths: List[str],
    entrada_usuario: Dict,
    archivo_modelo: str = "modelo_precio_computador.pkl",
    vision_endpoint: Optional[str] = None,
    vision_key: Optional[str] = None,
    factor_mercado: Optional[float] = None,
    objetivo_precio: Optional[float] = None,
) -> Dict:
    """
    Orquesta el análisis de múltiples imágenes y ejecuta la predicción con el modelo entrenado.
    """
    from modelo_prediccion import ModeloPrecioComputador
    from config import MARKET_RULES

    # Permite pasar credenciales directamente desde CLI; si no, usa variables de entorno.
    avc = AzureVisionClient(endpoint=vision_endpoint, key=vision_key)

    # Combinar resultados de varias imágenes
    combined = {"brands": [], "objects": [], "tags": [], "ocr_lines": []}
    for p in image_paths:
        r = avc.analizar_imagen(p)
        combined["brands"].extend(r.get("brands", []))
        combined["objects"].extend(r.get("objects", []))
        combined["tags"].extend(r.get("tags", []))
        combined["ocr_lines"].extend(r.get("ocr_lines", []))

    features = construir_features_desde_vision(combined, entrada_usuario)
    danios = inferir_danios(combined.get("tags", []), combined.get("objects", []), combined.get("ocr_lines", []))

    # Crear modelo sin argumentos; ajustar ruta si es distinta del default
    modelo = ModeloPrecioComputador()
    if archivo_modelo:
        try:
            modelo.archivo_modelo = archivo_modelo
        except Exception:
            # fallback silencioso: mantener default
            pass
    modelo.cargar_modelo()
    precio = modelo.predecir_precio(features)
    precio_base_ml = precio
    # Aplicar penalización por daños físicos antes de la regla de mercado
    danio_factor = danios.get("factor_danio") or 1.0
    if precio is not None:
        precio = precio * danio_factor
    precio_post_danio = precio
    # Aplicar regla de compraventa si está configurada para ML en contexto de demo
    # Permitir override por CLI; si viene None usa config
    factor = factor_mercado if (factor_mercado is not None) else MARKET_RULES.get('factor_compraventa', 1.0)
    # Si el usuario define un objetivo explícito, calcular el factor necesario
    if precio is not None and objetivo_precio is not None:
        try:
            calculado = objetivo_precio / precio
            if calculado > 0:
                factor = calculado
        except Exception:
            pass
    minimo = MARKET_RULES.get('min_prestamo', 100000)
    precio_mercado_sin_minimo = None
    if precio is not None and MARKET_RULES.get('aplicar_factor_ml', True):
        precio_mercado_sin_minimo = precio * factor
        precio = max(minimo, precio_mercado_sin_minimo)

    return {
        "precio_predicho": precio,
        "features": features,
        "vision": combined,
        "detalle": {
            "precio_base_ml": precio_base_ml,
            "factor_danio": danios.get("factor_danio"),
            "precio_post_danio": precio_post_danio,
            "danios": danios.get("danios_detectados"),
            "motivos_danios": danios.get("motivos"),
            "factor_compraventa": factor,
            "min_prestamo": minimo,
            "objetivo_precio": objetivo_precio,
            "evidencias_danio": danios.get("evidencias"),
            "precio_mercado_sin_minimo": precio_mercado_sin_minimo,
        },
    }