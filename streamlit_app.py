import os
import tempfile
from typing import List, Dict
import io
from datetime import date
import json
import hashlib
import re
import difflib

import streamlit as st
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from streamlit_drawable_canvas import st_canvas

from avaluador_interactivo import AvaluadorComputador
from sistema_completo import SistemaAvaluadorCompleto
from vision_integration import predecir_precio_con_imagenes, AzureVisionClient
from modelo_prediccion import ModeloPrecioComputador
from config import ARCHIVOS, MARKET_RULES

st.set_page_config(page_title="Avaluador de Computadores", page_icon="🖥️", layout="wide")

st.title("🏪 Sistema Avaluador de Computadores")
st.caption("Evaluación tradicional, ML con imágenes y contrato express")

if "usuario" not in st.session_state:
    st.session_state["usuario"] = None
if "evaluacion" not in st.session_state:
    st.session_state["evaluacion"] = None
if "imagenes_paths" not in st.session_state:
    st.session_state["imagenes_paths"] = []
if "override_modo" not in st.session_state:
    st.session_state["override_modo"] = None

modo_default = ["Inicio", "Ingresar", "Crear cuenta", "Avalúa mi computador", "Evaluación tradicional", "Contrato express", "Reportes"]
modo_sel = st.sidebar.selectbox("Modo", modo_default)
modo = st.session_state.get("override_modo") or modo_sel

# Configuración opcional de Azure Vision (usa variables de entorno si se deja vacío)
st.sidebar.subheader("Azure Vision")
try:
    if not os.getenv("AZURE_VISION_ENDPOINT"):
        os.environ["AZURE_VISION_ENDPOINT"] = st.secrets.get("AZURE_VISION_ENDPOINT", "")
    if not os.getenv("AZURE_VISION_KEY"):
        os.environ["AZURE_VISION_KEY"] = st.secrets.get("AZURE_VISION_KEY", "")
except Exception:
    pass
azure_endpoint = st.sidebar.text_input("Endpoint", value=os.getenv("AZURE_VISION_ENDPOINT") or "")
azure_key = st.sidebar.text_input("Key", value=os.getenv("AZURE_VISION_KEY") or "", type="password")
if azure_endpoint:
    os.environ["AZURE_VISION_ENDPOINT"] = azure_endpoint
if azure_key:
    os.environ["AZURE_VISION_KEY"] = azure_key

def guardar_usuario(data: Dict):
    st.session_state["usuario"] = data

def ir_a(modo: str):
    st.session_state["override_modo"] = modo
    try:
        st.rerun()
    except Exception:
        pass

USERS_FILE = "usuarios.json"

def _load_users() -> List[Dict]:
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_users(users: List[Dict]) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def _find_user(username_or_email: str) -> Dict:
    users = _load_users()
    for u in users:
        if u.get("usuario") == username_or_email or u.get("correo") == username_or_email:
            return u
    return {}

def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9áéíóúñ\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def validar_cedula_via_ocr(frente_path: str, dorso_path: str, nombre: str, cedula: str, fecha_nacimiento_str: str) -> Dict:
    try:
        endpoint = os.getenv("AZURE_VISION_ENDPOINT")
        key = os.getenv("AZURE_VISION_KEY")
        if not endpoint or not key:
            return {"skip": True, "motivo": "Sin credenciales Azure Vision"}
        avc = AzureVisionClient(endpoint=endpoint, key=key)
        lines: List[str] = []
        for p in [frente_path, dorso_path]:
            r = avc.analizar_imagen(p)
            lines.extend(r.get("ocr_lines", []))
        text = " ".join(lines)
        text_norm = _normalize_text(text)
        numeros = re.findall(r"\b\d{7,11}\b", text)
        doc_ocr = None
        if numeros:
            doc_ocr = max(numeros, key=len)
        cedula_num = re.sub(r"\D", "", cedula or "")
        doc_match = (doc_ocr == cedula_num) if doc_ocr else False
        # Fecha: buscar todas y comparar en formato yyyymmdd
        fecha_usr_digits = re.sub(r"\D", "", str(fecha_nacimiento_str))
        try:
            parts = re.findall(r"(\d{4})-(\d{2})-(\d{2})", str(fecha_nacimiento_str))
            if parts:
                y, m2, d = parts[0]
                fecha_usr_digits = f"{y}{m2}{d}"
        except Exception:
            pass
        fechas = []
        for dd, mm, yyyy in re.findall(r"(\d{2})[/-](\d{2})[/-](\d{4})", text_norm):
            fechas.append(f"{yyyy}{mm}{dd}")
        for yyyy, mm, dd in re.findall(r"(\d{4})[/-](\d{2})[/-](\d{2})", text_norm):
            fechas.append(f"{yyyy}{mm}{dd}")
        fecha_ocr = fechas[0] if fechas else None
        date_match = any(f == fecha_usr_digits for f in fechas) if fecha_usr_digits else False
        # Nombre: exigir al menos dos tokens del nombre presentes
        tokens = [t for t in re.split(r"\s+", _normalize_text(nombre)) if len(t) >= 4]
        hits = sum(1 for t in tokens if t in text_norm)
        line_best = max(lines or [""], key=lambda ln: difflib.SequenceMatcher(None, _normalize_text(nombre), _normalize_text(ln)).ratio())
        ratio_best = difflib.SequenceMatcher(None, _normalize_text(nombre), _normalize_text(line_best)).ratio()
        name_match = (hits >= 2) or (ratio_best >= 0.60)
        return {
            "doc_ocr": doc_ocr,
            "doc_match": doc_match,
            "fecha_ocr": fecha_ocr,
            "date_match": date_match,
            "name_similarity": ratio_best,
            "name_match": name_match,
            "lines": lines,
        }
    except Exception as e:
        return {"error": str(e)}

def generar_pdf_contrato(usuario: Dict, resultado: Dict, firma_img: Image.Image, imagenes: List[str], id_fotos: List[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-50, "Contrato Compraventa con Pacto de Retroventa")
    c.setFont("Helvetica", 11)
    c.drawString(40, h-80, f"Nombre: {usuario.get('nombre','')}")
    c.drawString(40, h-100, f"Cédula: {usuario.get('cedula','')}")
    c.drawString(40, h-120, f"Fecha de nacimiento: {usuario.get('fecha_nacimiento','')}")
    c.drawString(300, h-80, f"Correo: {usuario.get('correo','')}")
    c.drawString(300, h-100, f"Celular: {usuario.get('celular','')}")
    c.drawString(300, h-120, f"Ciudad: {usuario.get('ciudad','')}")
    c.drawString(40, h-150, f"Dirección: {usuario.get('direccion','')}")
    precio = resultado.get("precio_predicho") or resultado.get("precio_final")
    c.drawString(40, h-180, f"Avalúo: ${int(precio or 0):,}")
    c.drawString(40, h-200, "Bien: Computador")
    c.rect(40, h-260, 200, 40)
    c.drawString(45, h-245, "Firma vendedor")
    c.rect(300, h-260, 200, 40)
    c.drawString(305, h-245, "El comprador")
    if firma_img:
        sig = ImageReader(firma_img)
        c.drawImage(sig, 45, h-255, width=180, height=30, mask='auto')
        c.drawImage(sig, 305, h-255, width=180, height=30, mask='auto')
    y = h-320
    for p in imagenes[:3]:
        try:
            im = Image.open(p)
            ir = ImageReader(im)
            c.drawImage(ir, 40, y-80, width=120, height=80, preserveAspectRatio=True, mask='auto')
            y -= 90
        except Exception:
            continue
    c.setFont("Helvetica-Bold", 12)
    c.drawString(200, h-320, "Documento de identidad")
    y_id = h-340
    for p in id_fotos[:2]:
        try:
            im = Image.open(p)
            ir = ImageReader(im)
            c.drawImage(ir, 200, y_id-80, width=160, height=80, preserveAspectRatio=True, mask='auto')
            y_id -= 90
        except Exception:
            continue
    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

if modo == "Inicio":
    st.subheader("Bienvenido")
    st.markdown("Seleccione en el menú lateral Crear cuenta para comenzar.")

elif modo == "Crear cuenta":
    with st.form("form_registro"):
        nombre = st.text_input("Nombre")
        cedula = st.text_input("Cédula")
        fecha_nacimiento = st.date_input("Fecha de nacimiento", value=date(1990,1,1), min_value=date(1900,1,1), max_value=date.today())
        direccion = st.text_input("Dirección")
        ciudad = st.text_input("Ciudad")
        correo = st.text_input("Correo electrónico")
        celular = st.text_input("Celular")
        usuario_txt = st.text_input("Usuario")
        contrasena = st.text_input("Contraseña", type="password")
        contrasena2 = st.text_input("Confirmar contraseña", type="password")
        id_frente = st.file_uploader("Foto cédula (frente)", type=["jpg","jpeg","png"], accept_multiple_files=False)
        id_dorso = st.file_uploader("Foto cédula (reverso)", type=["jpg","jpeg","png"], accept_multiple_files=False)
        submit_reg = st.form_submit_button("Crear cuenta")
    if submit_reg:
        if not usuario_txt:
            st.error("Debes definir un usuario")
        elif not contrasena or len(contrasena) < 6:
            st.error("La contraseña debe tener al menos 6 caracteres")
        elif contrasena != contrasena2:
            st.error("Las contraseñas no coinciden")
        elif not id_frente or not id_dorso:
            st.error("Debes anexar las fotos de cédula por ambos lados")
        else:
            users = _load_users()
            if any(u.get("usuario") == usuario_txt for u in users):
                st.error("Usuario ya existe")
            elif any(u.get("correo") == correo for u in users if correo):
                st.error("Correo ya registrado")
            else:
                tmp_front = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(id_frente.name)[1] or ".jpg")
                tmp_front.write(id_frente.read())
                tmp_front.flush(); tmp_front.close()
                tmp_back = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(id_dorso.name)[1] or ".jpg")
                tmp_back.write(id_dorso.read())
                tmp_back.flush(); tmp_back.close()
                val = validar_cedula_via_ocr(tmp_front.name, tmp_back.name, nombre, cedula, str(fecha_nacimiento))
                if val.get("error"):
                    st.error(f"Error validando cédula: {val['error']}")
                    st.stop()
                if not val.get("skip"):
                    if not val.get("doc_match"):
                        st.error("El número de cédula no coincide con el documento escaneado")
                        st.stop()
                    if not (val.get("name_match") or val.get("date_match")):
                        st.error("El nombre o la fecha de nacimiento no coincide con la cédula")
                        st.stop()
                user_record = {
                    "usuario": usuario_txt,
                    "password_hash": _hash_password(contrasena),
                    "nombre": nombre,
                    "cedula": cedula,
                    "fecha_nacimiento": str(fecha_nacimiento),
                    "direccion": direccion,
                    "ciudad": ciudad,
                    "correo": correo,
                    "celular": celular,
                    "id_frente_path": tmp_front.name,
                    "id_dorso_path": tmp_back.name,
                    "validacion_id": val,
                }
                users.append(user_record)
                _save_users(users)
                guardar_usuario(user_record)
                st.success("Cuenta creada")
                st.button("Siguiente: Avalúa mi computador", on_click=lambda: ir_a("Avalúa mi computador"))

elif modo == "Ingresar":
    with st.form("form_login"):
        usuario_o_correo = st.text_input("Nombre de usuario o correo electrónico")
        contrasena_login = st.text_input("Contraseña", type="password")
        recordar = st.checkbox("Recuérdame", value=False)
        submit_login = st.form_submit_button("INGRESAR")
    if submit_login:
        user = _find_user(usuario_o_correo)
        if not user:
            st.error("Usuario no encontrado")
        elif user.get("password_hash") != _hash_password(contrasena_login or ""):
            st.error("Contraseña incorrecta")
        else:
            guardar_usuario(user)
            st.success("Ingreso exitoso")
            st.button("Siguiente: Avalúa mi computador", on_click=lambda: ir_a("Avalúa mi computador"))

elif modo == "Evaluación tradicional":
    av = AvaluadorComputador()
    marcas_disponibles = [info["nombre"] for info in av.MARCA_SCORES.values()] if hasattr(av, "MARCA_SCORES") else [
        "Apple","Dell","Lenovo","HP","Asus","Acer","Samsung","Sony","Victus","Koorui","Windows","Genérico","LG","MSI","Toshiba"
    ]

    with st.form("form_tradicional"):
        marca = st.selectbox("Marca", marcas_disponibles)
        modelo = st.text_input("Modelo")
        anio = st.number_input("Año de fabricación", min_value=2010, max_value=2025, value=2021, step=1)
        tipo_disco = st.radio("Tipo de disco", ["HDD", "SSD"], horizontal=True)
        capacidad_disco = st.number_input("Capacidad de disco (GB)", min_value=128, max_value=4000, value=512)
        ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8)
        procesador = st.text_input("Procesador", value="Intel Core i5")
        tiene_grafica_dedicada = st.radio("¿Tiene tarjeta gráfica dedicada?", ["Sí", "No"], horizontal=True)
        grafica_gamer = False
        if tiene_grafica_dedicada == "Sí":
            grafica_gamer = st.radio("¿Es gamer/alto rendimiento?", ["Sí", "No"], horizontal=True) == "Sí"
        condicion = st.selectbox("Condición", ["Excelente", "Muy buena", "Buena", "Regular", "Mala"])
        funciona_correctamente = st.radio("¿Funciona correctamente?", ["Sí", "No"], horizontal=True)
        guardar = st.checkbox("Guardar evaluación")
        submitted = st.form_submit_button("Calcular precio")

    if submitted:
        if funciona_correctamente != "Sí":
            st.warning("El computador debe estar en buen estado funcional para ser aceptado.")
        else:
            eval_marca = av.evaluar_marca(marca)
            eval_proc = av.evaluar_procesador(procesador)
            datos = {
                **eval_marca,
                "modelo": modelo,
                "anio": int(anio),
                "tipo_disco": tipo_disco,
                "capacidad_disco_gb": float(capacidad_disco),
                "ram_gb": float(ram_gb),
                "grafica_gamer": 1 if grafica_gamer else 0,
                "tiene_grafica": 1 if grafica_gamer else 0,
                "condicion": condicion,
                "funciona_correctamente": True,
                **eval_proc,
            }
            datos["es_ssd"] = 1 if tipo_disco == "SSD" else 0
            precio_base = av.calcular_precio_base(datos)
            precio_aj = av.ajustar_por_condicion(precio_base, condicion)
            precio_usado = av.ajustar_por_antiguedad(precio_aj, int(anio))
            factor = MARKET_RULES.get("factor_compraventa", 1.0)
            minimo = MARKET_RULES.get("min_prestamo", 100000)
            precio_final = max(minimo, precio_usado * factor)
            categoria = "bajo" if precio_final < 300000 else ("medio" if precio_final < 700000 else "alto")
            resultado = {
                "datos_computador": datos,
                "precio_base": precio_base,
                "precio_ajustado_condicion": precio_aj,
                "precio_final": precio_final,
                "categoria": categoria,
            }
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Información del computador")
                st.text(f"Marca: {datos['nombre_comercial']}")
                st.text(f"Modelo: {modelo}")
                st.text(f"Año: {int(anio)}")
                st.text(f"Condición: {condicion}")
                st.text(f"Procesador: {datos['procesador_original']}")
                st.text(f"Generación: {datos['generacion_procesador']}")
                st.text(f"RAM: {int(ram_gb)} GB")
                st.text(f"Disco: {int(capacidad_disco)} GB {'SSD' if datos['es_ssd'] else 'HDD'}")
                st.text(f"Gráfica gamer/dedicada alta: {'Sí' if bool(datos.get('grafica_gamer', datos.get('tiene_grafica', 0))) else 'No'}")
            with col2:
                st.subheader("Resultado financiero")
                st.metric("Precio base (valor usado)", f"${precio_base:,.0f}")
                st.metric("Ajuste por condición", f"${precio_aj:,.0f}")
                st.metric("Precio de préstamo", f"${precio_final:,.0f}")
                st.text(f"Categoría: {categoria.upper()}")
            if guardar:
                av.guardar_evaluacion({**resultado, "fecha_evaluacion": None})
                st.success(f"Evaluación guardada en {ARCHIVOS.get('evaluaciones_tradicionales','evaluaciones_computadores.json')}")
            st.session_state["evaluacion"] = {"precio_final": precio_final, "datos": datos}
            st.button("¿Desea realizar su contrato express?", on_click=lambda: st.session_state.update({"override_modo": "Contrato express"}))

elif modo == "Avalúa mi computador":
    st.info("Requiere modelo entrenado y Azure Vision configurado. Opcionalmente puede funcionar solo con entrada manual.")
    with st.form("form_ml"):
        imagenes = st.file_uploader("Imágenes del equipo", type=["jpg","jpeg","png"], accept_multiple_files=True)
        es_ssd = st.radio("¿Tiene SSD?", ["Sí","No"], horizontal=True) == "Sí"
        capacidad_disco = st.number_input("Capacidad de disco (GB)", min_value=128, max_value=4000, value=512)
        ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=64, value=16)
        generacion = st.number_input("Generación del procesador", min_value=1, max_value=20, value=11)
        proc_score = st.slider("Score del procesador", min_value=1, max_value=5, value=4)
        tiene_grafica = st.radio("¿GPU gamer/dedicada alta?", ["Sí","No"], horizontal=True) == "Sí"
        objetivo_precio = st.number_input("Objetivo de precio (opcional)", min_value=0, value=0)
        submitted_ml = st.form_submit_button("Predecir precio")
    if submitted_ml:
        entrada_usuario: Dict = {
            "es_ssd": 1 if es_ssd else 0,
            "capacidad_disco_gb": int(capacidad_disco),
            "ram_gb": int(ram_gb),
            "generacion_procesador": int(generacion),
            "procesador_score": int(proc_score),
            "tiene_grafica": 1 if tiene_grafica else 0,
        }
        rutas: List[str] = []
        if imagenes:
            for img in imagenes:
                suffix = os.path.splitext(img.name)[1] or ".jpg"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(img.read())
                tmp.flush()
                tmp.close()
                rutas.append(tmp.name)
        try:
            if rutas:
                resultado = predecir_precio_con_imagenes(
                    rutas,
                    entrada_usuario,
                    archivo_modelo=ARCHIVOS.get("modelo_ml", "modelo_precio_computador.pkl"),
                    vision_endpoint=(azure_endpoint or None),
                    vision_key=(azure_key or None),
                    factor_mercado=None,
                    objetivo_precio=(objetivo_precio if objetivo_precio > 0 else None),
                )
            else:
                modelo = ModeloPrecioComputador()
                modelo.archivo_modelo = ARCHIVOS.get("modelo_ml", "modelo_precio_computador.pkl")
                modelo.cargar_modelo()
                precio_base_ml = modelo.predecir_precio(entrada_usuario)
                minimo = MARKET_RULES.get("min_prestamo", 100000)
                factor = MARKET_RULES.get("factor_compraventa", 1.0)
                precio_final = None
                precio_mercado_sin_minimo = None
                if precio_base_ml is not None:
                    precio_mercado_sin_minimo = precio_base_ml * factor
                    precio_final = max(minimo, precio_mercado_sin_minimo)
                resultado = {
                    "precio_predicho": precio_final,
                    "features": entrada_usuario,
                    "vision": {"brands": [], "objects": [], "tags": [], "ocr_lines": []},
                    "detalle": {
                        "precio_base_ml": precio_base_ml,
                        "factor_danio": 1.0,
                        "precio_post_danio": precio_base_ml,
                        "danios": {},
                        "motivos_danios": [],
                        "factor_compraventa": factor,
                        "min_prestamo": minimo,
                        "objetivo_precio": (objetivo_precio if objetivo_precio > 0 else None),
                        "evidencias_danio": [],
                        "precio_mercado_sin_minimo": precio_mercado_sin_minimo,
                    },
                }
            precio = resultado.get("precio_predicho")
            detalle = resultado.get("detalle", {})
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Precio predicho")
                if precio is not None:
                    st.metric("Precio final", f"${precio:,.0f}")
                pb = detalle.get("precio_base_ml")
                if pb is not None:
                    st.metric("Precio base ML", f"${pb:,.0f}")
                fd = detalle.get("factor_danio")
                if fd is not None:
                    st.text(f"Factor por daños: {fd:.2f}")
                fc = detalle.get("factor_compraventa")
                mp = detalle.get("min_prestamo")
                if fc is not None:
                    st.text(f"Regla de mercado: {int(fc*100)}% mínimo ${mp:,.0f}")
            with col2:
                st.subheader("Características usadas")
                feats = resultado.get("features", {})
                for k, v in feats.items():
                    if not str(k).startswith("_"):
                        st.text(f"{k}: {v}")
                if feats.get("_gpu_detectada"):
                    st.text(f"GPU detectada: {feats.get('_gpu_detectada')}")
            if detalle.get("danios"):
                st.subheader("Daños detectados")
                for dk, dv in detalle["danios"].items():
                    if dv:
                        st.text(dk.replace("_"," "))
            st.session_state["evaluacion"] = {"precio_predicho": precio, "features": resultado.get("features"), "detalle": detalle}
            st.session_state["imagenes_paths"] = rutas
            st.button("¿Desea realizar su contrato express?", on_click=lambda: st.session_state.update({"override_modo": "Contrato express"}))
        except Exception as e:
            st.error(str(e))

elif modo == "Reportes":
    st.subheader("Evaluaciones tradicionales")
    arch_trad = ARCHIVOS.get("evaluaciones_tradicionales", "evaluaciones_computadores.json")
    if os.path.exists(arch_trad):
        import json
        with open(arch_trad, "r", encoding="utf-8") as f:
            tradicionales = json.load(f)
        st.text(f"Total: {len(tradicionales)}")
        if tradicionales:
            precios = [e.get("precio_final", 0) for e in tradicionales]
            st.text(f"Promedio: ${np.mean(precios):,.0f}")
            st.text(f"Rango: ${min(precios):,.0f} - ${max(precios):,.0f}")
    else:
        st.warning("No hay evaluaciones tradicionales guardadas.")
    st.subheader("Evaluaciones completas (ML)")
    arch_comp = ARCHIVOS.get("evaluaciones_completas", "evaluaciones_completas.json")
    if os.path.exists(arch_comp):
        import json
        with open(arch_comp, "r", encoding="utf-8") as f:
            completas = json.load(f)
        st.text(f"Total: {len(completas)}")
        precios_ml = []
        for ev in completas:
            comp = ev.get("comparacion_metodos", {})
            pr = comp.get("precio_recomendado")
            if pr:
                precios_ml.append(pr)
        if precios_ml:
            st.text(f"Promedio ML: ${np.mean(precios_ml):,.0f}")
            st.text(f"Rango ML: ${min(precios_ml):,.0f} - ${max(precios_ml):,.0f}")
    else:
        st.warning("No hay evaluaciones completas guardadas.")

elif modo == "Contrato express":
    usuario = st.session_state.get("usuario")
    evaluacion = st.session_state.get("evaluacion")
    imagenes = st.session_state.get("imagenes_paths") or []
    if not usuario:
        st.warning("Cree su cuenta primero")
    else:
        st.subheader("Datos del usuario")
        st.text(f"Nombre: {usuario.get('nombre','')}")
        st.text(f"Cédula: {usuario.get('cedula','')}")
        st.text(f"Ciudad: {usuario.get('ciudad','')}")
        st.text(f"Correo: {usuario.get('correo','')}")
        st.subheader("Resumen del avalúo")
        precio = None
        if evaluacion:
            precio = evaluacion.get("precio_predicho") or evaluacion.get("precio_final")
            st.metric("Avalúo", f"${int(precio or 0):,}")
        if imagenes:
            st.subheader("Fotos")
            cols = st.columns(3)
            for i, p in enumerate(imagenes[:3]):
                try:
                    cols[i].image(Image.open(p), use_column_width=True)
                except Exception:
                    pass
        st.subheader("Firma virtual")
        canvas_result = st_canvas(stroke_color="#000000", background_color="#FFFFFF", height=200, width=600, drawing_mode="freedraw", key="firma_canvas")
        firma_img = None
        if canvas_result.image_data is not None:
            firma_img = Image.fromarray((canvas_result.image_data).astype("uint8"))
        gen = st.button("Generar contrato PDF")
        if gen:
            id_fotos = []
            if usuario.get("id_frente_path"):
                id_fotos.append(usuario["id_frente_path"])
            if usuario.get("id_dorso_path"):
                id_fotos.append(usuario["id_dorso_path"])
            pdf_bytes = generar_pdf_contrato(usuario, evaluacion or {}, firma_img, imagenes, id_fotos)
            st.download_button("Descargar contrato", data=pdf_bytes, file_name="contrato.pdf", mime="application/pdf")
            st.subheader("Estado")
            st.markdown("🚚 Alguien irá a recoger tu pedido. Está en validación el proceso.")