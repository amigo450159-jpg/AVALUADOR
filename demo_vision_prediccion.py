"""
Demostración: Predicción de precio combinando Azure Vision (fotos) + entrada del usuario.

Uso:
  python demo_vision_prediccion.py --imagenes ruta1.jpg ruta2.jpg --es_ssd 1 --ram_gb 16 --capacidad_disco_gb 512 --generacion_procesador 11 --procesador_score 5 --tiene_grafica 1

Nota importante:
  "--tiene_grafica" debe ser 1 SOLO si la GPU es gamer/dedicada de alto rendimiento (NVIDIA GTX/RTX, AMD RX). Para gráfica integrada o dedicada normal, use 0.

Requisitos:
  - Variables de entorno: AZURE_VISION_ENDPOINT y AZURE_VISION_KEY
  - Paquetes: azure-cognitiveservices-vision-computervision
  - Modelo entrenado: modelo_precio_computador.pkl en el directorio actual
"""

import argparse
from typing import List, Dict

from vision_integration import predecir_precio_con_imagenes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predicción con fotos + usuario")
    p.add_argument("--imagenes", nargs="+", help="Rutas o URLs de imágenes (mínimo 1)")
    p.add_argument("--es_ssd", type=int, default=1, help="1 si tiene SSD, 0 si no")
    p.add_argument("--capacidad_disco_gb", type=int, default=256)
    p.add_argument("--ram_gb", type=int, default=8)
    p.add_argument("--generacion_procesador", type=int, default=10)
    p.add_argument("--procesador_score", type=int, default=4)
    p.add_argument("--tiene_grafica", type=int, default=0, help="1 solo si es GPU gamer/dedicada (NVIDIA GTX/RTX, AMD RX); 0 si es integrada o dedicada normal")
    p.add_argument("--archivo_modelo", type=str, default="modelo_precio_computador.pkl")
    # Modo de salida y ajuste de mercado
    p.add_argument("--modo_cliente", type=int, default=0, help="1 para mostrar solo el mensaje simple para el cliente")
    p.add_argument("--factor_mercado", type=float, default=None, help="Override del factor de mercado (ej. 1.40 para aumentar 40%)")
    p.add_argument("--objetivo_precio", type=float, default=None, help="Objetivo de precio final (ej. 300000). Calcula factor automático.")
    # Credenciales opcionales para Azure Vision (si no se usan variables de entorno)
    p.add_argument("--vision_endpoint", type=str, default=None, help="Endpoint de Azure Vision")
    p.add_argument("--vision_key", type=str, default=None, help="Clave de Azure Vision")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.imagenes:
        raise SystemExit("Debe proporcionar al menos una imagen con --imagenes")

    entrada_usuario: Dict = {
        "es_ssd": args.es_ssd,
        "capacidad_disco_gb": args.capacidad_disco_gb,
        "ram_gb": args.ram_gb,
        "generacion_procesador": args.generacion_procesador,
        "procesador_score": args.procesador_score,
        "tiene_grafica": args.tiene_grafica,
    }

    resultado = predecir_precio_con_imagenes(
        args.imagenes,
        entrada_usuario,
        archivo_modelo=args.archivo_modelo,
        vision_endpoint=args.vision_endpoint,
        vision_key=args.vision_key,
        factor_mercado=args.factor_mercado,
        objetivo_precio=args.objetivo_precio,
    )
    # Modo cliente: salida mínima con validación de mínimo
    if args.modo_cliente:
        detalle = resultado.get("detalle", {})
        minimo = detalle.get("min_prestamo")
        sin_minimo = detalle.get("precio_mercado_sin_minimo")
        precio_final = resultado['precio_predicho']

        # Si el cálculo de mercado queda por debajo del mínimo, rechazar contrato
        if sin_minimo is not None and minimo is not None and sin_minimo < minimo:
            print("\nNo es posible realizar el contrato.")
            print("Para mayor información, comuníquese a la línea de atención o acérquese a una sede física.")
            return

        # Mensaje simple para el cliente
        print("\nTu avalúo del pc enviado es de ${:,.0f}.".format(precio_final))
        print("¿Deseas continuar con el contrato? (S/N)")
        return

    # Modo desarrollador/terminal: salida detallada
    print("\n=== RESULTADO DE PREDICCIÓN ===")
    print(f"Precio predicho: ${resultado['precio_predicho']:,.0f}")
    detalle = resultado.get("detalle", {})
    if detalle:
        pb = detalle.get("precio_base_ml")
        fd = detalle.get("factor_danio")
        fc = detalle.get("factor_compraventa")
        mp = detalle.get("min_prestamo")
        motivos = detalle.get("motivos_danios") or []
        if pb is not None:
            print(f"Precio base (ML): ${pb:,.0f}")
        if fd is not None:
            etiqueta_motivos = ", ".join(motivos) if motivos else "sin daños detectados"
            print(f"Factor por daños físicos: {fd:.2f} ({etiqueta_motivos})")
        if fc is not None:
            porcentaje = int((fc or 1.0) * 100)
            if mp is not None:
                print(f"Regla de mercado aplicada: {porcentaje}% del valor, mínimo ${mp:,.0f}")
            else:
                print(f"Regla de mercado aplicada: {porcentaje}% del valor")
        if detalle.get("objetivo_precio"):
            print(f"Objetivo solicitado: ${detalle['objetivo_precio']:,.0f}")
        # Mostrar evidencias de daño (si existen) para auditoría técnica
        evidencias = detalle.get("evidencias_danio") or []
        if evidencias:
            print("Evidencias de daño (fuente y muestra):")
            for ev in evidencias[:3]:
                tipo = ev.get("tipo","?").replace("_"," ")
                origen = ev.get("origen","?")
                muestra = ev.get("muestra") or "(sin línea OCR)"
                print(f" - {tipo} | origen: {origen} | muestra: {muestra}")
    print("Características usadas:")
    for k, v in resultado["features"].items():
        if not k.startswith("_"):
            print(f" - {k}: {v}")
    print("\nInformación detectada por Vision:")
    print(f" - Marcas: {resultado['vision'].get('brands')}")
    print(f" - OCR (muestras): {resultado['vision'].get('ocr_lines')[:5]}")
    gpu_detectada = resultado["features"].get("_gpu_detectada")
    if gpu_detectada:
        print(f" - GPU detectada (OCR): {gpu_detectada}")
    danios = detalle.get("danios") if detalle else None
    if danios:
        print(" - Daños detectados:")
        for dk, dv in danios.items():
            if dv:
                print(f"   * {dk.replace('_',' ')}")


if __name__ == "__main__":
    main()