# Comandos de Azure: configuración y evaluación

Este documento reúne los comandos necesarios para:
- Configurar Azure Vision (Computer Vision) y obtener credenciales.
- Guardar credenciales en el entorno de Windows y/o `.env` del proyecto.
- Ejecutar la evaluación de ítems con las herramientas del proyecto (CLI, Streamlit y API).

## 1) Prerrequisitos
- Azure CLI instalado:
  - PowerShell: `winget install Microsoft.AzureCLI`
  - Chocolatey: `choco install azure-cli`
- Python y dependencias:
  - `pip install azure-cognitiveservices-vision-computervision`
  - Opcional: `pip install streamlit uvicorn fastapi`

## 2) Inicio de sesión y suscripción
- Iniciar sesión: `az login`
- Ver suscripciones: `az account list -o table`
- Seleccionar suscripción: `az account set --subscription "<NOMBRE_O_ID_DE_SUSCRIPCION>"`

## 3) Grupo de recursos
- Crear grupo: `az group create --name <rg-avaluador> --location eastus`
- Ver grupos: `az group list -o table`

## 4) Crear Azure Vision (Computer Vision)
> Servicio usado por el proyecto vía `AZURE_VISION_ENDPOINT` y `AZURE_VISION_KEY`.
- Crear recurso (SKU pago S1):
  - `az cognitiveservices account create --name <vision-avaluador> --resource-group <rg-avaluador> --kind ComputerVision --sku S1 --location eastus --yes`
- Alternativa SKU gratis (si está disponible en su suscripción):
  - `az cognitiveservices account create --name <vision-avaluador> --resource-group <rg-avaluador> --kind ComputerVision --sku F0 --location eastus --yes`

## 5) Obtener endpoint y claves
- Endpoint (URL):
  - `az cognitiveservices account show --name <vision-avaluador> --resource-group <rg-avaluador> --query properties.endpoint -o tsv`
- Claves:
  - `az cognitiveservices account keys list --name <vision-avaluador> --resource-group <rg-avaluador> -o tsv`
  - Devuelve `key1` y `key2`. Use cualquiera de las dos.

## 6) Guardar credenciales en Windows
- Sesión actual (PowerShell):
  - `$env:AZURE_VISION_ENDPOINT = "https://<vision-avaluador>.cognitiveservices.azure.com/"`
  - `$env:AZURE_VISION_KEY = "<key1_o_key2>"`
- Persistente (nuevas ventanas):
  - `setx AZURE_VISION_ENDPOINT "https://<vision-avaluador>.cognitiveservices.azure.com/"`
  - `setx AZURE_VISION_KEY "<key1_o_key2>"`
  - Abra una nueva ventana de PowerShell para que apliquen.
- Alternativa `.env` en la raíz del proyecto:
  - `Add-Content .env "AZURE_VISION_ENDPOINT=https://<vision-avaluador>.cognitiveservices.azure.com/"`
  - `Add-Content .env "AZURE_VISION_KEY=<key1_o_key2>"`

## 7) Verificación rápida
- Probar que Azure Vision responde (solo credenciales):
  - `az cognitiveservices account show --name <vision-avaluador> --resource-group <rg-avaluador>`
- Ver versión de `az`: `az version`
- Si el proyecto muestra "Sin credenciales Azure Vision", verifique variables `AZURE_VISION_ENDPOINT` y `AZURE_VISION_KEY`.

## 8) Evaluación de ítems (comandos)

### Opción A: Script CLI de demo
Ejecuta la predicción usando fotos y características del equipo.

```powershell
python demo_vision_prediccion.py --imagenes "C:\ruta\frente.jpg" "C:\ruta\dorso.jpg" \
  --es_ssd 1 --ram_gb 16 --capacidad_disco_gb 512 \
  --generacion_procesador 11 --procesador_score 5 --tiene_grafica 1 \
  --modo_cliente 1
```

Notas:
- Requiere `AZURE_VISION_ENDPOINT` y `AZURE_VISION_KEY` definidos.
- Las rutas pueden ser locales o URLs.

### Opción B: Interfaz Streamlit
Interfaz gráfica para evaluar y validar datos.

```powershell
streamlit run streamlit_app.py
```

- En la barra lateral: ajuste `Azure Vision` con su `Endpoint` y `Key` si desea sobreescribir variables.
- Modo "Avalúa mi computador": suba fotos y complete los campos.
- Modo "Crear cuenta"/"Ingresar": flujo de validación de cédula vía OCR.

### Opción C: API FastAPI
Servidor HTTP para consumo programático.

```powershell
python -m uvicorn api_server:app --reload --port 8000 --log-level debug
```

Solicitud de ejemplo con PowerShell (multipart):

```powershell
$form = @{
  imagenes = @(
    @{ filePath = "C:\ruta\frente.jpg"; fileName = "frente.jpg" },
    @{ filePath = "C:\ruta\dorso.jpg";  fileName = "dorso.jpg"  }
  );
  es_ssd = 1;
  capacidad_disco_gb = 512;
  ram_gb = 16;
  generacion_procesador = 11;
  procesador_score = 5;
  tiene_grafica = 1;
}
Invoke-WebRequest -Uri "http://127.0.0.1:8000/avaluo" -Method Post -Form $form
```

Respuesta: JSON con `precio_predicho`, bloqueos por política y mensajes cliente.

## 9) Solución de problemas
- SDK no instalado:
  - `pip install azure-cognitiveservices-vision-computervision`
- Sin credenciales:
  - Verifique `AZURE_VISION_ENDPOINT` y `AZURE_VISION_KEY` en su sesión o `.env`.
- 403/401 al llamar Vision:
  - Use `key1/key2` vigentes y endpoint correcto de su recurso.
- Errores de evaluación:
  - Confirme que las imágenes existen y son válidas (JPG/PNG).
  - Revise el log por mensajes detallados.

## 10) Limpieza
- Eliminar el grupo (borra todo el recurso asociado):
  - `az group delete --name <rg-avaluador> --yes --no-wait`