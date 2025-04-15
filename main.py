from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import cv2
import imutils

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Medidas oficiales de billetes mexicanos (en mm)
BILLETES_MM = {
    "20": (120, 65),
    "50": (127, 66),
    "100": (134, 66),
    "200": (139, 66),
    "500": (146, 66),
    "1000": (153, 66)
}

def identificar_billete(img_np):
    gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    borrosa = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(borrosa, 50, 150)

    contornos = cv2.findContours(bordes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = imutils.grab_contours(contornos)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    for c in contornos:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            largo_px = max(w, h)
            alto_px = min(w, h)
            for valor, (mm_largo, mm_alto) in BILLETES_MM.items():
                ratio_real = mm_largo / mm_alto
                ratio_img = largo_px / alto_px
                if 0.9 * ratio_real <= ratio_img <= 1.1 * ratio_real:
                    return valor, largo_px, mm_largo  # Valor del billete, pixeles, mm
    return None, None, None

@app.post("/api/detectar_billete")
async def detectar_billete(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)

    valor, px, mm = identificar_billete(img_np)
    if not valor:
        return {"error": "No se detectó ningún billete válido."}
    return {"billete": f"${valor}", "pixeles": px, "mm": mm}

def detectar_largo_pie(img_np, escala_px_mm):
    gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    borrosa = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(borrosa, 50, 150)
    contornos = cv2.findContours(bordes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = imutils.grab_contours(contornos)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    if len(contornos) < 2:
        return None

    pie_contorno = contornos[1]
    x, y, w, h = cv2.boundingRect(pie_contorno)
    largo_px = max(w, h)
    largo_mm = largo_px / escala_px_mm
    return round(largo_mm / 10, 1)  # mm a cm

def convertir_a_talla(cm):
    tabla = {
        22: "22 MX", 23: "23 MX", 24: "24 MX",
        25: "25 MX", 26: "26 MX", 27: "27 MX",
        28: "28 MX", 29: "29 MX", 30: "30 MX"
    }
    redondeado = round(cm)
    return tabla.get(redondeado, f"{redondeado} MX")

@app.post("/api/medir")
async def medir(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)

    valor, largo_px, largo_mm = identificar_billete(img_np)
    if not valor:
        return {"error": "No se detectó un billete válido."}

    escala_px_mm = largo_px / largo_mm
    largo_cm = detectar_largo_pie(img_np, escala_px_mm)
    if not largo_cm or largo_cm < 15 or largo_cm > 35:
        return {"error": f"Largo fuera de rango: {largo_cm} cm. Verifica la imagen."}

    talla = convertir_a_talla(largo_cm)
    return {"length_cm": largo_cm, "size": talla, "referencia": f"${valor}"}
