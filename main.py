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

def detectar_longitud_del_pie(imagen_np):
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    borrosa = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(borrosa, 50, 150)

    contornos = cv2.findContours(bordes.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = imutils.grab_contours(contornos)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    if len(contornos) < 2:
        return None, "No se detectaron suficientes contornos (hoja y pie)."

    hoja = None
    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimetro, True)
        if len(approx) == 4:
            hoja = approx
            break

    if hoja is None:
        return None, "No se detectó una hoja A4 válida en la imagen."

    x, y, w, h = cv2.boundingRect(hoja)
    altura_px = max(w, h)
    escala_px_por_cm = altura_px / 29.7

    pie_contorno = contornos[1]
    x, y, w, h = cv2.boundingRect(pie_contorno)
    largo_px = max(w, h)
    largo_cm = largo_px / escala_px_por_cm
    largo_cm = round(largo_cm, 1)

    if largo_cm < 15 or largo_cm > 35:
        return None, f"Largo fuera de rango: {largo_cm} cm. Asegúrate de usar una hoja A4 y una foto tomada desde arriba."

    return largo_cm, None

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
    imagen_np = np.array(img)

    largo_cm, error = detectar_longitud_del_pie(imagen_np)
    if error:
        return {"error": error}

    talla = convertir_a_talla(largo_cm)
    return {"length_cm": largo_cm, "size": talla}
