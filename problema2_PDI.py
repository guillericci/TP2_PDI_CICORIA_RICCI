import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import imutils

# Ruta de entrada y salida
INPUT_DIR = r"D:\TUIA\PROCESAMIENTO IMAGEN\TP2_PDI\TP2_PDI_CICORIA_RICCI\Resistencias"
OUTPUT_DIR = r"D:\TUIA\PROCESAMIENTO IMAGEN\TP2_PDI\TP2_PDI_CICORIA_RICCI\Resistencias_out"

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

#img = cv2.imread(r"D:\TUIA\PROCESAMIENTO IMAGEN\TP2_PDI\TP2_PDI_CICORIA_RICCI\Resistencias_out\R6_d_out.jpg")
   # Acá puede verse que OpenCV, por default, carga las imágenes color en formato BGR, no RGB.
#img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#imshow(img_RGB, title="Imagen Original con planos re-acomodados (BGR --> RGB)")
#hsv = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
#imshow(hsv, title="Imagen Original HSV") 

def rectify_resistor(img_path):
    """
    Toma la ruta de una imagen de resistencia en perspectiva y devuelve
    la vista superior recortada del rectángulo azul.
    """
    img = cv2.imread(img_path)
    # Convertir a HSV para segmentar el rectángulo azul
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Definir rango de azul (ajustar según iluminación)
    lower_blue = np.array([90, 50, 50])#[90, 50, 50]
    upper_blue = np.array([130, 255, 255])#[130, 255, 255]
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Encontrar contornos
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # Seleccionar contorno más grande
    c = max(cnts, key=cv2.contourArea)
    # Aproximar polígono a 4 vértices
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2)
    # Ordenar los puntos: tl, tr, br, bl
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # Calcular dimensiones
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    # Transformación de perspectiva
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warp


def process_all_images():
    """
    Procesa todas las imágenes de INPUT_DIR, aplica rectify_resistor y
    guarda las imágenes en OUTPUT_DIR con sufijo '_out'.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for fname in os.listdir(INPUT_DIR):
        if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png') or fname.lower().endswith('.tif')):
            continue
        in_path = os.path.join(INPUT_DIR, fname)
        warp = rectify_resistor(in_path)
        if warp is None:
            print(f"No se detectó rectángulo azul en {fname}")
            continue
        name, ext = os.path.splitext(fname)
        out_name = f"{name}_out{ext}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), warp)
        print(f"Guardada: {out_name}")


def detect_color_bands(img):
    """
    Detecta las 3 bandas de color principales en la vista superior de la resistencia.
    Devuelve una lista con los colores detectados en orden.
    """
    # Implementar segmentación de bandas: por ejemplo, proyección vertical,
    # detección de regiones de color y filtrado por posición.
    raise NotImplementedError


def main():
    # Punto a) y b)
    process_all_images()

    # Punto c), d) y e): detectar bandas y calcular valor
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.lower().endswith('_a_out.png') and not fname.lower().endswith('_a_out.jpg') and not fname.lower().endswith('_a_out.tif'):
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        img = cv2.imread(path)
        bands = detect_color_bands(img)
        print(f"{fname}: {bands}")
        # Aquí calcular valor en Ohm según la codificación de colores

if __name__ == '__main__':
    main()
