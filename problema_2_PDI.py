import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import imutils
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
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


DEBUG_DIR = "debug_fallas"
#VERSION QUE ANDA:
def rectify_resistor(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo leer la imagen {image_path}")
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 🔵 Rango HSV mejorado para el azul
    lower_blue = np.array([80, 30, 30])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 🌀 Blur y morfología
    mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=1.5)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # ✨ NUEVO: DILATACIÓN para cerrar huecos por alambres
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    # Contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"[FALLO] Sin contornos en {image_path}")
        save_debug_images(image_path, image, mask)
        return None
     
    max_area = 0
    best_approx = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and area > max_area:
                best_approx = approx
                max_area = area
            elif best_approx is None:
                # Fallback con rectángulo mínimo
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if area > max_area:
                    best_approx = box
                    max_area = area

    if best_approx is None:
        print(f"[FALLO] Sin rectángulo en {image_path}")
        return None

    pts = best_approx.reshape(4, 2).astype("float32")

    # Ordenar los puntos para homografía (top-left, top-right, bottom-right, bottom-left)
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(pts)

    width = 400
    height = 150
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    H, status = cv2.findHomography(rect, dst)
    warped = cv2.warpPerspective(image, H, (width, height))

    return warped
#---
def save_debug_images(image_path, image, mask, contours=None):
    """
    Guarda la máscara y los contornos sobre la imagen original para debug.
    """
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_mask.png"), mask)

    if contours:
        debug = image.copy()
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_contours.png"), debug)
    else:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base_name}_no_contours.png"), image)

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



# Diccionario estándar de colores de resistencias
colores_resistencia = {
    "Negro": ((0, 0, 0), 0),
    "Marrón": ((139, 69, 19), 1),
    "Rojo": ((255, 0, 0), 2),
    "Naranja": ((255, 165, 0), 3),
    "Amarillo": ((255, 255, 0), 4),
    "Verde": ((0, 128, 0), 5),
    "Azul": ((0, 0, 255), 6),
    "Violeta": ((128, 0, 128), 7),
    "Gris": ((128, 128, 128), 8),
    "Blanco": ((255, 255, 255), 9),
    "Dorado": ((212, 175, 55), -1),   # x0.1
    "Plateado": ((192, 192, 192), -2) # x0.01
}

def proyectar_saturacion(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    projection = np.sum(sat, axis=0)
    return projection

def detectar_picos_saturacion(projection, min_distance=15, threshold_rel=0.4):
    height = np.max(projection)
    threshold = height * threshold_rel
    peaks, _ = find_peaks(projection, distance=min_distance, height=threshold)
    return peaks

def extraer_color_por_cluster(img, col_x, ancho=10, k=2):
    h, w, _ = img.shape
    x_start = max(0, col_x - ancho)
    x_end = min(w, col_x + ancho)
    region = img[:, x_start:x_end].reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(region)
    counts = np.bincount(labels)
    main_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int)

    return tuple(main_color)

def distancia_rgb(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def clasificar_color(detectado, tabla_colores, umbral=60):
    min_dist = float('inf')
    mejor_nombre = "Desconocido"
    for nombre, (ref_rgb, _) in tabla_colores.items():
        dist = distancia_rgb(detectado, ref_rgb)
        if dist < min_dist:
            min_dist = dist
            mejor_nombre = nombre
    return mejor_nombre if min_dist < umbral else "Desconocido"

def detectar_bandas_y_valor(img):
    projection = proyectar_saturacion(img)
    peaks = detectar_picos_saturacion(projection)

    colores_detectados = []
    for p in peaks:
        color = extraer_color_por_cluster(img, p)
        nombre_color = clasificar_color(color, colores_resistencia)
        if nombre_color != "Desconocido" and nombre_color not in colores_detectados:
            colores_detectados.append(nombre_color)
        if len(colores_detectados) == 3:
            break

    if len(colores_detectados) < 3:
        return colores_detectados, None

    # Calcular el valor de la resistencia
    d1 = colores_resistencia[colores_detectados[0]][1]
    d2 = colores_resistencia[colores_detectados[1]][1]
    multiplicador = colores_resistencia[colores_detectados[2]][1]
    valor = (10 * d1 + d2) * (10 ** multiplicador)
    return colores_detectados, valor


def procesar_resistencias_en_carpeta(carpeta_imgs):
    resultados = {}
    for nombre_archivo in sorted(os.listdir(carpeta_imgs)):
        if nombre_archivo.endswith('_a_out.jpg'):
            path = os.path.join(carpeta_imgs, nombre_archivo)
            img = cv2.imread(path)
            if img is None:
                continue
            bandas, valor = detectar_bandas_y_valor(img)
            resultados[nombre_archivo] = (bandas, valor)
    return resultados



if __name__ == '__main__':
    main()
    carpeta = r"D:\TUIA\PROCESAMIENTO IMAGEN\TP2_PDI\TP2_PDI_CICORIA_RICCI\Resistencias_out"
    resultados = procesar_resistencias_en_carpeta(carpeta)

    for archivo, (bandas, valor) in resultados.items():
        if valor is not None:
            print(f"{archivo}: Bandas detectadas = {bandas} → Valor = {valor} Ω")
        else:
            print(f"{archivo}: Bandas detectadas = {bandas} → Valor no determinado")
