import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Cargo imagen ---
img_path = os.path.join(os.getcwd(), 'placa.png') 
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 

if img is None:
    raise ValueError("La imagen no se pudo cargar. Verifica el path o nombre del archivo.")

plt.figure(), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Imagen Original'), plt.show(block=False)

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
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
# --- CANNY ----
f = img
imshow(f)		 

f_blur = cv2.GaussianBlur(f, ksize=(3, 3), sigmaX=1.5)
plt.figure()
ax = plt.subplot(121)
imshow(f, new_fig=False, title="Imagen Original", ticks=True)
plt.subplot(122, sharex=ax, sharey=ax), imshow(f_blur, new_fig=False, title="Gaussian Blur")
plt.show(block=False)		 

gcan2 = cv2.Canny(f_blur, threshold1=0.3*255, threshold2=0.5*255) #0.4-0.5, 0304

plt.figure()
ax = plt.subplot(221)
imshow(f, new_fig=False, title="Imagen Original")
plt.subplot(223, sharex=ax, sharey=ax), imshow(gcan2, new_fig=False, title="Canny - U1=20% | U2=40%")
plt.show(block=False)


# ---- Clausura (Closing) -----------------------
A = gcan2
#imshow(A, title="Imagen Original")

B = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 11)) #(25,12)

#kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#A_dilatado = cv2.dilate(gcan2, kernel_dilatacion)

#Aclau = cv2.morphologyEx(A_dilatado, cv2.MORPH_CLOSE, B)
Aclau = cv2.morphologyEx(A, cv2.MORPH_OPEN, B)
Aclau = cv2.morphologyEx(A, cv2.MORPH_CLOSE, B)

plt.figure()
ax1 = plt.subplot(121); imshow(A, new_fig=False, title="Original")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(Aclau, new_fig=False, title="Clausura")
plt.show(block=False)

# ---- Componentes conectados -----------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Aclau, 
connectivity, cv2.CV_32S)#cv2.CV_32S cv2.CC_STAT_AREA

# ---- Bounding box (codigo profe)
#im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)
#for centroid in centroids:
#    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
#for st in stats:
#    cv2.rectangle(im_color,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(0,255,0),thickness=2)
#imshow(img=im_color, color_img=True)

# ---- Bounding box ------- 
MIN_AREA = 3000
im_color = cv2.applyColorMap(np.uint8(255 / num_labels * labels), cv2.COLORMAP_JET)

for centroid in centroids[1:]:  # Omitimos el fondo
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
        continue
    x, y, w, h = stats[i, :4]
    cv2.rectangle(im_color, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

imshow(img=im_color, color_img=True)


# --- Parámetros ---------------------------------------------------------------
RHO_TH = 0.35      # Umbral del factor de forma para círculo
AREA_TH = 12000     # Área mínima
SQUARE_TOL = 0.2   # Tolerancia para considerar proporcion cuadrada (20%)
H, W = img.shape[:2]
aux = np.zeros_like(labels)
labeled_image = cv2.merge([aux, aux, aux])  # Imagen en RGB

# --- Clasificación ------------------------------------------------------------
for i in range(1, num_labels):

    # --- Descarto figuras que tocan los bordes -------------------------------
    if (stats[i, cv2.CC_STAT_LEFT] == 0 or 
        stats[i, cv2.CC_STAT_TOP] == 0 or 
        stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] == W or 
        stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] == H):
        continue

    # --- Descarto figuras con área muy pequeña -------------------------------
    if stats[i, cv2.CC_STAT_AREA] < AREA_TH:
        continue

    # --- Máscara del objeto actual -------------------------------------------
    obj = (labels == i).astype(np.uint8)

    # --- Relleno huecos internos con operación de cierre + relleno -----------
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    obj_closed = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, se)
    obj_filled = cv2.bitwise_not(cv2.bitwise_not(obj_closed))  # opcional si necesitás asegurar que sea sólido

    # --- Cálculo del factor de forma (rho) -----------------------------------
    contours, _ = cv2.findContours(obj_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue
    area = cv2.contourArea(contours[0])
    perimeter = cv2.arcLength(contours[0], True)
    if perimeter == 0:
        continue
    rho = 4 * np.pi * area / (perimeter ** 2)
    flag_circular = rho > RHO_TH

    # --- Verifico si tiene forma cuadrada ------------------------------------
    bbox_width  = stats[i, cv2.CC_STAT_WIDTH]
    bbox_height = stats[i, cv2.CC_STAT_HEIGHT]
    ratio = bbox_width / bbox_height
    flag_square_like = (1 - SQUARE_TOL) <= ratio <= (1 + SQUARE_TOL)

    print(f"Objeto {i:2d} --> Circular: {flag_circular}  /  Cuadrado: {flag_square_like}  /  Rho: {rho:.3f}  /  WH_ratio: {ratio:.2f}")

    # --- Clasificación visual ------------------------------------------------
    if flag_circular or flag_square_like:
        labeled_image[obj_filled == 1, 1] = 255  # Verde = círculo
    #elif flag_square_like:
        #labeled_image[obj_filled == 1, 2] = 255  # Azul = cuadrado o capacitor
    else:
        labeled_image[obj_filled == 1, 0] = 255  # Rojo = otra forma

# --- Visualización ------------------------------------------------------------
plt.figure()
plt.imshow(labeled_image)
plt.title("Clasificación: Verde = círculo, Azul = cuadrado, Rojo = otro")
plt.show(block=False)

#- -----RESISTENCIAS------
# --- Parámetros ---------------------------------------------------------------
AREA_MIN = 3000
AREA_MAX = 10000           # Área maxima para descartar ruido
ASPECT_RATIO_TH = 2.5    # Umbral de relación de aspecto para considerar resistencia
H, W = img.shape[:2]
aux = np.zeros_like(labels)
labeled_image = cv2.merge([aux, aux, aux])  # Imagen en RGB

# --- Clasificación por relación de aspecto -----------------------------------
for i in range(1, num_labels):

    # --- Descarto objetos que tocan el borde ---------------------------------
    if (stats[i, cv2.CC_STAT_LEFT] == 0 or 
        stats[i, cv2.CC_STAT_TOP] == 0 or 
        stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] == W or 
        stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] == H):
        continue
    # --- Descarto objetos Chicos -------------------------------------------
    if stats[i, cv2.CC_STAT_AREA] < AREA_MIN:
        continue
    # --- Descarto objetos grandes -------------------------------------------
    if stats[i, cv2.CC_STAT_AREA] > AREA_MAX:
        continue

    # --- Máscara del objeto --------------------------------------------------
    obj = (labels == i).astype(np.uint8)

    # --- Cálculo relación de aspecto -----------------------------------------
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    if height == 0:
        continue
    aspect_ratio = width / height

    # --- Clasificación -------------------------------------------------------
    # --- Cálculo relación de aspecto robusta -------------------------------------
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    if width == 0 or height == 0:
        continue
    aspect_ratio = max(width, height) / min(width, height)

    # --- Clasificación como resistencia ------------------------------------------
    is_resistor = aspect_ratio > ASPECT_RATIO_TH

    print(f"Objeto {i:2d} --> Resistencia: {is_resistor}  /  Aspect Ratio: {aspect_ratio:.2f}")

    # --- Visualización -------------------------------------------------------
    if is_resistor:
        labeled_image[obj == 1, 2] = 255  # Azul para resistencias
    else:
        labeled_image[obj == 1, 0] = 255  # Rojo para no-resistencias

# --- Visualización final -----------------------------------------------------
plt.figure()
plt.imshow(labeled_image)
plt.title("Clasificación: Azul = resistencia, Rojo = otro")
plt.show(block=False)


#AREA CHIP: 178164
#RESISTENCIAS_relacion:0.5