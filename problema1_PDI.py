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

gcan2 = cv2.Canny(f_blur, threshold1=0.4*255, threshold2=0.5*255)

plt.figure()
ax = plt.subplot(221)
imshow(f, new_fig=False, title="Imagen Original")
plt.subplot(223, sharex=ax, sharey=ax), imshow(gcan2, new_fig=False, title="Canny - U1=40% | U2=50%")
plt.show(block=False)


