import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Cargo imagen ------------------------------------------------------------
img_path = os.path.join(os.getcwd(), 'placa.png')
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("La imagen no se pudo cargar. Verifica el path o nombre del archivo.")

plt.figure(), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Imagen Original'), plt.show(block=False)

# --- 