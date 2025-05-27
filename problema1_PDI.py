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

# ---- Clausura (Closing) -----------------------
A = gcan2
#imshow(A, title="Imagen Original")

B = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
Aclau = cv2.morphologyEx(A, cv2.MORPH_CLOSE, B)

plt.figure()
ax1 = plt.subplot(121); imshow(A, new_fig=False, title="Original")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(Aclau, new_fig=False, title="Clausura")
plt.show(block=False)

# ---- Componentes conectados -----------------------
connectivity = 4
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Aclau, 
connectivity, cv2.CV_32S)

im_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)
for centroid in centroids:
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
    cv2.rectangle(im_color,(st[0],st[1]),(st[0]+st[2],st[1]+st[3]),color=(0,255,0),thickness=2)
imshow(img=im_color, color_img=True)


