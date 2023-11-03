import cv2
from matplotlib import pyplot as plt
import numpy as np


img_path = 'C:/Users/Cleverson M. Vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/validation/positive/5826.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))  # Redimensionar para o tamanho esperado pela CNN
img_orig = img.copy()

image_path = 'C:/Users/Cleverson M. Vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/teste.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))  # Redimensionar para o tamanho esperado pela CNN

lower = (0, 127, 127)
upper = (127, 255, 255)

mask = cv2.inRange(image, lower, upper)

#scim = cv2.medianBlur(mask, 5)

kernel = np.ones((5, 5), np.uint8)
scim = cv2.dilate(mask, kernel, iterations=1)

# Encontra os contornos na interseção
contours, hierarchy = cv2.findContours(scim, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Desenha o contorno na nova imagem
scim_contourns = cv2.drawContours(img_orig, contours, -1, (0, 0, 255), 2)

# Plotar a imagem original, o Grad-CAM, as importâncias do Shap e o mapa integrado
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,20))
ax1.set_title("Input")
ax1.imshow(img)
ax1.axis('off')
ax2.set_title("Interpretable Mapping")
ax2.imshow(image)
ax2.axis('off')
ax3.set_title("Filter Range")
ax3.imshow(mask)
ax3.axis('off')
ax4.set_title("Scim")
ax4.imshow(scim_contourns)
ax4.axis('off')
plt.show()