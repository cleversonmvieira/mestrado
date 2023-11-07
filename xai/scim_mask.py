import cv2
from matplotlib import pyplot as plt
import numpy as np


img_path = 'data/acrima/validation/positive/5826.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))  # Redimensionar para o tamanho esperado pela CNN
img_orig = img.copy()

#image_path = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/vgg16_shap.png'
image_path = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/vgg19_shap.png'
#image_path = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/inceptionv3_shap.png'
#image_path = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/densenet_shap.png'
#image_path = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/xceptionnet_shap.png'
#image_path = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/resnet50_shap.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))  # Redimensionar para o tamanho esperado pela CNN

image2_path = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/vgg19_gradcam.png'
image2 = cv2.imread(image2_path)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2 = cv2.resize(image2, (224, 224))  # Redimensionar para o tamanho esperado pela CNN



lower_rose = (215, 0, 0)
upper_rose = (255, 127, 220)

lower_blue = (0, 0, 150)
upper_blue = (200, 184, 255)


mask_rose = cv2.inRange(image, lower_rose, upper_rose)

mask_blue = cv2.inRange(image2, lower_blue, upper_blue)

#scim = cv2.medianBlur(mask, 5)

kernel = np.ones((3, 3), np.uint8)
mask_rose_dilated = cv2.dilate(mask_rose, kernel, iterations=2)
mask_blue_dilated = cv2.dilate(mask_blue, kernel, iterations=2)

intersection = cv2.bitwise_and(mask_rose_dilated, mask_blue_dilated)
intersection_dilated = cv2.dilate(intersection, kernel, iterations=3)

contours, hierarchy = cv2.findContours(intersection_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Desenha o contorno na nova imagem
scim = cv2.drawContours(img_orig, contours, -1, (0, 0, 0), 2)

heatmap = cv2.applyColorMap(intersection_dilated, cv2.COLORMAP_PINK)
result = cv2.addWeighted(img_orig, 0.7, heatmap, 0.3, 0)

# Plotar a imagem original, o Grad-CAM, as importâncias do Shap e o mapa integrado
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20,20))
ax1.set_title("Input")
ax1.imshow(img)
ax1.axis('off')
ax2.set_title("Mask SHAP")
ax2.imshow(mask_rose_dilated)
ax2.axis('off')
ax3.set_title("Mask Grad-CAM")
ax3.imshow(mask_blue_dilated)
ax3.axis('off')
ax4.set_title("Intersection")
ax4.imshow(intersection_dilated)
ax4.axis('off')
ax5.set_title("SCIM")
ax5.imshow(result)
ax5.axis('off')
plt.show()
