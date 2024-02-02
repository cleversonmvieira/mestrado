# ----------------------------------------------------------------
# Importa os pacotes necessários para rodar a aplicação
# ----------------------------------------------------------------
import lime
from lime import lime_image
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, slic
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import time
import tracemalloc

tracemalloc.start()
tpi = time.time()

# Verificar/ajustar para o padrão esperado pela CNN
size = (224,224)
#size = (229,229)

# ----------------------------------------------------------------
# Carrega o modelo treinado (arquivo .h5) utilizando o método 
# load_model do pacote keras.models
# ----------------------------------------------------------------
model = load_model("Caminho para carregar o modelo treinado (.h5)")

# ----------------------------------------------------------------
# Carrega a imagem de exemplo, converte o padrão de cor (RGB) 
# e redimensiona a imagem para o tamanho esperado pelo modelo
# ---------------------------------------------------------------- 
img = cv2.imread('C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/acrima/validation/positive/5826.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, size, interpolation = cv2.INTER_CUBIC)
img_orig = img.copy()

# ----------------------------------------------------------------
# Converte a imagem de exemplo para um array numpy, adiciona
# uma dimensão extra para simular o batch_size e normaliza
# ---------------------------------------------------------------- 
#img = np.array(img)
#img = np.expand_dims(img, axis=0)
#img = img / 255.0

# ----------------------------------------------------------------
# Cria a segmentação na imagem de entrada utilizando o algoritmo
# slic - Segmenta a imagem em 100 partes
# ---------------------------------------------------------------- 
segments_slic = slic(img, n_segments=100, compactness=10, sigma=1, start_label=1)

# ----------------------------------------------------------------
# Cria o explicador LIME para gerar uma explicação local usando
# o método LimeImageExplainer()
# ---------------------------------------------------------------- 
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    img.astype('double'), 
    model.predict, 
    top_labels = 2, 
    hide_color = 0, 
    num_samples = 1000, 
    segmentation_fn = slic
    )

# ----------------------------------------------------------------
# Obtém a imagem e a máscara da explicação local gerada com as 10
# regiões positivas e ocultando o restante da região da imagem  
# ---------------------------------------------------------------- 
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only = True, 
    num_features = 10, 
    hide_rest = True
    )

ind =  explanation.top_labels[0]
dict_heatmap = dict(explanation.local_exp[ind])
heatmap_orig = np.vectorize(dict_heatmap.get)(explanation.segments) 

# ----------------------------------------------------------------
# Cria uma máscara com transparência para as regiões importantes
# ---------------------------------------------------------------- 
heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
heatmap[np.where(mask == 0)] = 0
heatmap = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

# ----------------------------------------------------------------
# Aplica a máscara na imagem original
# ----------------------------------------------------------------
masked_img = img.copy()
masked_img[np.where(mask == 0)] = 0


tpf = time.time()
tpt = tpf - tpi
print("Memória utilizada: ", tracemalloc.get_traced_memory())
print("Tempo de processamento: ", tpt)

tracemalloc.stop()

# ----------------------------------------------------------------
# Visualiza a imagem de exemplo, o Grad-CAM e o mapa de calor 
# (heatmap) sobreposto na imagem de exemplo
# ----------------------------------------------------------------
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(5, 5), sharex=True, sharey=True)
ax1.set_title("Entrada")
ax1.imshow(img)
ax1.axis('off')
ax2.set_title("SLIC")
ax2.imshow(mark_boundaries(img, segments_slic))
ax2.axis('off')
ax3.set_title('LIME')
ax3.imshow(mark_boundaries(temp / 2 + 0.5, mask))
ax3.axis('off')
ax4.set_title('Regiões Importantes')
ax4.imshow(heatmap)
ax4.axis('off')
ax5.set_title('Heatmap')
heat = ax5.imshow(heatmap_orig, cmap = "RdBu", vmin  = -heatmap_orig.max(), vmax = heatmap_orig.max())
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(heat, cax=cax)
ax5.axis('off')
plt.show()
