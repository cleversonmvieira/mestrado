# ----------------------------------------------------------------
# Importa os pacotes necessários para rodar a aplicação
# ----------------------------------------------------------------
import tensorflow as tf
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import time
import tracemalloc


tracemalloc.start()
tpi = time.time()

# ----------------------------------------------------------------
# Parâmetros
# ----------------------------------------------------------------
resolution = 224

# ----------------------------------------------------------------
# Método que obtém o nome do último layer convolucional da CNN
# ----------------------------------------------------------------
def get_last_conv_layer(model):
    last_conv_layer_name = None
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        return False
    else:
        return last_conv_layer_name
    
# ----------------------------------------------------------------
# Método que obtém o nome do último layer denso da CNN
# ----------------------------------------------------------------
def get_last_dense_layer(model):
    last_dense_layer_name = None
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Dense):
            last_dense_layer_name = layer.name
            break

    if last_dense_layer_name is None:
        return False
    else:
        return last_dense_layer_name    

# ----------------------------------------------------------------
# Carrega o modelo treinado (arquivo .h5) utilizando o método 
# load_model do pacote keras.models
# ----------------------------------------------------------------
#model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_vgg16.h5")
#model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_vgg19.h5")
#model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_inceptionv3.h5")
#model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_densenet.h5")
#model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_xceptionnet.h5")
model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_resnet50.h5")

# ----------------------------------------------------------------
# Carrega a imagem de exemplo, converte o padrão de cor (RGB) 
# e redimensiona a imagem para o tamanho esperado pelo modelo
# ---------------------------------------------------------------- 
img = cv2.imread('C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/acrima/validation/positive/5826.jpg')
#img = cv2.imread('C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/acrima/validation/positive/0004.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (resolution,resolution), interpolation = cv2.INTER_CUBIC)
img_orig = img.copy()
# Removendo canal AZUL 
#img[:,:,0] = 0  
# Removendo canal VERDE (comentado)
#img[:,:,1] = 0  
# Removendo canal VERMELHO 
#img[:,:,2] = 0  

# ----------------------------------------------------------------
# Converte a imagem de exemplo para um array numpy, adiciona
# uma dimensão extra para simular o batch_size e normaliza
# ---------------------------------------------------------------- 
img = np.array(img)
img = np.expand_dims(img, axis=0)
#img = img / 255.0

# ----------------------------------------------------------------
# Obtenha a camada convolucional final e a camada de classificação
# ----------------------------------------------------------------
last_conv_layer = get_last_conv_layer(model) 
classifier_layer = get_last_dense_layer(model)

# Obter as camadas do Keras pelo nome
classifier_layer = model.get_layer(classifier_layer)
last_conv_layer = model.get_layer(last_conv_layer)

# ----------------------------------------------------------------
# Calcule os gradientes da saída da camada de classificação com 
# relação à saída da camada convolucional final
# ----------------------------------------------------------------
with tf.GradientTape() as tape:
    preds, last_conv_layer_output = tf.keras.models.Model(inputs=model.input, outputs=[classifier_layer.output, last_conv_layer.output])(img)
    grads = tape.gradient(preds, last_conv_layer_output)[0]

# ----------------------------------------------------------------
# Calcula o mapa de ativação da classe (CAM) multiplicando os 
# gradientes pela saída da camada convolucional final e tomando 
# a média dos canais
# ----------------------------------------------------------------
cam = np.mean(last_conv_layer_output.numpy()[0], axis=-1)
cam = np.maximum(cam, 0)
cam = cam / np.max(cam)
cam = cv2.resize(cam, (resolution, resolution))

# ----------------------------------------------------------------
# Aplica o ColorMap Jet e sobrepõe o mapa de calor na imagem de 
# exemplo
# ----------------------------------------------------------------
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
cam_img = heatmap + np.float32(img_orig) / 255
cam_img = cam_img / np.max(cam_img)

tpf = time.time()

tpt = tpf-tpi

print("Memória utilizada: ", tracemalloc.get_traced_memory())
print("Tempo de processamento: ", tpt)
tracemalloc.stop()
# ----------------------------------------------------------------
# Visualiza a imagem de exemplo e a sobreposição do mapa de 
# ativação da classe (CAM)
# ----------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,20))
ax1.set_title("Input")
ax1.imshow(img_orig)
ax1.axis('off')
ax2.set_title("CAM")
ax2.imshow(heatmap)
ax2.axis('off')
ax3.set_title("Output")
ax3.imshow(cam_img)
ax3.axis('off')
plt.show()