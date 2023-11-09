import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model


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


# Carregue uma imagem de fundo de retina
input_image_path = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/acrima/validation/positive/269.jpg'  # Substitua pelo caminho da sua imagem
img = cv2.imread(input_image_path)
img = cv2.resize(img, (224, 224))  # Redimensione a imagem para o tamanho esperado pela rede

# Carregue o modelo de rede neural (usaremos o VGG16 como exemplo)
model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_vgg19.h5")

# Pré-processamento da imagem
img = np.array(img)
img = np.expand_dims(img, axis=0)

# Classe de destino (classe de interesse)
target_class = 0  # Substitua pelo índice da classe de interesse

# Obtenha a camada de saída da última camada convolucional da rede
last_conv_layer_name = get_last_conv_layer(model)

# Crie um modelo que retorna ativações e gradientes para a camada alvo
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

with tf.GradientTape() as tape:
    last_conv_layer_output, model_output = grad_model(img)
    loss = model_output[:, target_class]

# Calcule gradientes
grads = tape.gradient(loss, last_conv_layer_output)

# Calcule os pesos do Grad-CAM
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
last_conv_layer_output = last_conv_layer_output[0]
heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

# Normalize o mapa de calor
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.math.reduce_max(heatmap)

# Converta o mapa de calor para uma imagem
heatmap = heatmap.numpy()
heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
heatmap = np.uint8(255 * heatmap)

# Aplique o mapa de calor à imagem de entrada
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img[0], 0.6, heatmap, 0.4, 0)

# Exiba a imagem original, o mapa de calor e a imagem resultante
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")

plt.subplot(132)
plt.imshow(heatmap, cmap='jet')
plt.title("Mapa de Calor")

plt.subplot(133)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("Imagem com Mapa de Calor")

plt.show()
