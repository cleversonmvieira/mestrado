# ----------------------------------------------------------------
# Importa os pacotes necessários para rodar a aplicação
# ----------------------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tracemalloc
import time

tracemalloc.start()
tpi = time.time()

# ----------------------------------------------------------------
# Parâmetros - Ajustar de acordo com o padrão esperado pela CNN
# ----------------------------------------------------------------
resolution = 224
#resolution = 229

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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(tpi, image, heatmap, conv_layer_name, cam_path="cam.jpg", alpha=0.9):
    # Load the original image
    #img = keras.preprocessing.image.load_img(img)
    img = keras.preprocessing.image.img_to_array(image)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    tpf = time.time()
    tpt = tpf - tpi
    print("Memória utilizada: ", tracemalloc.get_traced_memory())
    print("Tempo de processamento: ", tpt)
    tracemalloc.stop()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,20))

    ax1.set_title("Input")
    ax1.imshow(image)
    ax1.axis("off")

    ax2.set_title("Grad-CAM")
    ax2.imshow(heatmap)
    ax2.axis("off")

    ax3.set_title("Heatmap")
    ax3.imshow(superimposed_img)
    ax3.axis("off")
    
    plt.show()



# ----------------------------------------------------------------
# Carrega o modelo treinado (arquivo .h5) utilizando o método 
# load_model do pacote keras.models
# ----------------------------------------------------------------
model = load_model("Caminho para carregar o modelo treinado (.h5)")

# ----------------------------------------------------------------
# Armazena na variável "last_conv_layer_name" o nome do último 
# ----------------------------------------------------------------
#layer convolucional da CNN carregada
last_conv_layer_name = get_last_conv_layer(model)

# ----------------------------------------------------------------
# Carrega a imagem de exemplo, converte o padrão de cor (RGB) 
# e redimensiona a imagem para o tamanho esperado pelo modelo
# ---------------------------------------------------------------- 
img = cv2.imread('Caminho para carregar a imagem de entrada')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (resolution,resolution), interpolation = cv2.INTER_CUBIC)
img_orig = img.copy()

# ----------------------------------------------------------------
# Converte a imagem de exemplo para um array numpy, adiciona
# uma dimensão extra para simular o batch_size e normaliza
# ---------------------------------------------------------------- 
img = np.array(img)
img = np.expand_dims(img, axis=0)
#img = img / 255.0
  
heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)
save_and_display_gradcam(tpi, img_orig, heatmap, last_conv_layer_name) 

