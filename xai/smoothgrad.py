# Importando as bibliotecas
import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
from keras.models import load_model
import cv2

import saliency.core as saliency

import time
import tracemalloc


tracemalloc.start()
tpi = time.time()



def ShowImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')
  P.imshow(im)
  P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')

  P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
  P.title(title)

def ShowHeatMap(im, title, ax=None):
  if ax is None:
    P.figure()
  P.axis('off')
  P.imshow(im, cmap='inferno')
  P.title(title)

def LoadImage(file_path):
  im = PIL.Image.open(file_path)
  im = im.resize((224,224))
  im = np.asarray(im)
  return im

def PreprocessImage(im):
  im = tf.keras.applications.vgg16.preprocess_input(im)
  return im

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
# Parâmetros - Ajustar de acordo com o padrão esperado pela CNN
# ----------------------------------------------------------------
resolution = 224
#resolution = 229

# ----------------------------------------------------------------
# Carrega o modelo treinado (arquivo .h5) utilizando o método 
# load_model do pacote keras.models
# ----------------------------------------------------------------
model = load_model("Caminho para carregar o modelo treinado (.h5)", compile=False)

conv_layer = get_last_conv_layer(model)
#conv_layer = str(conv_layer)
conv_layer = model.get_layer(conv_layer)
#print (conv_layer)
model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    target_class_idx =  call_model_args[class_idx_str]
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            _, output_layer = model(images)
            output_layer = output_layer[:,target_class_idx]
            gradients = np.array(tape.gradient(output_layer, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv_layer, output_layer = model(images)
            gradients = np.array(tape.gradient(output_layer, conv_layer))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}



# ----------------------------------------------------------------
# Carrega a imagem de exemplo, converte o padrão de cor (RGB) 
# e redimensiona a imagem para o tamanho esperado pelo modelo
# ---------------------------------------------------------------- 
im = cv2.imread('Caminho para carregar a imagem de entrada')
img_orig = im.copy()
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (resolution,resolution), interpolation = cv2.INTER_CUBIC)
im = PreprocessImage(im)

_, predictions = model(np.array([im]))
prediction_class = np.argmax(predictions[0])
call_model_args = {class_idx_str: prediction_class}

# Construct the saliency object. This alone doesn't do anthing.
gradient_saliency = saliency.GradientSaliency()

# Compute the vanilla mask and the smoothed mask.
smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, call_model_function, call_model_args)

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

imagem_rgb_np = np.stack((smoothgrad_mask_grayscale,)*3, axis=-1)

tpf = time.time()

tpt = tpf-tpi

#print("Memória utilizada: ", tracemalloc.get_traced_memory())
print("Tempo de processamento: ", tpt)
tracemalloc.stop()

fig, (ax1, ax2) = P.subplots(1, 2, figsize=(20,20))
fig.suptitle(cnn, fontsize=16)

ax1.set_title("Imagem de Entrada")
ax1.imshow(cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR))
ax1.axis('off')

ax2.set_title("Imagem de Saída")
ax2.imshow(imagem_rgb_np)
ax2.axis('off')

# Adicionando legendas personalizadas
ax1.text(0.5, -0.04, "Nervo Óptico Centralizado", size=12, ha="center", transform=ax1.transAxes)
ax2.text(0.5, -0.04, "SmoothGrad", size=12, ha="center", transform=ax2.transAxes)

# Adicionando legenda para o rodapé da figura explicando as cores do mapa de ativação
fig.text(0.5, 0.02, "A cor BRANCA na imagem SmoothGrad representa os pixels mais significativos para o diagnóstico.", ha="center", fontsize=12)

P.show()

