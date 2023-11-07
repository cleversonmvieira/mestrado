# ----------------------------------------------------------------
# Importa os pacotes necessários para rodar a aplicação
# ----------------------------------------------------------------
import shap
#from shap import explainers as explainers
import numpy as np
from keras.models import load_model
import cv2
import os
import tracemalloc
import time


import time

tracemalloc.start()
tpi = time.time()

# Caminho dos diretórios de treinamento e teste
train_dir = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/acrima_wm/data_augmentation'
#test_dir = 'C:/Users/Cleverson M. Vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/acrima/test'

size = (224,224)

def imagearray(test_dir, size):
  data = []
 
  for folder in os.listdir(test_dir):
    sub_path = test_dir + "/" + folder
    
    for img in os.listdir(sub_path):
      image_path = sub_path + "/" + img
      img_arr = cv2.imread(image_path)
      img_arr = cv2.resize(img_arr, size)
      data.append(img_arr)
  
  return data



test = imagearray(train_dir, size)
x_test = np.array(test)
x_test = x_test/255
x_test = x_test.astype('float32')


# ----------------------------------------------------------------
# Carrega o modelo treinado (arquivo .h5) utilizando o método 
# load_model do pacote keras.models
# ----------------------------------------------------------------
model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_vgg16.h5")
#model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_vgg19.h5")
#model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_inceptionv3.h5")
#model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_densenet.h5")
#model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_xceptionnet.h5")
#model = load_model("C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models/final_model_resnet50.h5")

# ----------------------------------------------------------------
# Carrega a imagem de exemplo, converte o padrão de cor (RGB) 
# e redimensiona a imagem para o tamanho esperado pelo modelo
# ---------------------------------------------------------------- 
img = cv2.imread('C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/acrima/validation/positive/5826.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224,224), interpolation = cv2.INTER_CUBIC)
img_orig = img.copy()


# ----------------------------------------------------------------
# Converte a imagem de exemplo para um array numpy, adiciona
# uma dimensão extra para simular o batch_size e normaliza
# ---------------------------------------------------------------- 
img = np.array(img)
img = np.expand_dims(img, axis=0)
#img = img / 255.0

#explainer = shap.DeepExplainer(model, img)
#shap_values = explainer.shap_values(img)

# Define o objeto explainer para o SHAP
explainer = shap.GradientExplainer(model, x_test)
# Gera a explicação com SHAP
shap_values = explainer.shap_values(img)


tpf = time.time()
tpt = tpf - tpi
print("Memória utilizada: ", tracemalloc.get_traced_memory())
print("Tempo de processamento: ", tpt)

tracemalloc.stop()

shap.image_plot(shap_values, img, width = 20, aspect = 0.2, hspace = 0.2, show = True)