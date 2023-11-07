# ----------------------------------------------------------------
# Importa os pacotes necessários para rodar a aplicação
# ----------------------------------------------------------------
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# ----------------------------------------------------------------
# Diretório contendo as imagens originais
# ----------------------------------------------------------------
dir_path = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/acrima_wm/negative'
dest_path = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/acrima_wm/data_augmentation/negative'

# ----------------------------------------------------------------
# Configuração do gerador de imagens aumentadas
# ----------------------------------------------------------------
datagen = ImageDataGenerator(
    brightness_range=[0.2, 0.8], # brilho de 20% a 80%
    rotation_range=20, # range de rotação de 20 graus
    width_shift_range=0.1, # deslocamento horizontal de 10%
    height_shift_range=0.1, # deslocamento vertical de 10%
    shear_range=0.1, # range de cisalhamento de 10%
    zoom_range=0.1, # range de zoom de 10%
    horizontal_flip=True, # espelhamento horizontal
    #vertical_flip=True, # espelhamento vertical
    fill_mode='nearest' # preenchimento dos pixels vazios
)

# ----------------------------------------------------------------
# Loop pelos arquivos do diretório
# ----------------------------------------------------------------
for filename in os.listdir(dir_path):
    # Carrega a imagem em um array numpy
    img = np.array(Image.open(os.path.join(dir_path, filename)))
    # Adiciona uma dimensão ao array (para se adequar ao formato de entrada da CNN)
    img = np.expand_dims(img, axis=0)
    # Gera 10 novas imagens aumentadas para cada imagem do diretório (dir_path)
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=dest_path, save_prefix='aug_'+filename[:-4]+'_', save_format='jpg'):
        i += 1
        if i >= 2:
            break