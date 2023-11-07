# ----------------------------------------------------------------
# Importa os pacotes necessários para rodar a aplicação
# ----------------------------------------------------------------
import numpy as np
from keras.models import load_model

k = 5

# ----------------------------------------------------------------
# Caminho para os diretórios
# ----------------------------------------------------------------
models_dir = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/models'

models = []
for i in range(k):
    model = load_model(models_dir+'/inceptopnv3_fold_'+str(i)+'.h5')
    models.append(model)

# ----------------------------------------------------------------
# Calculando a média dos pesos de cada camada
# ----------------------------------------------------------------
num_layers = len(models[0].get_weights())
avg_weights = []
for layer_idx in range(num_layers):
    layer_weights = []
    for model in models:
        layer_weights.append(model.get_weights()[layer_idx])
    avg_layer_weights = np.mean(layer_weights, axis=0)
    avg_weights.append(avg_layer_weights)

# ----------------------------------------------------------------
# Criando um novo modelo com os pesos médios
# ----------------------------------------------------------------
final_model = load_model(models_dir+'/inceptopnv3_fold_0.h5')
final_model.set_weights(avg_weights)

# ----------------------------------------------------------------
# Salvando o modelo final
# ----------------------------------------------------------------
final_model.save(models_dir+'/final_model_inceptionv3.h5')