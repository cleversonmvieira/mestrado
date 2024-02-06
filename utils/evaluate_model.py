from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Parâmetros
epochs = 10
resolution = 224
#resolution = 229
batch_size = 8

# Caminho dos diretórios (treinamento, teste, relatórios)
train_dir = 'Caminho para o diretório de treinamento'
test_dir = 'Caminho para o diretório de teste'
models_dir = 'Caminho para o diretório dos modelos'
reports_dir = 'Caminho para o diretório dos relatórios'


# Caminhos dos diretórios com os modelos treinados
model_vgg16 = 'Caminho para o diretório com o modelo treinado - vgg16.h5'
model_vgg19 = 'Caminho para o diretório com o modelo treinado - vgg19.h5'
model_inceptionv3 = 'Caminho para o diretório com o modelo treinado - inceptionv3.h5'
model_densenet = 'Caminho para o diretório com o modelo treinado - densenet.h5'
model_xception = 'Caminho para o diretório com o modelo treinado - xception.h5'
model_resnet50 = 'Caminho para o diretório com o modelo treinado - resnet50.h5'


# Carregar o modelo treinado - Ajustar conforme modelo 
model = load_model(model_vgg16)

# Resumo do modelo
model.summary()

# Criação dos geradores de imagens
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(resolution, resolution), batch_size=batch_size, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(resolution, resolution), batch_size=batch_size, class_mode='binary')


print('')
print('----------------------------------------')
print('Iniciando a avaliação do modelo')
print('----------------------------------------')

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator), verbose=1)

print('')
print('Acurácia do modelo:', test_acc)
print('Perda do modelo:', test_loss)

print('')
print('----------------------------------------')
print('Avaliação do modelo finalizada')
print('----------------------------------------')

print('')
print('----------------------------------------')
print('Iniciando as previsões do modelo')
print('----------------------------------------')

# Previsões do modelo
y_pred = model.predict(test_generator, steps=len(test_generator)).round()

print('')
print('----------------------------------------')
print('Previsões do modelo finalizadas')
print('----------------------------------------')

print('')
print('----------------------------------------')
print('Métricas de desempenho')
print('----------------------------------------')

print("Acurácia: ",accuracy_score(test_generator.classes, y_pred))
print("Sensibilidade: ",recall_score(test_generator.classes, y_pred))
print("Precisão: ",precision_score(test_generator.classes, y_pred))
print("F1-Score: ",f1_score(test_generator.classes, y_pred))
print("AUC: ",roc_auc_score(test_generator.classes, y_pred))
