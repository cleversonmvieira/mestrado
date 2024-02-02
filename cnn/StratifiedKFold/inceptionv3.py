# ----------------------------------------------------------------
# Importa os pacotes necessários para rodar a aplicação
# ----------------------------------------------------------------
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import itertools
from keras.applications.inception_v3 import InceptionV3

# ----------------------------------------------------------------
# Parâmetros
# ----------------------------------------------------------------
epochs = 10
resolution = 229
batch_size = 8

# Definir o número de partições para a validação cruzada estratificada
k = 5

# ----------------------------------------------------------------
# Caminho para os diretórios
# ----------------------------------------------------------------
models_dir = 'Diretório para salvar os modelos treinados'
reports_dir = 'Diretório para salvar os relatórios'
data_path = "Diretório com o dataset"

# ----------------------------------------------------------------
# Função para plotar a figura com a matriz de confusão para cada Fold
# ----------------------------------------------------------------
def plot_confusion_matrix_fold(cm, fold, classes, normalize=False, cmap=plt.cm.Blues):
    """
    Esta função plota a figura com a matriz de confusão.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print('Matriz de confusão sem normalização')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(reports_dir+'/confusion_matrix/confusion_matrix_fold_'+str(fold)+'_inceptionv3.jpg')

# ----------------------------------------------------------------
# Inicializa os vetores que irão armazenar os dados de treinamento
# e os rótulos das classes
# ----------------------------------------------------------------
data = []
labels = []

# ----------------------------------------------------------------
# Definir as classes do problema (negative/positive)
# O nome das classes deve ser o nome do diretório
# ----------------------------------------------------------------
classes = ["negative", "positive"]

# ----------------------------------------------------------------
# Percorrer as subpastas e carregar as imagens em matrizes NumPy
# ----------------------------------------------------------------
for cls in classes:
    cls_path = os.path.join(data_path, cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (resolution, resolution))  # Redimensionar a imagem para o tamanho esperado pelo modelo
        data.append(img)
        labels.append(classes.index(cls))
        print("Processando: ", img_name)

# ----------------------------------------------------------------
# Converter as listas de dados e labels em matrizes NumPy
# ----------------------------------------------------------------
data = np.array(data)
labels = np.array(labels)

print("Quantidade de imagens: ", len(data))

# ----------------------------------------------------------------
# Definir o tamanho das imagens de entrada
# ----------------------------------------------------------------
img_size = (resolution, resolution)

# ----------------------------------------------------------------
# Criar gerador de imagens para aumento de dados
# ----------------------------------------------------------------
""" datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.1,
                             zoom_range=0.1, horizontal_flip=True) """

datagen = ImageDataGenerator()
# ----------------------------------------------------------------
# Criação do modelo InceptionV3
# ----------------------------------------------------------------
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(resolution, resolution, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# ----------------------------------------------------------------
# Congela as camadas da base da VGG19
# ----------------------------------------------------------------
for layer in base_model.layers:
    layer.trainable = False

# ----------------------------------------------------------------
# Compilação do modelo
# ----------------------------------------------------------------
model.compile(loss='binary_crossentropy', 
              optimizer=Adam(learning_rate=0.0001), 
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
              )

# ----------------------------------------------------------------
# Resumo do modelo
# ----------------------------------------------------------------
model.summary()

# ----------------------------------------------------------------
# Cria o checkpoint para salvar os melhores pesos
# ----------------------------------------------------------------
checkpoint = ModelCheckpoint(models_dir+'/checkpoints/best_inceptionv3.h5', 
                             monitor="val_accuracy", 
                             verbose=1, 
                             save_best_only=True, 
                             mode="auto", 
                             save_freq="epoch"
                             )

# ----------------------------------------------------------------
# Dividir os dados em k partições de treinamento e validação
# ----------------------------------------------------------------
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
skf.get_n_splits(data, labels)

# ----------------------------------------------------------------
# Executar a validação cruzada estratificada
# ----------------------------------------------------------------
acc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
y_true_list = []
y_pred_list = []
cm_list = []
fpr_list = []
tpr_list = []
thresholds_list = []
roc_auc_list = []

for i, (train_index, val_index) in enumerate(skf.split(data, labels)):
    print('Fold: ', i+1)
    train_data, val_data = data[train_index], data[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]
    
    # ----------------------------------------------------------------
    # Gerar mais dados para o conjunto de treinamento
    # ----------------------------------------------------------------
    train_datagen = datagen.flow(train_data, train_labels, batch_size=batch_size)
    
    # ----------------------------------------------------------------
    # Treinar o modelo
    # ----------------------------------------------------------------
    history = model.fit(train_datagen, epochs=epochs, validation_data=(val_data, val_labels), callbacks=checkpoint)
    
    # ----------------------------------------------------------------
    # Salvar o modelo final
    # ----------------------------------------------------------------
    model.save(models_dir+'/inceptopnv3_fold_'+str(i)+'.h5')
    
    # ----------------------------------------------------------------
    # Avaliar o modelo no conjunto de validação
    # ----------------------------------------------------------------
    print("")
    print("Avaliando o modelo no conjunto de validação - Fold ", i+1)
    loss, accuracy, precision, recall = model.evaluate(val_data, val_labels)
    acc_scores.append(accuracy * 100)
    precision_scores.append(precision * 100)
    recall_scores.append(recall * 100)
    
    f1_s = (2 * (precision * recall) / (precision + recall) * 100)
    
    f1_scores.append(f1_s)
    
    # ----------------------------------------------------------------
    # Prever as classes do conjunto de validação
    # ----------------------------------------------------------------
    print("")
    print("Prevendo as classes no conjunto de validação - Fold ", i+1)
    y_pred = model.predict(val_data)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    y_true_list.extend(val_labels)
    y_pred_list.extend(y_pred)
    
    # ----------------------------------------------------------------
    # Gerar e armazenar a matriz de confusão para a partição de validação
    # ----------------------------------------------------------------
    print("")
    print("Gerando a matriz de confusão - Fold ", i+1)
    cm = confusion_matrix(val_labels, y_pred)
    cm_list.append(cm)

    # ----------------------------------------------------------------
    # Calcular as estatísticas dos resultados
    # ----------------------------------------------------------------
    print("")
    print("Calculando as estatísticas dos resultados - Fold ", i+1)
    mean_acc = np.mean(acc_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    
    std_acc = np.std(acc_scores)
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_f1 = np.std(f1_scores)
    
    # ----------------------------------------------------------------
    # Exibir as estatísticas dos resultados
    # ----------------------------------------------------------------
    print("")
    print("Exibindo as métricas - Fold ", i+1)
    print("Accuracy (average): {:.4f}%".format(mean_acc))
    print("Precision (average): {:.4f}%".format(mean_precision))
    print("Recall (average) {:.4f}%".format(mean_recall))
    print("F1-Score (average): {:.4f}%".format(mean_f1))
    print("")
    print("Desvio padrão - Accuracy: {:.4f}%".format(std_acc))
    print("Desvio padrão - Precision: {:.4f}%".format(std_precision))
    print("Desvio padrão - Recall: {:.4f}%".format(std_recall))
    print("Desvio padrão - F1-Score: {:.4f}%".format(std_f1))

    # ----------------------------------------------------------------
    # Gerar previsões para todo o conjunto de dados
    # ----------------------------------------------------------------
    print("")
    print("Gerando previsões para todo o conjunto de dados - Fold ", i+1)
    preds = model.predict(data)

    # ----------------------------------------------------------------
    # Arredondar as previsões para obter as classes
    # ----------------------------------------------------------------
    preds_rounded = np.round(preds)

    # ----------------------------------------------------------------
    # Calcular as métricas de avaliação
    # ----------------------------------------------------------------
    accuracy = accuracy_score(labels, preds_rounded)
    precision = precision_score(labels, preds_rounded)
    recall = recall_score(labels, preds_rounded)
    f1 = f1_score(labels, preds_rounded)

    # ----------------------------------------------------------------
    # Exibir as métricas de avaliação
    # ----------------------------------------------------------------
    print("Accuracy: {:.4f}%".format(accuracy))
    print("Precision: {:.4f}%".format(precision))
    print("Recall: {:.4f}%".format(recall))
    print("F1-score: {:.4f}%".format(f1))

    # ----------------------------------------------------------------
    # Plotar os gráficos de treinamento e validação
    # Acurácia
    # ----------------------------------------------------------------
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(reports_dir+'/accuracy/acc_inceptionv3_fold_'+str(i)+'.jpg')
    #plt.show()

    # ----------------------------------------------------------------
    # Perda
    # ----------------------------------------------------------------
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(reports_dir+'/loss/loss_inceptionv3_fold_'+str(i)+'.jpg')
    #plt.show()
    
    plt.figure()
    plot_confusion_matrix_fold(cm, i, classes)
    
    fpr, tpr, thresholds = roc_curve(labels, preds_rounded)
    
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    thresholds_list.append(thresholds)
    roc_auc_list.append(roc_auc_score(labels, preds_rounded))
    
    fig, ax = plt.subplots(figsize=(12,4))
    plt.plot(fpr, tpr, color='darkorange', linewidth=2, label='InceptionV3')
    plt.plot([0, 1.05],[0, 1.05], color='navy', linestyle='--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve (AUC = {:.4f}%)' .format(roc_auc_score(labels, preds_rounded)), fontsize=14)
    plt.savefig(reports_dir+'/roc/inceptionv3_fold_'+str(i)+'.jpg')

    print("Área sob a Curva ROC: {:.4f}%" .format(roc_auc_score(labels, preds_rounded)))
    
# ----------------------------------------------------------------
# Calcular as estatísticas dos resultados
# ----------------------------------------------------------------

print("")
print("Calculando as médias das estatísticas dos resultados")
mean_acc = np.mean(acc_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)

std_acc = np.std(acc_scores)
std_precision = np.std(precision_scores)
std_recall = np.std(recall_scores)
std_f1 = np.std(f1_scores)

# ----------------------------------------------------------------
# Calcular a matriz de confusão média
# ----------------------------------------------------------------
mean_cm = np.mean(cm_list, axis=0)

print("")
print("Matriz de confusão média:")
print(mean_cm)


# ----------------------------------------------------------------
# Calcular a curva ROC média
# ----------------------------------------------------------------
mean_tpr = np.mean(tpr_list, axis=0)
mean_fpr = np.mean(fpr_list, axis=0)
mean_thresholds = np.mean(thresholds_list)
mean_roc_auc = np.mean(roc_auc_list)

# ----------------------------------------------------------------
# Exibir as estatísticas dos resultados (média)
# ----------------------------------------------------------------

print("")
print("Exibindo as métricas finais")
print("Accuracy (average): {:.4f}%".format(mean_acc))
print("Precision (average): {:.4f}%".format(mean_precision))
print("Recall (average): {:.4f}%".format(mean_recall))
print("F1-Score (average): {:.4f}%".format(mean_f1))

print("")
print("Desvio padrão - Accuracy: {:.4f}%".format(std_acc))
print("Desvio padrão - Precision: {:.4f}%".format(std_precision))
print("Desvio padrão - Recall: {:.4f}%".format(std_recall))
print("Desvio padrão - F1-Score: {:.4f}%".format(std_f1))    

print("")
print("Área sob a Curva ROC Média: {:.4f}%" .format(mean_roc_auc))
print("")

# ----------------------------------------------------------------
# Função para plotar a figura com a matriz de confusão média
# ----------------------------------------------------------------
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    """
    Esta função plota a figura com a matriz de confusão.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print('Matriz de confusão sem normalização')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(reports_dir+'/confusion_matrix/confusion_matrix_inceptionv3.jpg')

# ----------------------------------------------------------------
# plotar a figura referente a curva ROC média
# ----------------------------------------------------------------
plt.figure()
plot_confusion_matrix(mean_cm, classes)
#plt.show()

fig, ax = plt.subplots(figsize=(12,4))
plt.plot(mean_fpr, mean_tpr, color='darkorange', linewidth=2, label='InceptionV3')
plt.plot([0, 1.05],[0, 1.05], color='navy', linestyle='--')
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('Average ROC Curve (AUC = {:.4f}%)' .format(mean_roc_auc), fontsize=14)
plt.savefig(reports_dir+'/roc/average_roc_curve_inceptionv3.jpg')

# ----------------------------------------------------------------
# Carregando os modelos em uma lista
# ----------------------------------------------------------------
models = []
for i in range(k):
    model = load_model(models_dir+'/inceptionv3_fold_'+str(i)+'.h5')
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
final_model = load_model(models_dir+'/inceptionv3_fold_0.h5')
final_model.set_weights(avg_weights)

# ----------------------------------------------------------------
# Salvando o modelo final
# ----------------------------------------------------------------
final_model.save(models_dir+'/final_model_inceptionv3.h5')
