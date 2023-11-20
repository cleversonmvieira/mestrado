from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
#from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
from keras.callbacks import ModelCheckpoint

# Parâmetros
epochs = 20
resolution = 224
batch_size = 8

# Caminho dos diretórios (treinamento, teste, relatórios)
#train_dir = 'C:/Users/Cleverson M. Vieira/Desktop/projeto-glaucoma/data/acrima/train'
#test_dir = 'C:/Users/Cleverson M. Vieira/Desktop/projeto-glaucoma/data/acrima/test'
#models_dir = 'C:/Users/Cleverson M. Vieira/Desktop/projeto-glaucoma/models'
#reports_dir = 'C:/Users/Cleverson M. Vieira/Desktop/projeto-glaucoma/reports'

train_dir = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma/data/acrima/train'
test_dir = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma/data/acrima/test'
models_dir = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma/models'
reports_dir = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma/reports'

# Criação do modelo DenseNet
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(resolution, resolution, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Congela as camadas da base da VGG19
for layer in base_model.layers:
    layer.trainable = False

# Compilação do modelo
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Criação dos geradores de imagens
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(resolution, resolution), batch_size=batch_size, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(resolution, resolution), batch_size=batch_size, class_mode='binary')

checkpoint = ModelCheckpoint(models_dir+'/checkpoints/best_densenetnet.h5', monitor="val_acc", verbose=1, save_best_only=True, mode="auto", save_freq="epoch")

print('')
print('----------------------------------------')
print('Iniciando treinamento')
print('----------------------------------------')

# Treinamento do modelo
hist = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=epochs, validation_data=test_generator, validation_steps=len(test_generator), callbacks=checkpoint)
#hist = model.fit(train_generator, steps_per_epoch=1, epochs=epochs, validation_data=test_generator, validation_steps=1)

print('')
print('----------------------------------------')
print('Treinamento finalizado')
print('----------------------------------------')

print('')
print('----------------------------------------')
print('Salvando modelo')
print('----------------------------------------')


# Salvando o modelo
model.save(models_dir+'/densenet.h5')

model_h5 = os.listdir(models_dir)

for file in model_h5:
    if(file[:-3] == "densenet"):
        print('')
        print('----------------------------------------')
        print('Modelo salvo com sucesso')
        print('----------------------------------------')
        

print('')
print('----------------------------------------')
print('Iniciando a avaliação do modelo')
print('----------------------------------------')

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
#test_loss, test_acc = model.evaluate(test_generator, steps=1, verbose=1)

print('')
print('Acurácia do modelo:', test_acc)
print('Perda do modelo:', test_loss)


#print('')
#print('----------------------------------------')
#print('Previsões do modelo e matriz de confusão')
#print('----------------------------------------')


# Previsões do modelo
#y_pred = model.predict(test_generator, steps=len(test_generator)).round()
 
#y_pred = model.predict(test_generator)

# Criação da matriz de confusão
#cm = confusion_matrix(test_generator.classes, y_pred)
#print(cm)

# Plot da matriz de confusão
#plt.imshow(cm, cmap=plt.cm.Blues)
#plt.colorbar()
#plt.xticks(range(2), ['Glaucoma', 'Normal'])
#plt.yticks(range(2), ['Glaucoma', 'Normal'])
#plt.xlabel('Previsão')
#plt.ylabel('Real')
#plt.show()


print('')
print('----------------------------------------')
print('Salvando os gráficos de Acurácia e Perda')
print('----------------------------------------')

# Plotando dados de acurácia
hist = hist.history
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])
plt.title("Accuracy plot")
plt.legend(["train","test"])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig(reports_dir+'/acc_densenet.jpg')
plt.show()

# Plotando dados de perdas
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])
plt.title("Accuracy loss")
plt.legend(["train","test"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(reports_dir+'/loss_densenet.jpg')
plt.show()

reports_file = os.listdir(reports_dir)

for file_report in reports_file:
    if(file_report[:-4] == "acc_densenet" and file_report[:-4] == "loss_densenet"):
        print('')
        print('----------------------------------------')
        print('Gráficos salvo com sucesso')
        print('----------------------------------------')
