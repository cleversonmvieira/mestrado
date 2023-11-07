import csv
import shutil
import os

# definir diretório padrão e diretórios para mover imagens
diretorio_padrao = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/papila'
diretorio_positive = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/papila/positive'
diretorio_negative = 'C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/papila/negative'

# ler arquivo csv
with open('C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/papila/patient_data_os.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # obter nome do arquivo e valor da coluna "Glaucoma"
        filename = row['Filename']+".jpg"
        glaucoma = row['Label']
        
        # definir diretório de destino
        if glaucoma == '0':
            destino = diretorio_negative
        elif glaucoma == '1':
            destino = diretorio_positive
        else:
            print(f'Valor inválido para coluna "Glaucoma": {glaucoma}')
            continue
        
        # mover arquivo para diretório de destino
        origem = os.path.join(diretorio_padrao, filename)
        shutil.move(origem, destino)
        print(f'Arquivo {filename} movido para {destino}')
