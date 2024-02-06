import csv
import shutil
import os

# definir diretório padrão e diretórios para mover imagens
diretorio_padrao = 'Caminho do diretório com as imagens'
diretorio_positive = 'Caminho do diretório positive'
diretorio_negative = 'Caminho do diretório negative'

# ler arquivo csv
with open('Caminho do arquivo .csv', 'r') as csvfile:
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
