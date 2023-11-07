import pandas as pd
import json

# Carregando o arquivo JSON
with open('C:/Users/cleverson.vieira/Desktop/projeto-glaucoma_bkp_25042023/projeto-glaucoma/data/refuge/index1.json', 'r') as json_file:
    data = json.load(json_file)

# Criando uma lista de dicionários com os dados
dados = []
for item in data:
    dados.append({
        'ID': item['ID'],
        'ImgName': item['ImgName'],
        'Label': item['Label']
    })

# Criando um DataFrame a partir da lista de dicionários
df = pd.DataFrame(dados)

# Exibindo a tabela
print(df)
