# ----------------------------------------------------------------
# Importa os pacotes necessários para rodar a aplicação
# ----------------------------------------------------------------
import os
import random
import shutil

# ----------------------------------------------------------------
# diretório original das imagens e diretório para onde as imagens 
# selecionadas serão movidas
# ----------------------------------------------------------------
original_dir = 'Caminho para o diretório com as imagens (negative/positive)'
selected_dir = 'Caminho para o diretório onde as imagens serão movidas (negative/positive)'

# ----------------------------------------------------------------
# lista de todos os arquivos no diretório original
# ----------------------------------------------------------------
files = os.listdir(original_dir)

# ----------------------------------------------------------------
# número total de arquivos
# ----------------------------------------------------------------
num_files = len(files)

# ----------------------------------------------------------------
# número de arquivos que serão selecionados para mover
# Neste caso, 20% 
# ----------------------------------------------------------------
num_selected = int(num_files * 0.2)

# ----------------------------------------------------------------
# seleciona aleatoriamente os índices dos arquivos a serem movidos
# ----------------------------------------------------------------
selected_indices = random.sample(range(num_files), num_selected)

# ----------------------------------------------------------------
# move os arquivos selecionados para o diretório selecionado
# ----------------------------------------------------------------
for i in selected_indices:
    file_name = files[i]
    file_path = os.path.join(original_dir, file_name)
    shutil.move(file_path, selected_dir)
