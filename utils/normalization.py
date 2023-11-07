""" 
Parâmetros para cada dataset

Acrima

negative
faux = f[-16:]
positive
faux = f[-18:] 

Origa
negative e positive
faux = f[-7:]

"""


import cv2
from matplotlib import pyplot as plt
from glob import glob
from skimage import io, exposure
import os


def allImagesExecute(path, path_dest):
    files = sorted(glob(path+'*.jpg'))
    #print(files)
    cont = 0
    for f in files:
        #negative
        #faux = f[-16:]
        #positive
        faux = f[-7:]
        print('Processando: ',faux)       
        execute(path, path_dest, faux)
        cont += 1
    print('Imagens processadas: ', cont)


def execute(path, path_dest, imgName):
  img = cv2.imread(path+imgName)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #img_orig = img.copy()
  img_norm = exposure.equalize_adapthist(img, clip_limit=0.01)
  #img_norm2 = exposure.equalize_adapthist(img, clip_limit=0.02)
  #img_norm3= exposure.equalize_adapthist(img, clip_limit=0.03)
  io.imsave(os.path.join(path_dest, imgName), img_norm)
  #plt.imshow(img_norm)    
  #plt.show()
  """ 
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,20))

  ax1.set_title("Input")
  ax1.imshow(img_orig)
  ax1.axis('off')

  ax2.set_title("0.01")
  ax2.imshow(img_norm)
  ax2.axis('off')

  ax3.set_title("0.02")
  ax3.imshow(img_norm2)
  ax3.axis('off')

  ax4.set_title("0.03")
  ax4.imshow(img_norm3)
  ax4.axis('off')

  plt.show()
 """

path_dest = "C:/Users/cleverson.vieira/Desktop/projeto-glaucoma/data/origa/negative_norm/"
path = "C:/Users/cleverson.vieira/Desktop/projeto-glaucoma/data/origa/negative/"
allImagesExecute(path, path_dest)
