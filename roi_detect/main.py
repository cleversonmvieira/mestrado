import OpticalLocation_mod
import time
import os

def execute(path,imgName,ext): 

    tempo = False
    
    data = [imgName+'.'+ext]
    rSize = 300
    
    if tempo:
        inicio = time.time()
        print('Início: 0')

    rW,rw,rH,rh,fw,fh,nerve,fnW,fnH = 0,0,0,0,0,0,0,0,0
    if (path != 'rimone/'):
        rW,rw,rH,rh,fw,fh,nerve,fnW,fnH = OpticalLocation_mod.isolaNervo(path,imgName,ext,rSize)
        W = int(rW/fw)
        w = int(rw/fw)
        H = int(rH/fh)
        h = int(rh/fh)       
        if tempo:
            parcial = time.time()    
            print('Optical Location: ',parcial-inicio)    


    
def allImagesExecute(path,ext):
    
    files = os.listdir(path)
    cont = 0
    inicio = time.time()
    for img in files:
        if img.endswith(".jpg"):
            print('Processando: ',img)       
            execute(path,img[:-4],ext)
            cont += 1
    fim = time.time()
    print('Imagens processadas: ', cont)    
    #print('Tempo Médio '+path+': ',(fim-inicio)/cont)
    print('Tempo Total '+path+': ',(fim-inicio))

    

def main():
    
    path = 'beh/positive/'    
    execute(path,'IM000000_L','jpg')
    ext = 'jpg'    
    #allImagesExecute(path,ext)
    
    
main()
#cv2.waitKey(0)