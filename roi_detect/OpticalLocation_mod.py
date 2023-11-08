import cv2
import utils_mod
import retinex_mod
import numpy as np

def isolaNervo(path,imgName,ext,rSize):
    img = cv2.imread(path+imgName+"."+ext)
    altura,largura,cores = img.shape    
    if altura < largura :        
        rH = rSize
        rW = int(rSize*largura/altura)
        fw = largura/rW
        fh = altura/rH
    else:        
        rW = rSize
        rH = int(rSize*altura/largura)
        fw = largura/rW
        fh = altura/rH
    img = cv2.resize(img,(rW,rH),interpolation=cv2.INTER_CUBIC)           
    if path != 'acrima/':
        W,w,H,h,nerve = OpticalLocationCenter2Outside(path,imgName,ext,rW,rH)
    else:
        W = largura
        w = 0
        H = altura
        h = 0
    
    img = cv2.imread(path+imgName+"."+ext)    
    img = cv2.resize(img,(rW,rH),interpolation=cv2.INTER_CUBIC)  
    if (w > rW or h > rH):
        w = int(rW//2 - rSize//4)
        W = int(rW//2 + rSize//4)
        h = int(rH//2 - rSize//4)
        H = int(rH//2 + rSize//4)
    a,l,c = img.shape
    img = img[h:H,w:W]
    f = a//2
    a,l,c = img.shape

    img = cv2.resize(img,(300,300),interpolation=cv2.INTER_CUBIC)
    #img = cv2.resize(img,(f,f),interpolation=cv2.INTER_CUBIC)
    an,ln,cn = img.shape
    fnW = l/ln
    fnH = a/an
    utils_mod.salvarImagem(path,img,imgName,ext,'isolated_nerve')
    
    return int(W*fw),int(w*fw),int(H*fh),int(h*fh),fw,fh,nerve,fnW,fnH



def OpticalLocationCenter2Outside(path,imgName,ext,rW,rH):
    img = cv2.imread(path+imgName+"."+ext)
    img = cv2.resize(img,(rW,rH),interpolation=cv2.INTER_CUBIC)  
    imgR = cv2.imread(path+imgName+"."+ext)
    imgR = cv2.resize(imgR,(rW,rH),interpolation=cv2.INTER_CUBIC)  
    imgblk = cv2.imread(path+imgName+"."+ext,cv2.IMREAD_GRAYSCALE)
    imgblk = cv2.resize(imgblk,(rW,rH),interpolation=cv2.INTER_CUBIC)  
    altura,largura,cores = img.shape
    alturaR,larguraR,coresR = imgR.shape
    md,ml,ld,ll = cv2.minMaxLoc(imgblk)  
    xc = int(largura//2)
    yc = int(altura//2)
    busca = 1
    nbusca = 12
    nerve = int(altura / 5.4)
    fa = altura//nbusca
    fl = largura//nbusca
    x,y,X,Y = (1,1,1,1)

    
    while busca <= nbusca and x>0 and y>0 and X<largura and Y < altura:
        fbuscaA = int(fa*busca)
        fbuscaL = int(fl*busca)
        x = xc - fbuscaL
        X = xc + fbuscaL
        y = yc - fbuscaA
        Y = yc + fbuscaA
        
        imgBusca = img[y:Y,x:X]
        #imgBusca = retinex_mod.exRetinexDirecaoTone(imgBusca)
        imgBusca = utils_mod.histogram_equalization(imgBusca)
        imgBusca = utils_mod.medianBlurCustom(imgBusca,9,9)
        imgAux = cv2.cvtColor(imgBusca,cv2.COLOR_BGR2GRAY)
        md,ml,ld,ll = cv2.minMaxLoc(imgAux)
        px = ll[0]
        py = ll[1]
        
        if (np.sum(imgBusca[py][px]) > 700 and np.sum(img[y:Y,x:X][py][px])>300) :
            xf = x + px - nerve
            Xf = x + px + nerve
            yf = y + py - nerve
            Yf = y + py + nerve
            if xf < 0: xf = 0
            if Xf > largura: Xf = largura
            if yf < 0: yf = 0
            if Yf > altura: Yf = altura
            imgBusca = imgR[yf:Yf,xf:Xf]           
            imgAux = imgblk[yf:Yf,xf:Xf]
            md,ml,ld,ll = cv2.minMaxLoc(imgAux)
            px = ll[0]
            py = ll[1]
     
            
            auxX = xf
            auxY = yf
            xf = xf + px - nerve
            yf = yf + py - nerve
            Xf = auxX + px + nerve            
            Yf = auxY + py + nerve            
            if xf < 0: xf = 0
            if Xf > larguraR: Xf = larguraR
            if yf < 0: yf = 0
            if Yf > alturaR: Yf = alturaR            
            return Xf,xf,Yf,yf,nerve
        busca += 1
    return largura,0,altura,0,nerve
    
def maskToJpg(path,imgName):
    #print(path+imgName)
    img = cv2.imread(path+imgName,cv2.COLOR_BGR2GRAY)
    img = img*100
    return img

def recentralizarNervo(path,imgName,ext,rW,rH,Xrec,xrec,Yrec,yrec,W,w,H,h,nerve,fnW,fnH):
    #nerve = int(nerve*0.9)
    img = cv2.imread(path+imgName+"."+ext)    
    a,l,c = img.shape
    a = int(a/rH)
    l = int(l/rW)
    img = cv2.resize(img,(l,a),interpolation=cv2.INTER_CUBIC)
    a,l,c = img.shape
    px = int((Xrec-xrec)//2)
    py = int((Yrec-yrec)//2)    
    x = xrec + px
    y = yrec + px    
    PX = int(x*fnW)+w
    PY = int(y*fnH)+h    
    xf = PX - nerve   
    yf = PY - nerve
    Xf = PX + nerve
    Yf = PY + nerve
    if xf < 0: xf = 0
    if Xf > l: Xf = l
    if yf < 0: yf = 0
    if Yf > a: Yf = a
    #cv2.imshow(imgName+' 1',img)
    
    img = img[yf:Yf,xf:Xf]
    

    f = (a//2)
    img = cv2.resize(img,(f,f),interpolation=cv2.INTER_CUBIC)
    utils_mod.salvarImagem(path,img,imgName,ext,'isolated_nerve')
    
    xf = int(xf*rW)   
    yf = int(yf*rH)
    Xf = int(Xf*rW)
    Yf = int(Yf*rH)
    
    #img2 = maskToJpg(path+'marcacoes-jpg/',imgName+"."+ext)
    #img2 = img2[yf:Yf,xf:Xf]
    #img2 = cv2.resize(img2,(f,f),interpolation=cv2.INTER_CUBIC)
    #utils_mod.salvarImagem(path,img2,imgName,ext,'mat2jpg')
    
    #cv2.imshow(imgName+' 1',img)
    #cv2.imshow(imgName+' 2',img2)
    #cv2.waitKey()
    #print(Xf,xf,Yf,yf)
    return Xf,xf,Yf,yf

def defineDirecao(path,imgName,ext,rW,rw,rH,rh,fw,fh,recW,recw,recH,rech):
    img = cv2.imread(path+imgName+'.'+ext)
    
    a,l,cores = img.shape
    #dir,dif,nW,nw,nH,nh= opticalDirection(img,rW,rw,rH,rh,fw,fh,recW,recw,recH,rech)
    #centro = rw+(rW-rw)/2
    #if centro > l//2:
    #    dir = True
    #    dif = centro - l//2
    #else:
    #    dir = False
    #    dif = l//2 - centro
    img = cv2.imread(path+'processed/isolated_nerve/'+imgName+'.'+ext)
    
    dir = opticalDirectionRedTone(img[rech:recH,recw:recW],imgName)
    
    if (dir):
        v = [1]
        #print(imgName,v)
        #utils_mod.toCSV('direcao.csv',v)
    else:
        v = [0]
        #print(imgName,v)
        #utils_mod.toCSV('direcao.csv',v)
    return v
def opticalDirectionIsolatedNerve(path,imgName,ext):
    img = cv2.imread(path+imgName+'.'+ext)
    altura,largura,cores = img.shape    
    rSize=300
    if altura < largura :        
        rH = rSize
        rW = int(rSize*largura/altura)
        fw = largura/rW
        fh = altura/rH
    else:        
        rW = rSize
        rH = int(rSize*altura/largura)
        fw = largura/rW
        fh = altura/rH
    img = cv2.resize(img,(rW,rH),interpolation=cv2.INTER_CUBIC)   
    img = retinex_mod.exRetinexDirecao(img)
    

def opticalDirection(img,rW,rw,rH,rh,fw,fh,recW,recw,recH,rech):
    a,l,cores = img.shape
    print(fw,fh)
    nw = int(rw+ (recw*fw))
    nh = int(rh+ (rech*fh))
    nW = int(nw + (recW-recw)*fw)
    nH = int(nh + (recH-rech)*fh)
    mN = nw + (nW-nw)//2
    cv2.imshow('imgName',img[nh:nH,nw:nW])
    if (mN < l//2):
        return False,l//2-mN,nW,nw,nH,nh
    else: 
        return True,mN-l//2,nW,nw,nH,nh
    
    cv2.imshow(imgName,img[rh:rH,rw:rW])

def opticalDirectionMacula(path,imgName,ext,nW,nw,nH,nh,s):
    img = cv2.imread(path+imgName+'.'+ext)
    altura,largura,cores = img.shape  
    dw = nW-nw
    dh = int((nH-nh)//2)
    dw *= 2.4

    imgo = img[nh:nH,nw:nW]

    if (s == 1):
        nw = nw - int(dw)
        nW = nW - int(dw)
    else:
        nw = nw + int(dw)
        nW = nW + int(dw)
    
    nh = nh + int(dh)
    nH = nH + int(dh)
    if (nw < 0 or nw > largura):
        return False
    img = img[nh:nH,nw:nW]
    a,l,c = img.shape
    f = 9
    nw = int(nw//f)
    nh = int(nh//f)
    nW = int(nW//f)
    nH = int(nH//f)
    img = cv2.resize(img,(int(l//f),int(a//f)),interpolation=cv2.INTER_CUBIC)
    imgo = cv2.resize(imgo,(int(l//f),int(a//f)),interpolation=cv2.INTER_CUBIC)
    img = retinex_mod.exRetinexDirecao(img)
    a,l,c = img.shape
    dir = False
    img = utils_mod.medianBlurCustom(img,7,200)
    #cv2.imshow(imgName,img)
    aux = 0
    mac = 0
    imgblk = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    md,ml,fd,fl = cv2.minMaxLoc(imgblk)
    limiar = md*2
    ret,imgblk = cv2.threshold(imgblk,limiar,255,cv2.THRESH_BINARY_INV)
    #imgblk = noise.noiseSideRemoveMacula(imgblk,imgName)

    for i in range (0,a):
        for j in range(0,l):
            if (imgblk[i][j] == 255):
            #if (img[i][j][2] <= md and img[i][j][0] >= md and img[i][j][2] <= md ):
                mac += 1 
    
    print(mac,l*a*0.1,l*a*0.3,s)
    cv2.imshow(imgName+' macula'+'-A'+str(s),imgblk)
    if mac > (l*a*0.1) and mac < (l*a*0.3):      
        aux = 1
        if (s == 1):
            dir = True
        else:
            dir = False
    return dir,imgo,aux


def BVR(path,imgName,ext,Xrec,xrec,Yrec,yrec):
    
    
    def getCondicao1(p):
        #c = (p[0] >= ml//1.6 and p[1]<ml//1.5 and p[1]>0 and p[2] > md )
        c = (p == 0)
        return c
    
    img = cv2.imread(path+'processed/retinex/'+imgName+'.'+ext)    
    img = img[yrec:Yrec,xrec:Xrec]

    redE = 0
    redD = 0
    a,l,c = img.shape
    f = 3
    #try:
    #    img = cv2.resize(img,(int(l//f),int(l//f)),interpolation=cv2.INTER_CUBIC)
    #except:
    #    print('continuando.')
    a,l,c = img.shape
    #cv2.imshow(imgName+' Retinex',img)
    img = utils_mod.histogram_equalization(img)  
    #cv2.imshow(imgName+' Histogram',img)
    imgblk = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgblk = utils_mod.medianBlurCustom(imgblk,5,3)
    md,ml,lmd,lml = cv2.minMaxLoc(imgblk)
    md = int(md)
    ml = int(ml)
    redC = 0
    imgblk = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,imgblk = cv2.threshold(imgblk,127,255,cv2.THRESH_BINARY)     
    #cv2.imshow(imgName+'1',imgblk)
    for i in range (a//2-a//5,a//2+a//5):
        for j in range(0,l):
            p = imgblk[i][j]
            if (getCondicao1(p)):
                img[i][j]=255
                redC += 1
    '''
    for i in range (0,a):
        for j in range(l//3+l//3,l):
            p = img[i][j]
            if (getCondicao1(p)):
                #img[i][j]=(0,0,255)
                redC += 1
    '''
    #cv2.imshow(imgName+'2',img)
    #print(md,ml,redC)
    return [round(10*redC/(l*a),2)]




def opticalDirectionRedTone(img,imgName):
    def getCondicao2(p):
        c = p[0] >= ml and p[1] >= ml and p[2] > md
        return c    
    def getCondicao1(p):
        c = (p[0] in range(md,ml) and p[1] in range(md,ml) and p[2] in range(md,ml) )
        return c
    redE = 0
    redD = 0
    a,l,c = img.shape
    f = 5
    try:
        img = cv2.resize(img,(int(l//f),int(l//f)),interpolation=cv2.INTER_CUBIC)
    except:
        print('continuando.')
    a,l,c = img.shape
    img = utils_mod.histogram_equalization(img)  
    imgblk = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgblk = utils_mod.medianBlurCustom(imgblk,5,3)
    md,ml,lmd,lml = cv2.minMaxLoc(imgblk)
    md = int(md)
    ml = int(ml)
    redC = 0
    for i in range (0,l//2):
        for j in range(0,l//2-1):
            p = img[i][j]
            if (getCondicao1(p)):
                redE += 1
                if i > a//2:
                    redC += 1
            else:
                if (getCondicao2(p)):
                    redD += 1
    for i in range (0,l//2):
        for j in range(l//2,l):
            p = img[i][j]
            if (getCondicao1(p)):
                redD += 1
            else:
                if (getCondicao2(p)):
                    redE += 1
    if (redE > redD):
        return False
    if (redD > redE):
        return True

def OpticalDiscLocation(imgColor,imgName):  
    ## - Cria uma cópia da imagem e converte para escala de cinza, após 
    ## - o procedimento, executa-se o medianBlur para ajustar as imagens
    imgblk = cv2.cvtColor(imgColor,cv2.COLOR_BGR2GRAY)
    #cv2.imshow(imgName+'1',imgColor)    
    imgblk = utils_mod.medianBlurCustom(imgblk,9,10)
    imgColor = utils_mod.medianBlurCustom(imgColor,9,5)
    altura, largura = imgblk.shape 
    ok = False
    s = 0.3 #utilizada para determinar o tamanho do corte da imagem a cada passo
    fatorN = altura//2 #utilizado para determinar o tamanho do corte da imagem a cada passo
    nerveSize = int(altura / 4 ) #medida utilizada para cortar a imagem final
    
    fmD,fmL,flD,flL = cv2.minMaxLoc(imgblk)
    x = flL[0]
    y = flL[1]

    if (x + nerveSize< largura): W = x + nerveSize
    else: W = x
    if (x - nerveSize> 0): w = x - nerveSize
    else: w = x
    if (y + nerveSize< altura): H = y + nerveSize
    else: H = y
    if (y - nerveSize> 0): h = y - nerveSize
    else: h = y
    pX = largura #utilizadas como pontos de corte da imagem a cada passada
    pY = altura
    px = 0
    py = 0    
    fx = 1
    fy = 1
    if (altura / largura < 1):
        fx = 2
    else:
        fy = int(altura/largura)
    x = 0
    y = 0
    dist = altura//9 #distância da borda, para auxiliar nas imagens com bordas claras
    ajustNerve = fatorN*s #fator de corte a cada passada
    while (ok == False and x < largura and y < altura):
       
        mD,mL,lD,lL = cv2.minMaxLoc(imgblk[py:pY,px:pX]) #pontos mais claro e mais escuro da imagem cortada         
        a,l = imgblk[py:pY,px:pX].shape
        x = lL[0]+px #posição do ponto mais claro na imagem original
        y = lL[1]+py

        if (x < largura and y < altura):
            
            # - O pixel vermelho da imagem original deverá ser maior que o mais claro da imagem cortada em escala de cinza
            # - juntamente com o pixel azul menor que 230, além verificar a distância da borda
            if (imgColor[y][x][2]>=mL and imgColor[y][x][0]<=240 and lL[0]>dist and lL[1]>dist):                      
                W = x + nerveSize
                w = x - nerveSize
                H = y + nerveSize
                h = y - nerveSize
                if (W > largura): W = largura
                if (w < 0): w = 0
                if (H > altura): H = altura
                if (h < 0): h = 0
                return W,w,H,h
                
            else:
                # - ajusta o próximo corte
                px += int(ajustNerve)*fx
                py += int(ajustNerve)*fy
                pX -= int(ajustNerve)*fx
                pY -= int(ajustNerve)*fy
        
        W = x + nerveSize
        w = x - nerveSize
        H = y + nerveSize
        h = y - nerveSize
        if (W > largura): W = largura
        if (w < 0): w = 0
        if (H > altura): H = altura
        if (h < 0): h = 0
        if (largura//2 - x > 0):
            aux =  largura//2 - x
        else:
            aux = (largura//2 - x) * -1
    return W,w,H,h