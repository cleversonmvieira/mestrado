import cv2
import pandas as pd
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
#import seaborn as sns
from typing import Tuple
from sklearn.metrics import confusion_matrix

def salvarImagem(path,img,imgName,ext,pasta):
    cv2.imwrite(path+'processed/'+pasta+'/'+imgName+'.'+ext,img)

def medianBlurCustom(img,x,r):
    for i in range(0,r):
        img = cv2.medianBlur(img,x)
    return img

def toCSV(file,values,mode):
    df = pd.DataFrame(values)    
    df.to_csv(file+'.csv',index=False,mode=mode,header=False)

def histogram_equalization(img):
    
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #rgb_img = cv2.imread(image_path)

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    #cv2.imshow('equalized_img', equalized_img)
    return equalized_img
def gaussianCustom(img,x,r):
    for i in range(0,r):
        img = cv2.GaussianBlur(img,(x,x),cv2.BORDER_CONSTANT)
    return img

def equalize(path,imgName,ext):
    def contapixel(img):
        r = 0
        g = 0
        b = 0
        a,l,c = img.shape
        for i in range(0,a):
            for j in range(0,l):
                r += img[i][j][2]
                g += img[i][j][1]
                b += img[i][j][0]
        print('RG: ',r/g)
        print('RB: ',r/b)
        print('GB: ',g/b)
        print('R: ',r/(a*l))
        print('G: ',g/(a*l))
        print('B: ',b/(a*l))
        return r/(a*l)
    #contapixel()
    #return
    img = cv2.imread(path+'processed/isolated_nerve/'+imgName+'.'+ext)
    a,l,c = img.shape
    r = contapixel(img)
    #md,ml,ld,ll = cv2.minMaxLoc(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    R = 205    
    f = R / r
    for i in range(0,a):
        for j in range(0,l):
            aux = int(img[i][j][0]*f)
            if aux > 255: aux=250            
            img[i][j][0] = aux

            aux = int(aux*f)
            if aux > 255: aux=250
            img[i][j][1] = aux
            
            aux = int(img[i][j][2]*f)
            if aux > 255: aux=250
            img[i][j][2] = aux
    salvarImagem(path,img,imgName,'isolated_nerve')

def cross_val_predict(model, kfold : KFold, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:

    model_ = cp.deepcopy(model)
    
    no_classes = len(np.unique(y))
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba

def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):

    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    
    plt.figure(figsize=(12.8,6))
    #sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

    plt.show()

def confusion_matrix1(actual_classes : np.array, predicted_classes : np.array):
    matrix = confusion_matrix(actual_classes, predicted_classes)
    return matrix

def reorganizaMatrix(matrix):
    p = 0
    n = 0
    fp = 0
    fn = 0
    for i in range(0,len(matrix)):
        for j in range(0,matrix[i].size):
            if i == 1 and j == 1 :
                p = matrix[i][j]
            else: 
                if i == 1 and j != 1 :
                    fp += matrix[i][j]
                else:
                    if j == 1 and i != 1:
                        fn += matrix[i][j]
                    else:
                        n += matrix[i][j]
    matrix = [[n,fn]]+[[fp,p]]
    return pd.DataFrame(matrix)


