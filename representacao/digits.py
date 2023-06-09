## script para extrair Dummy Features da base de digitos manuscritos
## As imagens sao normalizadas no tamanho indicado nas variavies X e Y
## Aprendizagem de Maquina, Prof. Luiz Eduardo S. Oliveira
##
##
import sys
import cv2
import os
import numpy as np
import random

def load_images(path_images, fout, X, Y, use_hog=False):
    print ('Loading images...')
    archives = os.listdir(path_images)
    images = []
    arq = open('digits/files.txt')
    lines = arq.readlines()
    print ('Extracting dummy features')
    for line in lines:
        aux = line.split('/')[1]
        image_name = aux.split(' ')[0]
        label = line.split(' ')[1]
        label = label.split('\n')
        
        for archive in archives:
            if archive == image_name:
                image = cv2.imread(path_images +'/'+ archive, 0)
                if use_hog:
                    feature_extractor(image, label[0], fout, X, Y)
                else:
                    rawpixel(image, label[0], fout, X, Y)
                
                #images.append((image, label))

    print ('Done!')
    return images

def correct_skew(img, delta=1, limit=5):
    """
    Corrige a inclinação da imagem\n
    Modificado de https://stackoverflow.com/questions/57964634/python-opencv-skew-correction-for-ocr
    """

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    best_score = 0.0
    corrected = None
    best_angle = 0.0
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        data = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)

        if score > best_score:
            best_score = score
            corrected = data
            best_angle = angle

    return corrected, best_angle

def feature_extractor(image, label, fout, X, Y):
    image = cv2.resize(image, (X,Y) )
    retval, thrsh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    if not retval:
        return

    rotated, _ = correct_skew(thrsh)

    winSize = (X,Y)     # 24 24
    cellSize = (6,6)    # 6 6
    blockSize = (cellSize[0]*2, cellSize[1]*2)
    blockStride = cellSize
    nbins = 9
    signedGradients = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride, cellSize, nbins, 1, -1, 0, 0.2, 1, 64, signedGradients)
    
    descriptor = hog.compute(rotated)
    
    fout.write(str(label) +  " ")
    for d in descriptor:
        fout.write(str(d)+" ")

    fout.write("\n")

#########################################################
# Usa o valor dos pixels como caracteristica
#
#########################################################
        
def rawpixel(image, label, fout, X, Y):
    image = cv2.resize(image, (X,Y) )
        
    indice = 0
    fout.write(str(label) +  " ")
    for i in range(Y):
        #vet= []
        for j in range(X):
            if( image[i][j] > 128):
                v = 0
            else:
                v = 1    
            #vet.append(v)        
        
            fout.write(str(v)+" ")
            indice = indice+1

    fout.write("\n")

if __name__ == "__main__":
        
    if len(sys.argv) < 4:
        sys.exit("digits.py fname X Y <-h>")

    fout = open(sys.argv[1],"w")

    X = int(sys.argv[2])
    Y = int(sys.argv[3])
    print (X,Y)
        
    use_hog = "-h" in sys.argv
    if use_hog:
        print("Using HOG as descriptor...")

    images = load_images('digits/data', fout, X,Y, use_hog)

    fout.close()
