# GRR20203895 Gabriel de Oliveira Pontarolo
import cv2
import numpy as np
from sys import argv
from matplotlib import pyplot as plt

def filter_pipeline(img, functions:list[tuple]):
    """Aplica o pipeline de fitlros em *img*"""

    result = img
    for function in functions:
        result = function[0](result, **function[1])

    return result

def frequency_filter(img, maskSize):
    """Filtro de frequencia para remoção de ruido periodico"""

    h,w = img.shape

    # aplica a DFT
    f = np.fft.fft2(img) 
    fshift = np.fft.fftshift(f) 

    # cobre os pontos na imagem
    for i in range(1, 11):
        if i == 5: # nao cobre a regiao branca central
            continue
        fshift[(i*h//10 - maskSize):(i*h//10 + maskSize), (w//2 - maskSize):(w//2 + maskSize)] = 1
    
    # aplica a dft inversa
    f_ishift = np.fft.ifftshift(fshift) # shift zero-frequency component back to origin 
    img_back = np.fft.ifft2(f_ishift) # inverse 2D DFT

    # converte para uint8 novamente
    img_back = np.abs(img_back)
    img_back[img_back > 255] = 255
    img_back[img_back < 0] = 0
    
    return img_back.astype(np.uint8)

def glahe_filter(img, clipLimit):
    """Filtro de GLAHE para melhoria do contraste e bordas"""

    glahe = cv2.createCLAHE(clipLimit=clipLimit)
    return glahe.apply(img)

def unsharp_mask(img):
    """Unsharp masking para realce de bordas
    https://en.wikipedia.org/wiki/Unsharp_masking
    https://stackoverflow.com/questions/32454613/python-unsharp-mask"""

    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(img, 2.0, gaussian, -1.0, 0)

    return unsharp_image

    
def main():
    if len(argv) != 3:
        print("Uso: python3 " + argv[0] + " <imagem de entrada> <imagem de saida>")
        exit(1)

    input_img = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
    output_filename = argv[2]

    # aplica os filtros para melhoria da visualizacao da placa
    result = filter_pipeline(
        input_img,
        [
            (frequency_filter, {"maskSize" : 3}),
            (cv2.medianBlur, {"ksize" : 3}),
            (glahe_filter, {"clipLimit" : 9}),
            (unsharp_mask, {}),
        ]
    )   

    # tentativa de fazer a segmentacao da placa
    ret, mask = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 3)
    result = cv2.bitwise_and(result, result, mask=mask)

    cv2.imwrite(output_filename, result)

if __name__ == "__main__":
    main()