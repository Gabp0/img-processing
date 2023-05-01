import cv2
import numpy as np
from sys import argv
from matplotlib import pyplot as plt

def filter_pipeline(img, functions:list[tuple]):
    """Aplica o pipeline de funcoes em *img*"""

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

def glahe_filter(img, clipLimit, tileGridSize):
    """Filtro de GLAHE para melhoria do contraste e bordas"""

    glahe = cv2.createCLAHE(clipLimit=clipLimit)
    return glahe.apply(img)
    
def main():
    if len(argv) != 3:
        print("Uso: python3 " + argv[0] + " <imagem de entrada> <imagem de saida>")
        exit(1)

    input_img = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
    h,w = input_img.shape
    output_file = argv[2]

    result = filter_pipeline(
        input_img,
        [
            (frequency_filter, {"maskSize" : 3}),
            (cv2.medianBlur, {"ksize" : 3}),
            #(cv2.GaussianBlur, {"ksize": (5,5), "sigmaX": 0}),
            #(cv2.bilateralFilter, {"d" : 17, "sigmaColor" : 0, "sigmaSpace" : 0}),
            (glahe_filter, {"clipLimit" : 9, "tileGridSize" : (3,3)}),
        ]
    )

    # result = frequency_filter(result)
    # result = cv2.medianBlur(result, 3)
    # result = glahe.apply(result)
    #result = cv2.bilateralFilter(result, 17, 0, 0)
    #result = cv2.Laplacian(result, cv2.CV_8U, ksize=7)
    #result = cv2.inRange(result, 34, 255)
    #result = cv2.bitwise_and(input_img, input_img, mask=mask)
    #result = cv2.equalizeHist(result)

    show = np.concatenate((input_img, result), axis=1)
    cv2.imshow("out", show)
    cv2.waitKey(0)
    cv2.imwrite(output_file, result)


if __name__ == "__main__":
    main()