import sys
import numpy as np
import cv2
import csv

NOISE_LEVELS = [0.01, 0.02, 0.05, 0.07, 0.1]

###------------------------------------
### Add Noise with a give probability
###------------------------------------
def sp_noise(image,prob):
        
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

###--------------------------------------------
### Stacks n noise images with a level of noise
###--------------------------------------------
def test_stacking(img, level, n):
    
    # array do tipo uint32 para não dar overflow
    stacked = np.zeros(img.shape, np.uint32)      
    
    for i in range(n):
        # cria imagem com ruído e soma
        noise = sp_noise(img, level)
        stacked += noise
    
    # divide pelo total de imagens pra dar a média
    stacked //= n
    
    res = stacked.astype(np.uint8)

    psnr = cv2.PSNR(img, res)
    print (f"NBimgs = {n}, Noise level = {level}, PSNR = ", psnr)
    
    return psnr


def test_av_blur(img, level, ksize):
    """
    Testa o filtro de média com kernel de tamanho (ksize x ksize), onde
    ksize é ímpar, e retorna o PSNR
    """

    noise_img = sp_noise(img, level)

    ksize = (ksize, ksize)
    blur = cv2.blur(noise_img, ksize)

    psnr = cv2.PSNR(img, blur)
    print (f"Ksize = {ksize}, Noise level = {level}, PSNR = ", psnr)

    return psnr

def test_av_gauss(img, level, ksize):
    """
    Testa o filtro gaussiano com kernel de tamanho (ksize x ksize), onde
    ksize é ímpar, e retorna o PSNR
    """

    noise_img = sp_noise(img, level)

    ksize = (ksize, ksize)
    blur = cv2.GaussianBlur(noise_img, ksize, sigmaX=0, sigmaY=0)

    psnr = cv2.PSNR(img, blur)
    print (f"Ksize = {ksize}, Noise level = {level}, PSNR = ", psnr)

    return psnr

def test_median(img, level, ksize):
    """
    Testa o filtro de mediana com kernel de tamanho (ksize x ksize), onde
    ksize é ímpar, e retorna o PSNR
    """

    noise_img = sp_noise(img, level)

    blur = cv2.medianBlur(noise_img, ksize)

    psnr = cv2.PSNR(img, blur)
    print (f"Ksize = {ksize}, Noise level = {level}, PSNR = ", psnr)

    return psnr

def test_filter(filename, img, kernel_range, filter, row_name):
    """
    Testa um filtro e salva os resultados em um arquivo CSV
    """

    print(f"Testando filtro '{filter.__name__}' com saída para '{filename}'")

    # cria o arquivo de saida e escreve a header
    f = open(filename, "w+", newline='')
    writer = csv.writer(f)
    writer.writerow([row_name] + [f"{level}" for level in NOISE_LEVELS])

    # testa o filtro para cada kernel e nivel de ruido
    for i in kernel_range:
        psnrs = []
        for level in NOISE_LEVELS:
            psnrs.append(filter(img, level, i))
        writer.writerow([i] + psnrs)

def main(argv):
    # ler a imagem
    img = cv2.imread(argv[1], 0)
    psnr = cv2.PSNR(img, img)
    print ('PSNR Max = ', psnr)

    # range para os kernels (ksize x ksize) onde ksize eh impar
    kernel_range = [i for i in range(1, 50) if i % 2 == 1]

    test_filter("stacking.csv", img, range(1, 50), test_stacking, "nb_imgs")
    test_filter("av_blur.csv", img, kernel_range, test_av_blur, "ksize")
    test_filter("av_gauss.csv", img, kernel_range, test_av_gauss, "ksize")
    test_filter("median.csv", img, kernel_range, test_median, "ksize")

if __name__ == '__main__':
    main(sys.argv)


