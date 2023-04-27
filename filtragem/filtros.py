# GRR20203895 Gabriel de Oliveira Pontarolo
import sys
import numpy as np
import cv2

# melhores parametros dos filtros encontrados nos testes
NB_IMGS = 20
BLUR_KSIZE = 5
GAUSS_KSIZE = 9
MEDIAN_KSIZE = 3
BILATERAL_SIGMA = 180

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
def stacking(img, level):
    
    # array do tipo uint32 para não dar overflow
    stacked = np.zeros(img.shape, np.uint32)      
    
    for i in range(NB_IMGS):
        # cria imagem com ruído e soma
        noise = sp_noise(img, level)
        stacked += noise
    
    # divide pelo total de imagens pra dar a média
    stacked //= NB_IMGS
    
    res = stacked.astype(np.uint8)

    cv2.imwrite("stacking.png", res)
    psnr = cv2.PSNR(img, res)
    print ("Stacking filter PSNR = ", psnr)
    
    return res
        

def test_av_blur(img, level):
    """
    Testa o filtro de média com kernel de tamanho (ksize x ksize), onde
    ksize é ímpar, e retorna o PSNR
    """

    noise_img = sp_noise(img, level)

    ksize = (BLUR_KSIZE, BLUR_KSIZE)
    blur = cv2.blur(noise_img, ksize)

    cv2.imwrite("blur.png", blur)
    psnr = cv2.PSNR(img, blur)
    print (f"Blur filter PSNR = ", psnr)

    return psnr

def test_av_gauss(img, level):
    """
    Testa o filtro gaussiano com kernel de tamanho (ksize x ksize), onde
    ksize é ímpar, e retorna o PSNR
    """

    noise_img = sp_noise(img, level)

    ksize = (GAUSS_KSIZE, GAUSS_KSIZE)
    blur = cv2.GaussianBlur(noise_img, ksize, sigmaX=0, sigmaY=0)

    cv2.imwrite("gauss.png", blur)
    psnr = cv2.PSNR(img, blur)
    print (f"Gaussian blur filter PSNR = ", psnr)

    return psnr

def test_median(img, level):
    """
    Testa o filtro de mediana com kernel de tamanho (ksize x ksize), onde
    ksize é ímpar, e retorna o PSNR
    """

    noise_img = sp_noise(img, level)
    blur = cv2.medianBlur(noise_img, MEDIAN_KSIZE)
    
    cv2.imwrite("median.png", blur)
    psnr = cv2.PSNR(img, blur)
    print (f"Median filter PSNR = ", psnr)

    return psnr

def test_bilateral(img, level):
    """
    Testa o filtro bilateral com sigmaSpace == sigmaColor == sigma
    e retorna o PSNR
    """

    noise_img = sp_noise(img, level)
    blur = cv2.bilateralFilter(noise_img, 9, BILATERAL_SIGMA, BILATERAL_SIGMA)

    cv2.imwrite("bilateral.png", blur)
    psnr = cv2.PSNR(img, blur)
    print (f"Bilateral filter PSNR = ", psnr)

    return psnr

def main(argv):

    if (len(sys.argv)!= 3):
        sys.exit("Use: stack <imageIn> <noise (0.01, 0.02...)>")

    # ler a imagem
    img = cv2.imread(argv[1], 0)
    level = float(argv[2])
    imgNoise = sp_noise (img, level)

    max_psnr = cv2.PSNR(img, img)
    min_psnr = cv2.PSNR(img, imgNoise)
    print (f"PSNR Max = {max_psnr}\nPSNR Min = {min_psnr}")

    test_av_blur(img, level)
    test_av_gauss(img, level)
    test_median(img, level)
    test_bilateral(img, level)
    stacking(img, level)

    cv2.imwrite("noise.png", imgNoise)


if __name__ == '__main__':
    main(sys.argv)


