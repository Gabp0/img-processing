# GRR20203895 Gabriel de Oliveira Pontarolo
import cv2 
import numpy as np
from sys import argv
from math import trunc

# divide a imgem em CROP_NUM x CROP_NUM areas
CROP_NUM = 8            

# intervalo do verde do hue (para a maioria das imagens)
GREEN_UPPER = 90    
GREEN_LOWER = 30

def get_bounds(img: np.ndarray):
    """Retorna o menor e o maior valor de verde na imagem"""

    h, w, _ = img.shape
    crop_h = h // CROP_NUM
    crop_w = w // CROP_NUM

    opt_lower = np.array([180, 255, 255], dtype=np.uint8)
    opt_upper = np.array([0, 0, 0], dtype=np.uint8)
    for i in range(CROP_NUM):
        for j in range(CROP_NUM):
            # encontra a media de pixels da area
            crop = img[i * crop_h: (i + 1) * crop_h, j * crop_w: (j + 1) * crop_w]
            curr_avgr = [trunc(x) for x in np.average(crop, axis = (0,1))]

            # se a media estiver no intervalo do verde, atualiza os limites
            if GREEN_LOWER < curr_avgr[0] < GREEN_UPPER:
                opt_lower = np.minimum(opt_lower, curr_avgr)
                opt_upper = np.maximum(opt_upper, curr_avgr)
            
    return opt_lower, opt_upper

def main():
    input_filename = argv[1]
    output_filename = argv[2]

    # le a imagem e converte para HSV
    input = cv2.imread(input_filename, cv2.COLOR_BGR2RGB)
    input_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)

    light_green, dark_green = get_bounds(input_hsv)

    # upper e lower bounds para o filtro
    # tive melhores resultados usando somente o hue
    light_green = np.array([light_green[0], 0, 0])
    dark_green = np.array([dark_green[0], 255, 255])

    # aplica o filtro
    mask = cv2.inRange(input_hsv, light_green, dark_green)
    result = cv2.bitwise_and(input_hsv, input_hsv, mask=mask)

    # converte de volta para BGR e salva 
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_filename, result)

if __name__ == "__main__":
    main()