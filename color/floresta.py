# GRR20203895 Gabriel de Oliveira Pontarolo
import cv2 
import numpy as np
from sys import argv
from math import trunc

CROP_NUM = 10            # divide a imgem em CROP_NUM x CROP_NUM areas
BLUE_THRESHOLD = 85    # valor minimo do hue para ser considerado nao verde

def find_green(img: np.ndarray):
    """Retorna o menor e o maior valor de verde na imagem"""

    h, w, _ = img.shape
    crop_h = h // CROP_NUM
    crop_w = w // CROP_NUM

    avgrs = []
    avgrs_blue = []
    for i in range(CROP_NUM):
        for j in range(CROP_NUM):
            # encontra a media de pixels da area
            crop = img[i * crop_h: (i + 1) * crop_h, j * crop_w: (j + 1) * crop_w]
            curr_avgr = np.average(crop, axis = (0,1))

            # remove areas que nao sao verdes
            if curr_avgr[0] < BLUE_THRESHOLD:
                avgrs.append([trunc(x) for x in curr_avgr])

    return np.array(min(avgrs, key=lambda x: x[0])), np.array(max(avgrs, key=lambda x: x[0]))

def main():
    input_filename = argv[1]
    output_filename = argv[2]

    # le a imagem e converte para HSV
    input = cv2.imread(input_filename, cv2.COLOR_BGR2RGB)
    input_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)

    light_green, dark_green = find_green(input_hsv)

    # upper e lower bounds para o filtro
    # tive melhores resultados usando somente o hue
    # menos para a imagem 6
    light_green = np.array([light_green[0], 0, 0])
    dark_green = np.array([dark_green[0], 255, 255])

    # aplica o filtro
    mask = cv2.inRange(input_hsv, light_green, dark_green)
    result = cv2.bitwise_and(input_hsv, input_hsv, mask=mask)

    # converte de volta para BGR e salva 
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    cv2.imshow("result", result)
    cv2.waitKey(0)

    cv2.imwrite(output_filename, result)

if __name__ == "__main__":
    main()