# GRR20203895 Gabriel de Oliveira Pontarolo
import cv2 
import numpy as np 
from sys import argv

def main():
    # le a imagem da como argumento ou default "img1.png"
    
    img_name = "img1.png"
    if len(argv) > 1:
        img_name = argv[1]

    # le imagem
    try:
        img = cv2.imread(img_name)
    except Exception:
        print("Erro ao ler a imagem " + img_name)
        return

    h,w,c = img.shape

    # cria imagens de saida
    zoom = np.ones((h*2, w*2, c), dtype=np.uint8)
    small = np.ones((h//2, w//2, c), dtype=np.uint8)

    # aumenta a imagem em 2x
    for i in range(h*2):
        for j in range(w*2):
            zoom[i][j][:] = img[i//2][j//2][:]

    # diminui a imagem em 2x
    for i in range(h//2):
        for j in range(w//2):
            pixel = [img[i*2][j*2], img[i*2][j*2 + 1], img[i*2 + 1][j*2], img[i*2 + 1][j*2 + 1]]
            small[i][j] = list(map(lambda *x: sum(x)//4, *pixel))  # soma elemento por elemento de cada canal e divide por 4

    # escreve a saida
    cv2.imwrite("zoom_" + img_name, zoom)
    cv2.imwrite("small_" + img_name, small)

if __name__ == "__main__":
    main()