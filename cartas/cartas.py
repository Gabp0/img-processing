import cv2
import numpy as np
from sys import argv
from os import listdir
from scipy.signal import find_peaks

DIR_NAME = "vl" # MUDAR PARA "." ANTES DE ENTREGAR
ACCEPTED_FORMATS = ["png", "jpg"]
RESIZE_FACTOR = 4 
DISTANCE = 20.5
PEAK_HEIGHT_FACTOR = 5

def correct_skew(image, delta=1, limit=5):
    """
    Corrige a inclinação da imagem\n
    Modificado de https://stackoverflow.com/questions/57964634/python-opencv-skew-correction-for-ocr
    """

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    best_score = 0.0
    corrected = None
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        data = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)

        if score > best_score:
            best_score = score
            corrected = data

    return corrected

def count_lines(img_paths):

    correct_imgs = 0
    for img_path in img_paths:

        # separa nome e numero de linhas
        actual_line_num = int(img_path.split("_")[-1].split(".")[0])
        img_name = img_path.split("/")[-1].split("_")[0]

        # pre processamento
        img = cv2.imread(img_path)
        (oh, ow) = img.shape[:2]
        img = cv2.resize(img, (ow//RESIZE_FACTOR, oh//RESIZE_FACTOR), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)

        # corrigi a inclinacao da escrita
        rotated = correct_skew(thresh)

        # remove ruidos e aumenta a largura das palavras
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        mask = cv2.dilate(rotated, hor_kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # calcula linhas pelos picos da projecao vertical
        projection = np.sum(mask, axis=1, dtype=np.uint32) / 255
        peaks = find_peaks(projection, distance=DISTANCE, height=np.average(projection)/PEAK_HEIGHT_FACTOR)
        lines_num = len(peaks[0])

        if lines_num == actual_line_num:
            correct_imgs += 1
        print(f"{img_name} {actual_line_num}  {lines_num}")

    print(f"Cartas corretas: {correct_imgs} {len(img_paths)}")

def count_words(img_paths):
    for img_path in img_paths:

        # pre processamento
        img = cv2.imread(img_path)
        (oh, ow) = img.shape[:2]
        img = cv2.resize(img, (ow//RESIZE_FACTOR, oh//RESIZE_FACTOR), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)

        # junta as letras das palavras
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.dilate(thresh, kernel, iterations=1)

        

        cv2.imshow("mask", mask)
        cv2.waitKey(0)
       

def main():
    if (len(argv) < 2):
        print("Uso: python3 " + argv[0] + " <modo: -l ou -w>")
        exit(1)

    mode = argv[1]

    # caminho das imagens
    img_paths = [f"{DIR_NAME}/{f}" for f in listdir(DIR_NAME) if f.split(".")[-1] in ACCEPTED_FORMATS]
    img_paths.sort()
    if len(img_paths) <= 0:
        print("Nenhuma imagem encontrada")
        return 1
    else:
        print(f"{len(img_paths)} imagens encontradas")

    if mode == "-l":
        count_lines(img_paths)
    elif mode == "-w":
        count_words(img_paths)
    else:
        print("Modo inválido")


if __name__ == "__main__":
    main()


# correcao de inclinacao skew slant
# morf matematica 
# bb nas palavras