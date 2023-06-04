# Gabriel de Oliveira Pontarolo GRR20203895
import cv2
import numpy as np
from sys import argv
from os import listdir
from scipy.signal import find_peaks

DIR_NAME = "tr"
ACCEPTED_FORMATS = ["png", "jpg"]
RESIZE_FACTOR = 4 

def correct_skew(img, delta=1, limit=5):
    """
    Corrige a inclinação da imagem\n
    Modificado de https://stackoverflow.com/questions/57964634/python-opencv-skew-correction-for-ocr
    """

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    best_score = 0.0
    corrected = None
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        data = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)

        if score > best_score:
            best_score = score
            corrected = data

    return corrected

def pre_processing(img):
    """
    Diminui o tamanho da imagem e converte para grayscale\n
    """
    (oh, ow) = img.shape[:2]
    img = cv2.resize(img, (ow//RESIZE_FACTOR, oh//RESIZE_FACTOR), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    return thresh

def centralize_letter(img):
    """
    Corta a imagem para centralizar a escrita\n
    """
    # unifica as linhas
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    dilatated = cv2.dilate(img, ver_kernel, iterations=5)

    # encontra o maior contorno
    contours, _ = cv2.findContours(dilatated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    return img[y:y+h, x:x+w]

def line_dilatation(img):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    diff = cv2.absdiff(opening, closing)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)

    return closing

    # aumenta a largura das linhas juntando as palavras
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1), anchor=(9, 0))

    # diminui a altura das linhas
    mask = cv2.dilate(img, hor_kernel, iterations=1)
    kernel = np.ones((5,5), np.uint8)

    return cv2.erode(mask, kernel, iterations=1)

def count_lines(img_paths):

    correct_imgs = 0
    for img_path in img_paths:

        # separa nome e numero de linhas
        actual_line_num = int(img_path.split("_")[-1].split(".")[0])
        img_name = img_path.split("/")[-1].split("_")[0]

        # pre processamento
        img = cv2.imread(img_path)
        thresh = pre_processing(img)

        # corrigi a inclinacao da escrita
        rotated = correct_skew(thresh)

        # corta a imagem para centralizar a carta
        crop = centralize_letter(rotated)

        dilatated = line_dilatation(crop)

        # calcula linhas pelos picos da projecao vertical
        projection = np.sum(dilatated, axis=1, dtype=np.uint32) / 255
        projection[projection < 10] = 0

        # encontra a distancia media esperada entre os picos
        (h, w) = dilatated.shape[:2]
        countors, _ = cv2.findContours(dilatated[0:h, 0:w//5], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moments = [cv2.moments(c) for c in countors]
        centroids = [(int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) for m in moments]
        centroids.sort(key=lambda x: x[1])
        mean_dist = np.mean([centroids[i+1][1] - centroids[i][1] for i in range(len(centroids)-1)])
        peaks = find_peaks(projection, distance=mean_dist, height=10)
        lines_num = len(peaks[0])

        d = cv2.cvtColor(dilatated[0:h, 0:w//5], cv2.COLOR_GRAY2BGR)
        for c in centroids:
            cv2.circle(d, c, 2, (0, 0, 255), 2)

        cv2.imshow("d", d)
        cv2.waitKey(0)

        if lines_num == actual_line_num:
            correct_imgs += 1
        print(f"{img_name} {actual_line_num}  {lines_num}")

        (h, w) = dilatated.shape[:2]
        projection_img = np.zeros((projection.shape[0], w, 3), dtype=np.uint8)
        for i in range(len(projection)):
            cv2.line(projection_img, (0, i), (int(projection[i]), i), (255, 255, 255), 1)
        
        for i in peaks[0]:
            cv2.line(projection_img, (0, i), (int(projection[i]), i), (0, 0, 255), 1)
        
        # mask_out = cv2.cvtColor(dilatated, cv2.COLOR_GRAY2BGR)
        # show = np.concatenate([mask_out, projection_img], axis=1)
        # cv2.imshow("img", show)
        # cv2.waitKey(0)

    print(f"Cartas corretas: {correct_imgs} {len(img_paths)}")

def count_words(img_paths):
    for img_path in img_paths:
        img_name = img_path.split("/")[-1]

        # pre processamento
        img = cv2.imread(img_path)
        (oh, ow) = img.shape[:2]
        img = cv2.resize(img, (ow//RESIZE_FACTOR, oh//RESIZE_FACTOR), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)

        # tenta aproximar as letras 
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
        mask = cv2.dilate(mask, hor_kernel, iterations=1)

        # conta os contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            size = cv2.contourArea(cnt)
            if size > 30:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)

        print(f"{img_name.split('_')[0]} {len(contours)} palavras encontradas")

        cv2.imwrite(f"output_{img_name}", img)

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