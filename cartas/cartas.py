# Gabriel de Oliveira Pontarolo GRR20203895
import cv2
import numpy as np
from sys import argv
from os import listdir
from scipy.signal import find_peaks

DIR_NAME = "vl"
ACCEPTED_FORMATS = ["png", "jpg"]
RESIZE_FACTOR = 4 
DISTANCE = 20.5 # valor encontrado empiricamente
HEIGHT_FACTOR = 7
MIN_WORD_SIZE = 15

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

def pre_processing(img):
    """
    Diminui o tamanho da imagem e converte para grayscale\n
    """
    (oh, ow) = img.shape[:2]
    img = cv2.resize(img, (ow//RESIZE_FACTOR, oh//RESIZE_FACTOR), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
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

    return (x, y, w, h), img[y:y+h, x:x+w]

def line_dilatation(img):
    """
    Unifica as palavras para aumentar a largura das linhas
    """
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    mask = cv2.dilate(img, hor_kernel, iterations=1)

    # diminui a altura
    kernel = np.ones((5, 5), np.uint8)

    return cv2.erode(mask, kernel, iterations=1)

def count_lines(img_paths, should_show=False):
    """
    Conta o número de linhas de cada carta\n
    """

    correct_imgs = 0
    for img_path in img_paths:

        # separa nome e numero de linhas
        actual_line_num = int(img_path.split("_")[-1].split(".")[0])
        img_name = img_path.split("/")[-1].split("_")[0]
        img = cv2.imread(img_path)

        thresh = pre_processing(img)
        rotated, _ = correct_skew(thresh)
        _, crop = centralize_letter(rotated)
        dilatated = line_dilatation(crop)

        projection = np.sum(dilatated, axis=1, dtype=np.uint32) / 255
        peaks, _ = find_peaks(projection, height=np.average(projection)//HEIGHT_FACTOR, distance=DISTANCE)
        lines_num = len(peaks)
        if lines_num == actual_line_num:
            correct_imgs += 1
        print(f"{img_name} {actual_line_num}  {lines_num}")

        # imagens para debug
        if should_show:
            (h, w) = dilatated.shape[:2]
            projection_img = np.zeros((projection.shape[0], w, 3), dtype=np.uint8)
            for i in range(len(projection)):
                cv2.line(projection_img, (0, i), (int(projection[i]), i), (255, 255, 255), 1)
            
            for i in peaks:
                cv2.line(projection_img, (0, i), (int(projection[i]), i), (0, 0, 255), 1)
            
            mask_out = cv2.cvtColor(dilatated, cv2.COLOR_GRAY2BGR)
            show = np.concatenate([mask_out, projection_img], axis=1)
            cv2.imshow("img", show)
            cv2.waitKey(0)

    print(f"Cartas corretas: {correct_imgs} {len(img_paths)}")

def separe_words(img):
    """
    Tenta deixar as letras mais próximas e as palavras mais distantes\n
    """

    mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE ,np.ones((3, 3), np.uint8), iterations=1)

    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    horz_opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, hkernel)

    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    vert_opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vkernel)

    diff = cv2.bitwise_and(vert_opening, horz_opening)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    mask = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("diff", mask)
    #cv2.waitKey(0)

    #mask = diff

    return mask

def count_words(img_paths, should_show=False):
    """
    Conta o número de palavras de cada carta\n
    """

    for img_path in img_paths:
        img_name = img_path.split("/")[-1]

        # pre processamento
        img = cv2.imread(img_path)
        thresh = pre_processing(img)

        rotated, angle = correct_skew(thresh)
        (cx, cy, cw, ch), crop = centralize_letter(rotated)

        mask = separe_words(crop)

        # conta os contornos e desenha os retangulos na imagem de saida
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            size = cv2.contourArea(cnt)
            if size > MIN_WORD_SIZE:
                x, y, w, h = cv2.boundingRect(cnt)

                # converte as coordenadas para o tamanho original
                ax = (x + cx) * RESIZE_FACTOR
                ay = (y + cy) * RESIZE_FACTOR
                aw = w * RESIZE_FACTOR
                ah = h * RESIZE_FACTOR

                cv2.rectangle(out, (ax, ay), (ax+aw, ay+ah), (255, 0, 0), 1)
                cv2.rectangle(mask_out, (x, y), (x+w, y+h), (255, 0, 0), 1)

        print(f"{img_name.split('_')[0]} {len(contours)} palavras encontradas")

        # (h, w) = img.shape[:2]
        # #mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
        # mask_out = cv2.resize(mask_out, (w, h), interpolation=cv2.INTER_AREA)

        # if should_show:
        #     cv2.imshow("img", np.concatenate([out, mask_out], axis=1))
        #     cv2.waitKey(0)
        
        cv2.imwrite(f"output_{img_name}", out)

def main():
    if (len(argv) < 2):
        print("Uso: python3 " + argv[0] + " <modo: -l ou -w>")
        exit(1)

    # caminho das imagens
    img_paths = [f"{DIR_NAME}/{f}" for f in listdir(DIR_NAME) if f.split(".")[-1] in ACCEPTED_FORMATS]
    img_paths.sort()
    if len(img_paths) <= 0:
        print("Nenhuma imagem encontrada")
        return 1
    else:
        print(f"{len(img_paths)} imagens encontradas")

    should_show = "-s" in argv

    if "-l" in argv:
        count_lines(img_paths, should_show)
    elif "-w" in argv:
        count_words(img_paths, should_show)
    else:
        print("Modo inválido")


if __name__ == "__main__":
    main()
