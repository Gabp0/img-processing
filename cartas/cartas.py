import cv2
import numpy as np
from sys import argv
from os import listdir

DIR_NAME = "tr"
ACCEPTED_FORMATS = ["png", "jpg"]

from scipy.ndimage import interpolation as inter

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def count_lines(img_paths):
    img_paths[0]

    for img_path in img_paths:
        actual_line_num = img_path.split("_")[-1].split(".")[0]
        print(actual_line_num)

        img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        (oh, ow) = img.shape[:2]
        
        img = cv2.resize(img, (ow//4, oh//4), interpolation=cv2.INTER_CUBIC)
        (h, w) = img.shape[:2]

        angle, rotated = correct_skew(img)

        retval, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)

        vertical_pixel_sum = np.sum(thresh, axis=0)
        myprojection = vertical_pixel_sum / 255
        print(myprojection)



        show = np.concatenate([img, rotated], axis=1)
        cv2.imshow("rotated", show)
        cv2.waitKey(0)

def count_words(img_paths):
    return True

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
        print(img_paths)

    if mode == "-l":
        count_lines(img_paths)
    elif mode == "-w":
        count_words(img_paths)
    else:
        print("Mo")
    


if __name__ == "__main__":
    main()


# correcao de inclinacao skew slant
# morf matematica 
# bb nas palavras