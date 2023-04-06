import cv2
from sys import argv
from os import listdir
from matplotlib import pyplot as plt
from collections import namedtuple

ComparisonMethod = namedtuple("ComparisonMehod", ["name", "id", "reverse"])
methods = [
    ComparisonMethod("Correlation", cv2.HISTCMP_CORREL, True),
    ComparisonMethod("Interssection", cv2.HISTCMP_INTERSECT, True),
    ComparisonMethod("Chi-Square", cv2.HISTCMP_CHISQR, False),
    ComparisonMethod("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA, False),
]

def analise_image(img_name, comp_img_names):
    # imagem de entrada
    print("\nCurrent image: " + img_name)
    img = cv2.imread(img_name)
    img_hist = cv2.calcHist([img],[0, 1, 2],None,[256, 256, 256],[0,256,0,256,0,256])
    img_class = img_name.split("/")[-1][0]

    # calcula os histogramas
    comp_hists = []
    for curr_img_name in comp_img_names:
        if curr_img_name == img_name:
            continue

        curr_img = cv2.imread(curr_img_name)
        curr_img_hist = cv2.calcHist([curr_img],[0, 1, 2],None,[256, 256, 256],[0,256,0,256,0,256])
        comp_hists.append((curr_img_name, curr_img_hist, curr_img_name.split("/")[-1][0]))


    for method in methods:
        errs = []
        for comp_hist in comp_hists:
            err  = cv2.compareHist(img_hist, comp_hist[1], method.id)
            errs.append((comp_hist[0], comp_hist[2], err))
    
        errs.sort(key=lambda x:x[2], reverse=method.reverse)
        print(f"Método: {method.name}")
        print("Match" if (errs[0][1] == img_class) else "Error")
        print("Top 5:")
        [print(e) for e in errs[0:5]]

def main():
    dir_name = "./" + argv[1]
    imgs_names = [f"{dir_name}{f}" for f in listdir(dir_name)]

    for img_name in imgs_names:
        analise_image(img_name, imgs_names)

if __name__ == "__main__":
    main()