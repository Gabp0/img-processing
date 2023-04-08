# GRR20203895 Gabriel de Oliveira Pontarolo
import cv2
from sys import argv
from os import listdir
import numpy as np
from dataclasses import dataclass

ACCEPTED_FORMATS = ["jpg", "png", "jpeg", "bmp"]

@dataclass
class ComparisonMethod:
    name:str
    id:int
    reverse:bool

@dataclass
class Histogram:
    id:int
    path:str
    hist:np.ndarray
    imgClass:str

methods = [
    ComparisonMethod("Correlation", cv2.HISTCMP_CORREL, True),
    ComparisonMethod("Interssection", cv2.HISTCMP_INTERSECT, True),
    ComparisonMethod("Chi-Square", cv2.HISTCMP_CHISQR, False),
    ComparisonMethod("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA, False),
]

comp_mat = None

def analise_image(img:Histogram, comp_imgs:list):
    # compara a imagem img com todas as outas
    global methods
    global comp_mat

    print(f"\nImagem atual: {img.path} ({img.imgClass})")
    for method in methods:
        errs = []
        for comp_img in comp_imgs:

            if comp_img.path == img.path:
                continue
            
            # evita comparar as mesmas imagems duas vezes
            # ordem das imagens importa apenas no chi-square  
            # ver https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
            curr_comp = comp_mat[img.id][comp_img.id][method.id]
            if curr_comp != np.NINF and method.id != cv2.HISTCMP_CHISQR:
                err = curr_comp 
            else :
                err = cv2.compareHist(img.hist, comp_img.hist, method.id)
                comp_mat[img.id][comp_img.id][method.id] = err
                comp_mat[comp_img.id][img.id][method.id] = err


            errs.append((comp_img.path, comp_img.imgClass, err))

        # ordena os erros de acordo com o método
        errs.sort(key=lambda x:x[2], reverse=method.reverse)

        # imprime os resultados
        print(f"Método: {method.name}")
        print("Match" if (errs[0][1] == img.imgClass) else "Error")
        print("Top 5:")
        [print(e) for e in errs[0:5]]

def genHist(img_path, id):
    # gera os histogramas e a classe da imagem
    img = cv2.imread(img_path)
    img_hist = cv2.calcHist([img],[0, 1, 2],None,[256, 256, 256],[0,256,0,256,0,256])
    img_class = img_path.split("/")[-1][0]
    return Histogram(id, img_path, img_hist, img_class)

def main():
    dir_name = "./imgs"
    if len(argv) > 1:
        dir_name = argv[1]
        if dir_name[0] != "/": # se o caminho nao for absoluto
            dir_name = "./" + dir_name

    # caminho das imagens
    imgs_paths = [f"{dir_name}/{f}" for f in listdir(dir_name) if f.split(".")[-1] in ACCEPTED_FORMATS]
    if len(imgs_paths) <= 0:
        print("Nenhuma imagem encontrada")
        return 1
    else:
        print(f"{len(imgs_paths)} imagens encontradas")

    global comp_mat
    comp_mat = np.full((len(imgs_paths), len(imgs_paths), len(methods)), np.NINF)

    # gera os histogramas
    id = 0
    hists = []
    for img_path in imgs_paths:
        hists.append(genHist(img_path, id))
        id += 1

    # compara as imagens
    for hist in hists:
        analise_image(hist, hists)
    
    return 0

if __name__ == "__main__":
    main()