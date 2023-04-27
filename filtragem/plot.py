# GRR20203895 Gabriel de Oliveira Pontarolo
import pandas as pd
from matplotlib import pyplot as plt
from sys import argv, exit
import numpy as np

def main():
    if len(argv) != 5:
        print(f"Uso: {argv[0]} <csv_input> <img_output> <plot_title> <plot_spacing>")
        exit(1)

    # le o csv em um dataframe
    df = pd.read_csv(argv[1], index_col=0)
    print(df.to_string())

    # plota o grafico
    df.plot()
    plt.xticks(np.arange(min(df.index), max(df.index), int(argv[4])))
    plt.title(argv[3])

    # salva a img de saida
    fig = plt.gcf()
    fig.savefig(argv[2], dpi=300)

if __name__ == '__main__':
    main()