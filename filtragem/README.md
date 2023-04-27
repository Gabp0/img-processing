# CI1394 - Exercício Filtragem de Imagens

* Aplica ruído em uma imagem de entrada e faz a filtragem utilizando diferentes filtros de suavização.
* Além do arquivo principal *filtros.py*, acompanha mais dois executáveis utilizados nos experimentos descritos em *relatorio.pdf*
  * *comparator.py* realiza a filtragem com diferentes parâmetros para cada um dos filtros e salva em arquivos **.csv**.
  * *plot.py* recebe como entrada um dos arquivos **.csv** gerados e gera uma imagem **.png** contendo um gráfico dos valores.

## Execução

* *python3 filtros.py {imagem de entrada} {nível de ruído}*
* O nível de ruído deve ser um decimal entre 0 e 1.

## Saída

* Escreve na saída padrão os valores de PSNR máximo e minimo, comparando a imagem original com ela mesma e com a versão com ruído respectivamente, e o resultado da aplicacação de cada um dos filtros.
* Gera 6 arquivos **.png**, sendo um a imagem com ruído (**noise.png**) e os outros 5 (**bilateral.png**, **blur.png**, **gauss.png**, **median.png**, **stacking.png**) os resultados das respectivas filtragens.

## Autor
* Gabriel de Oliveira Pontarolo, GRR20203895, gop20