# CI1394 - Exercício Representação

* Utiliza um classificador KNN (K-ésimo vizinho mais próximo) para fazer a identificação de dígitos manuscritos.
* O script *digits.py* gera os vetores de características e *knn.py* faz a classificação.

## Execução

* *python3 digits.py {fname} {X} {Y} {<-h>}*
  * Onde **fname** é o nome do arquivo de saída, **X** e **Y** são as dimensões da imagem e **-h** é um argumento opcional que faz o uso da técnica HOG (histograma de gradientes orientados) para a extração de características.
* *python3 knn.py {fname}* 
  * Onde **fname** é o nome do arquivo de características gerado por *digits.py*.

## Saída

* *digits.py* gera um arquivo de saída com o nome especificado contendo os vetores de características de cada imagem.
* *knn.py* gera na saída padrão um relatório com a acurácia do classificador e a matriz de confusão para cada digito.

## Autor
* Gabriel de Oliveira Pontarolo, GRR20203895, gop20