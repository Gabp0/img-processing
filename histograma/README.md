# CI1394 - Exercício Classificação com Histogramas

* Compara as imagens de um diretório umas com as outras

## Execução

* *python3 hist.py {diretorio de entrada}*
* Caso o argumento não seja fornecido, o diretório de entrada padrão é "./imgs"

## Saída

* Para cada uma das imagens dentro do diretório, escreve na saída padrão **stdout** um relatório no formato:

```console

Imagem atual: {caminho da imagem atual} ({classe da imagem})
Método: Correlation
{Resultado da comparação (Match ou Error)}
Top 5:
({caminho da imagem}, {classe}, {erro})
({caminho da imagem}, {classe}, {erro})
...
Método: Interssection
{Resultado da comparação (Match ou Error)}
Top 5:
({caminho da imagem}, {classe}, {erro})
({caminho da imagem}, {classe}, {erro})
...
Método: Chi-Square
{Resultado da comparação (Match ou Error)}
Top 5:
({caminho da imagem}, {classe}, {erro})
({caminho da imagem}, {classe}, {erro})
...
Método: Bhattacharyya
{Resultado da comparação (Match ou Error)}
Top 5:
({caminho da imagem}, {classe}, {erro})
({caminho da imagem}, {classe}, {erro})
...

...

```

## Autor
* Gabriel de Oliveira Pontarolo, GRR20203895, gop20