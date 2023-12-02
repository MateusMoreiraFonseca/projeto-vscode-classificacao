# Projeto de Classificação de Imagens

## Título do Projeto

Classificação de Imagens usando Hu Moments e Naive Bayes

## Equipe

Mateus Moreira Fonseca (1426885)

## Descrição Geral do Projeto
Este projeto tem como objetivo realizar a classificação de radiografias pulmonares, distinguindo entre casos normais e pacientes afetados pela COVID-19. Foram implementados diversos classificadores, como Multilayer Perceptron (MLP), Random Forest (RF), Support Vector Machine (SVM), e, recentemente, Naive Bayes (NB). A análise se baseia em dois descritores diferentes: o grayHistogramMoments, centrado nas características de cor, e o huMoments, que enfoca os momentos invariantes de Hu relacionados à forma dos pulmões.

## Descrição do Descritor implementado:

Neste projeto, utilizamos o descritor Hu Moments, uma técnica poderosa para a extração de características de imagens. Os Hu Moments são uma série de sete números calculados usando momentos centrais que são invariantes à escala, rotação e translação da imagem. Isso significa que, independentemente do tamanho, orientação ou posição do objeto na imagem, os Hu Moments permanecem os mesmos.

Essa propriedade os torna extremamente úteis na representação da forma de objetos em imagens, pois eles podem descrever a forma de um objeto de maneira única e concisa. Isso é particularmente útil em tarefas de reconhecimento de formas e padrões, onde a forma de um objeto é a característica mais discriminante.

Os Hu Moments são calculados a partir dos momentos de imagem, que são uma média ponderada dos intensidades dos pixels da imagem. Os momentos de imagem capturam informações básicas sobre a distribuição de intensidade na imagem, como a área total ( momento zero), o centro de massa (primeiro momento) e a orientação (segundo momento). Os Hu Moments são então derivados desses momentos de imagem através de uma série de transformações.

Em resumo, os Hu Moments são uma ferramenta poderosa e versátil para a extração de características de imagens, capaz de capturar a forma essencial de um objeto enquanto ignoram variações irrelevantes. No contexto deste projeto, eles são usados para extrair características de imagens que são então usadas para treinar um classificador Naive Bayes. O classificador é então capaz de classificar novas imagens com base nessas características.

## Descrição do Classificador implementado:

Foi incrementado o classificador de Naive Bayes, uma técnica de aprendizado de máquina que se baseia no teorema de Bayes. O princípio fundamental do Naive Bayes é a suposição de independência condicional entre as características, o que simplifica consideravelmente o processo de aprendizado e predição.

Em termos mais simples, o Naive Bayes opera calculando a probabilidade de uma instância pertencer a uma determinada classe com base nas probabilidades das características observadas. A suposição "naive" (ingênuo) de independência condicional significa que o impacto de uma característica na classificação é considerado de forma isolada, sem levar em conta a presença ou ausência de outras características.

Esse classificador é eficaz para problemas de classificação simples e rápidos, sendo especialmente adequado para conjuntos de dados menores.

## Repositório do Projeto

O código fonte e outros recursos relacionados ao projeto podem ser encontrados em nosso repositório do GitHub.
[https://github.com/MateusMoreiraFonseca/projeto-vscode-classificacao/tree/main]

## Classificador e Acurácia

Implementamos um novo classificador, o método Naive Bayes (NB), para realizar a classificação das imagens.

O projeto ja comtemplava 3 classificadores :
-Multilayer Perceptron (MLP)
-Random Forest (RF)
-Support Vector Machine (SVM)

Através do descritor grayHistogramMoments (classificador em escalas de cinza) as accuracias foram:
-MPL: 87,50%
-RF: 94,64%
-SVM:89,29%
-NB:83,93%

Através do descritor huMomentes (classificador do momento invariante de Hu) as accuracias foram:
-MPL: 50,00%
-RF: 53,57%
-SVM:58,93%
-NB:60,71%

## Conslusao

A análise comparativa dos descritores e classificadores revelou uma tendência clara em relação ao desempenho superior obtido com o descritor grayHistogramMoments. Esse resultado indica que a coloração e a distribuição de tons de cinza nas radiografias pulmonares fornecem informações mais distintivas para a classificação entre casos normais e afetados pela COVID-19. Por outro lado, o descritor de momento invariante de Hu (LU) baseado em formas apresentou um desempenho inferior, sugerindo que as características estritas de forma podem não ser tão relevantes para essa tarefa específica. A conclusão derivada desses resultados é que a condição de COVID-19 parece estar mais relacionada a alterações na coloração e textura dos pulmões do que a variações em sua forma. A importância da escolha do descritor destaca-se como um ponto crucial, sendo essencial considerar a natureza específica dos dados para capturar efetivamente as informações relevantes para a classificação de radiografias pulmonares.

## Requisitos

- **VSCode**
- **Miniconda3 ou Anaconda** (Indispensável devido às bibliotecas de dependências, além de já conterem o Python)

## Instruções de Uso

1. Faça o download e instale o gerenciador de pacotes Miniconda3 ou Anaconda.

2. Clone o repositório para o seu ambiente local.

   ```bash
   git clone https://github.com/MateusMoreiraFonseca/projeto-vscode-classificacao.git
   ```

3. Abra o prompt de comando Anaconda PowerShell ou Miniconda PowerShell.

4. Crie um ambiente Conda com Miniconda3 ou Anaconda (gerenciadores de pacotes):
   Abra o prompt de comando Anaconda PowerShell ou Miniconda PowerShell.

   ```bash
   conda create -name "nomedoAmbiente"  # Sugestão: procimgCOVID
   ```

5. Ative ou acesse o ambiente.

   ```bash
   conda activate "nomedoAmbiente"
   ```

6. Instale as dependências necessárias, ainda no terminal Conda PowerShell:

   ```bash
   pip install split-folders
   pip install numpy
   pip install scikit-learn
   pip install progress
   pip install matplotlib
   ```

7. Acesse o repositório onde o projeto foi clonado por meio do VSCode.

## Antes de Executar

1. Altere os caminhos das variáveis `input_path` e `output_path` dentro do arquivo "data_splitting.py".
2. Altere os caminhos das variáveis `trainImagePath`, `testImagePath`, `trainFeaturePath` e `testFeaturePath` dentro do arquivo "grayHistogram_FeatureExtraction.py".
3. Altere os caminhos das variáveis `trainImagePath`, `testImagePath`, `trainFeaturePath` e `testFeaturePath` dentro do arquivo "huMoments_FeatureExtraction.py".
4. Altere os caminhos das variáveis `trainFeaturePath` e `testFeaturePath` dentro de cada classificador "nomedoClassificador" + _classifier.py.
5. Altere os caminhos de saída dos classificadores dentro de cada classificador "nomedoClassificador" + _classifier.py; o caminho está originalmente na linha 78 de cada classificador.
6. Selecione o interpretador Python contido no ambiente criado no Anaconda.

## Executando

1. Execute o arquivo run_all_classifiers.py.

Certifique-se de formatar e verificar importacoes ou possíveis erros de denpendencias no código antes da execução.


## Contato

mateus.fonseca1992@gmail.com
