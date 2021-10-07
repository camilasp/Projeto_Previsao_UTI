# Machine Learning na avaliação de pacientes com COVID-19: construção de um modelo para prever a necessidade de internação em UTI.

<img src="https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/imagens/coronavirus-4972480_640.jpg" width="1000" height="500">

Esse projeto foi realizado durante o bootcamp de data science aplicado da Alura, utilizando como base um desafio proposto pelo Hospital Sírio Libanês
no [Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19).

Os seguintes notebooks fazem parte desse Projeto:

1. [Notebook Principal](https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/Machine_learning_na_avaliacao_pacientes_com_COVID_19.ipynb)
2. [Preparação dos dados](https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/preparacao_dados.ipynb)
3. [Modelos de Machine Learning](https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/Modelos.ipynb)
4. [Gráficos dados SEADE](https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/dados_seade.ipynb)

Além de um arquivo auxiliar com funções:

[Funções](https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/funcoes.py)



Comece pelo [notebook principal](https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/Machine_learning_na_avaliacao_pacientes_com_COVID_19.ipynb).

## 1. Objetivo

<p> Implementar um modelo de Machine Learning que consiga prever quais pacientes que se apresentarem no Hospital Sírio Libânes com COVID-19 precisarão de UTI
e quais não precisarão, utilizando apenas os resultados dos exames realizados dentro de duas horas da chegada do paciente no hospital.</p>

<p>Os dados utilizados foram coletados de pacientes com COVID-19 que estiveram no Sírio Libanês no ano de 2020 e que foram examinados/tratados no Hospital. 
Esses dados, anonimizados, foram disponibilizados no Kaggle, bem como a proposta de criação do modelo de classificação da necessidade de UTI pelos pacientes.</p>


## 2. Introdução

<p> Um dos problemas relacionados à pandemia de COVID-19 é o número limitado de leitos de unidades de tratamento intensivo disponíveis,
tanto no sistema de saúde público quanto no privado. Enquanto cerca de 80% das pessoas contaminadas com COVID-19 apresentarão sintomas respiratórios 
de grau leve a moderado e não precisarão de cuidados especiais para que se recuperem, cerca de 20% dos contaminados precisarão de assistência hospitalar
e nos casos mais graves, de tratamento intensivo. Alguns fatores parecem aumentar as chances do desenvolvimento de um quadro mais grave: pessoas mais velhas,
aquelas com comorbidades tais como problemas vasculares, cardiacos, diabetes, doenças respiratórias crônicas e cancer, por exemplo.[1][2] Apesar da maioria dos
casos não requerer internação em UTI como explicado acima, a superlotação se explica pela taxa de disseminação da doença, sendo que ao longo da pandemia,
milhares de casos novos têm sido confirmados todos os dias no Brasil. </p>

<p>Nesse projeto, o objetivo principal é auxiliar na detecção precoce da necessidade de internação em UTI de pacientes que se apresentem no hospital,
no intuito de separar aqueles que precisarão UTI dos demais, objetivando uma utilização mais eficiente dos recursos hospitalares e garantindo uma maior taxa de 
confiança na separação desses dois grupos. Para fazer essa predição, um modelo de classificação baseado nas características e sinais físicos de pacientes que 
necessitaram de UTI e de outros que não precisaram ser internados será construído. </p>

## 3. Os dados


A preparação dos dados foi feita e explicada nesse [notebook](https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/preparacao_dados.ipynb).

O dataset que será utilizado é composto pelas seguintes variáveis:

* Informações demográficas: 3 variáveis;

* Doenças pré-existentes: 9 variáveis;

* Exames de sangue: 36 variáveis, que quando pertinente, foram expandidas em média, mediana, max, min, diff e diff relativa;

* Sinais vitais: 6 variáveis;

* ICU: indica se o paciente foi ou não para UTI

* WINDOW: janela (em horas) em que os eventos ocorreram a partir da admissão no Hospital(0-2, 2-4, 4-6, 6-12 e ABOVE 12).
Os dados foram normalizados pela equipe do Sírio Libanês para que seus valores ficassem no intervalo entre -1 e 1, com exceção das variáveis 
categóricas ( informações demográficas, doenças pré-existentes, ICU e WINDOW).

O Sírio Libanês alertou para que os dados das janelas de tempo onde a variável alvo (ICU) estivesse presente não fossem utilizados, 
pois a ida para a UTI pode ter sido anterior aos resultados dos exames.

## 4. Seleção das features 

O dataset utilizado tem 231 colunas com diferentes informações sobre os pacientes, que são as features a serem utilizadas.
São muitas colunas e provavelmente várias delas não têm influência na gravidade da infecção por COVID-19, tendo pouco valor para um modelo de predição.
Para evitar que esses dados de pouca ou nenhuma relevância levem ao overfitting do modelo a ser implementado, será feita uma seleção das features que serão utilizadas.

As funções utilizadas para modificar o dataset na seleção das features estão implementadas
nesse [arquivo](https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/funcoes.py).

## 5.Construção do modelo e validação


Testei alguns modelos e utilizei algumas métricas de avaliação para validá-los e assim escolher o melhor modelo dentre os testados.

O modelo escolhido no final foi o Random Forest Classifier. Esse modelo foi testado com a base de dados 'dados_limpos', que sofreu as modificações mostradas nesse notebook,
mas também foi testado com a versão dados_finais, que é uma versão onde foram descartadas as features de menor importância pro modelo random forest. Essa exclusão foi feita
no [notebook](https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/Modelos.ipynb) de seleção e teste dos modelos.

### A validação 
Para validar o modelo de machine learning proposto, será feita uma cross-validation (validação cruzada).

Para saber mais sobre a construção e validação do modelo, bem como os resultados obtidos e conclusões, 
acesse o [notebook principal](https://github.com/camilasp/Projeto_Previsao_UTI/blob/master/Machine_learning_na_avaliacao_pacientes_com_COVID_19.ipynb).

## 6. Referências

[1](https://www.who.int/health-topics/coronavirus#tab=tab_1) World Health Organization - Coronavirus

[2](https://www.cdc.gov/coronavirus/2019-ncov/covid-data/investigations-discovery/hospitalization-death-by-age.html) Risk for COVID-19 Infection, Hospitalization, and Death By Age Group


[3](https://www.nature.com/articles/s41598-021-83853-2)  COVID-19: a simple statistical model for predicting intensive care unit load in exponential phases of the disease

[4](https://www.cdc.gov/media/releases/2021/p0607-mrna-reduce-risks.html) Study Shows mRNA Vaccines Reduce Risk of Infection by 91 Percent for Fully Vaccinated People

[5](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8118153/)  Modeling of Future COVID-19 Cases, Hospitalizations, and Deaths, by Vaccination Rates and Nonpharmaceutical Intervention Scenarios — United States, April–September 2021

[6](https://portal.fiocruz.br/en/news/astrazeneca-vaccine-92-effective-against-hospitalizations-due-delta-variant)AstraZeneca vaccine is 92% effective against hospitalizations due to Delta variant

[7](https://www.nejm.org/doi/full/10.1056/NEJMoa2108891)  Effectiveness of Covid-19 Vaccines against the B.1.617.2 (Delta) Variant


[8](https://www.cnnbrasil.com.br/saude/2021/06/27/internacoes-por-covid-19-em-sao-paulo-caem-8-7-em-uma-semana) Internações por covid-19 em São Paulo caem 8.7% em uma semana

[9](https://chrisalbon.com/code/machine_learning/trees_and_forests/feature_selection_using_random_forest/) Feature Selection Using Random Forest

[10](https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection) Applying Filter Methods in Python for Feature Selection

[11](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769) In depth parameter tuning for SVC

[12](https://www.pexels.com/pt-br/) Pexels

