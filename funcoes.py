import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

#recebe uma base de dados e cria uma lista com os nomes das colunas que tem apenas um único valor para todas as instâncias
#imprime as colunas a serem excluidas
#exclui as colunas da base de dados e devolve os dados limpos
def exclui_colunas_unico_valor(dados):
    lista=[]
    for x in dados.columns:
        if(len((dados[x]).unique())) == 1:
            lista.append(x)

    dados = dados.drop(lista, axis= 1)
    print(lista)
    return dados

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

#recebe uma base de dados e exclui as colunas com valor quase constantes
#fixei uma seed qualquer no random_state para que não haja variação a cada treino do modelo
def exclui_colunas_quase_constantes(dados):

    colunas = dados.columns
    y = dados["ICU"]
    X = dados[colunas].drop(["ICU"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state= 55596)
    qconstant_filter = VarianceThreshold(threshold=0.005)
    qconstant_filter.fit(X_train)
    qconstant_columns = [column for column in X_train.columns
                         if column not in X_train.columns[qconstant_filter.get_support()]]
    print(qconstant_columns)
    return dados.drop(qconstant_columns, axis=1)


#recebe um dataset, faz a transposição de linhas e colunas
#elimina linhas duplicadas, desfaz a transposição e devolve o dataset
def exclui_colunas_dados_duplicados(dados):
    dados_T = dados.T
    return dados_T.drop_duplicates(keep='first').T

#recebe um dataset e faz a matriz de correlação. Percorre a matriz e seleciona as colunas com correlação maior que 0.85
#exclui as colunas com alta correlação
def elimina_features_alta_correlacao(dados):
    correlated_features = set()
    correlation_matrix = dados.corr()
    for i in range(len(correlation_matrix .columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.85:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    print(correlated_features)
    return dados.drop(correlated_features, axis=1)

#recebe um modelo de machine learning e devolve as taxas de true positive e false positive 
from sklearn.metrics import roc_curve
def roda_modelos(modelo, dados):
            
    x_columns = dados.columns
    y = dados["ICU"]
    x = dados[x_columns].drop(["ICU"], axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state= 55596)

    modelo.fit(x_train, y_train)
    predicao = modelo.predict(x_test)
    prob_predict = modelo.predict_proba(x_test)

    fpr, tpr, thresh = roc_curve(y_test, prob_predict[:,1], pos_label=1)

    return fpr, tpr

#recebe as taxas de true positive e false positive dos modelos comparados e plota a roc curve comparando os modelos
import matplotlib.pyplot as plt   
def plota_roc_curve(fpr, tpr, fpr1, tpr1, fpr2, tpr2):
    
    plt.figure(figsize=(12,8))

   
    plt.plot(fpr, tpr, linestyle='--', color='purple', label = 'DummyClassifier')
    plt.plot(fpr1, tpr1, linestyle='--',color='green', label='Modelo Random Forest')
    plt.plot(fpr2, tpr2, linestyle='--',color='blue', label='Modelo SVC')
   
    
      
    font_t = {'family': 'sans-serif',
        'color': '#777D62',
        'weight': 'bold'
         }
    
    plt.title('ROC curve', pad= 20, fontdict=font_t, fontsize= 18)
    plt.xlabel('False Positive Rate', fontdict=font_t)
    plt.ylabel('True Positive Rate',  fontdict=font_t)
    plt.tick_params(labelcolor= '#777D62')
    plt.legend(loc='best') 
    plt.show()
    
    
    
    


  

    
