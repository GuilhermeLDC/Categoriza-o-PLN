import pandas as pd
import os
import re
import numpy as np
import random
import pickle
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC 
from nltk.corpus import stopwords


class Classificador_smartphone:
    
    def __init__(self, max_features = 25, lowercase = True):
        self.path = os.path.dirname(__file__)
        self.max_features = max_features
        self.svm = None

        #Remoção de stopwords em português + adicionais de palavras comuns nos títulos
        #de smartphones, mas que não tem tanta relevância na categorização
        self.stopwords = stopwords.words('Portuguese')
        psw_ad = ['gb', 'tela', 'preto', 'branco', 'dourado', 'vermelho', 'g', 'mb',
                  'câmera', 'chip', 'sim', 'plus', 'mp', 'fm', 'dual', 'bluetooth',
                  'desbloqueado', 'flip', 'ram', 'rádio', 'vga', 'sm','xt','sm', 
                  'cinza', 'studio', 'grand', 'azul', 'roxo', 'amarelo', 'rosa', 'play'
                  'pink', 'ghz', 'prata', 'cinza', 'pixi', 'mini', 'tank','xl', 'positivo',
                  'yellow','multilaser','câm', 'hd','cam', 'camera', 'core', 'edition', 'max']

        for w in psw_ad:
            self.stopwords.append(w)

        #Bag of words na conversão de texto em vetor.
        self.vectorizer = CountVectorizer(stop_words = self.stopwords,
                                          max_features = max_features,
                                          lowercase = lowercase)
                

    def treinar_classificador(self, positivos = 'smartphone_buscape.csv',
                              negativos = 'outros_buscape.csv'):
                        
        positivos_path = os.path.join(self.path, positivos)
        negativos_path = os.path.join(self.path, negativos)
        file_positivos = [w for w in pd.read_csv(positivos_path)['produto']]
        file_negativos = [w for w in pd.read_csv(negativos_path)['produto']]

        #lista para extrair vocabulário, contando também com exemplos de títulos
        #que não são smartphone (como capas), mas que apresentam palavras comuns aos títulos de
        #smartphones:
        file_vocabulario = file_positivos.copy()
        for w in file_negativos[650:]:
            file_vocabulario.append(w)

        file_vocabulario_norm = self.normalizar(file_vocabulario)
        random.shuffle(file_vocabulario_norm)
        

        file_positivos_norm = self.normalizar(file_positivos)
        random.shuffle(file_positivos_norm)
        
        file_negativos_norm = self.normalizar(file_negativos)
        random.shuffle(file_negativos_norm)

        

        #Extração de vocabulário:
        self.vectorizer.fit(file_vocabulario_norm)
        print('Vocabulário criado:', '______________',
              '{}'.format(self.vectorizer.get_feature_names()), '______________',sep = '\n')

        
        #Treino do classificador com 85% dos dados carregados,
        # o restante será usado como validação do classificador.
        q_treino_pos = int(0.85*len(file_positivos_norm))
        q_teste_pos = len(file_positivos_norm) - q_treino_pos

        q_treino_neg = int(0.85*len(file_negativos_norm))
        q_teste_neg = len(file_negativos_norm) - q_treino_neg

        #Formato adequado ao classificador:
        x_treino = np.zeros((q_treino_pos + q_treino_neg, self.max_features) )
        y_treino = np.zeros(q_treino_pos + q_treino_neg)
        
        for i in range(0, q_treino_pos):
            x_treino[i,:] = self.vectorizer.transform([file_positivos_norm[i]]).toarray()
            y_treino[i] = 1
            
        for i in range(q_treino_pos, q_treino_neg):
            x_treino[i,:] = self.vectorizer.transform([file_negatitivos_norm[i - q_treino_pos]]).toarray()

        #SVM
        self.svm = SVC( kernel = 'rbf' , gamma = 0.15 , C = 10)
        self.svm.fit(x_treino, y_treino)

        #teste de desempenho:
        x_teste = np.zeros((q_teste_pos + q_teste_neg, self.max_features) )
        y_teste = np.zeros(q_teste_pos + q_teste_neg)
        
        for i in range(0 , q_teste_pos):
            x_teste[i,:] = self.vectorizer.transform([file_positivos_norm[i + q_treino_pos]]).toarray()
            y_teste[i] = 1
            
        for i in range(q_teste_pos, q_teste_neg):
            x_teste[i,:] = self.vectorizer.transform([file_negatitivos_norm[i + q_treino_neg - q_teste_pos]]).toarray()

        print('acurácia média do classificador:','______________',
              '{}'.format(self.svm.score(x_teste, y_teste)), '______________', sep = '\n')

        
        


    def normalizar( self, documentos ):
        file_normalizado = [re.sub('[0-9]+', '', w).lower() for
                             w in documentos]

        return file_normalizado

                           

    def adicionar_stopwords(self, novas_stopwords):
        for w in novas_stopwords:
            self.stopwords.append(w)

    def predizer(self, documento):
        if self.svm != None:
            documento = self.normalizar([documento])
            vetor = self.vectorizer.transform(documento).toarray()
            pred = self.svm.predict(vetor)
            '''
            if pred == 1:
                print('Classificação:', 'O produto é um Smartphone',
                      '______________', sep = '\n')

            else:
                print('Classificação:', 'O produto não é um Smartphone',
                      '______________', sep = '\n')
            '''

            return pred

        else:
            print('Não há nenhum classificador treinado '
                    + 'para utilizar. Treine ou carregue um.')

            

    def salvar_modelo(self, nome_svm = 'classificador_smartphone_svm.sav',
                      nome_BoW = 'classificador_smartphone_BoW.sav'):
        if self.svm != None:
            pickle.dump(self.svm, open(nome_svm, 'wb'))
            pickle.dump(self.vectorizer, open(nome_BoW, 'wb'))
        else:
            print('Não há nenhum classificador treinado '
                    + 'para salvar. Treine um.')

    def carregar_modelo(self, nome_svm = 'classificador_smartphone_svm.sav',
                      nome_BoW = 'classificador_smartphone_BoW.sav'):
        self.svm = pickle.load(open(nome_svm, 'rb'))
        self.vectorizer = pickle.load(open(nome_BoW, 'rb'))

if __name__ == '__main__':
    
    clf = Classificador_smartphone()
    clf.carregar_modelo()

    data = pd.read_csv('data_estag_ds.tsv', sep='\t')

    

    with open('resultados.tsv', 'wt', encoding = 'utf-8') as file:
            fieldname = ['ID', 'TITLE', 'SMARTPHONE']
            writer = csv.DictWriter(file, delimiter='\t', fieldnames = fieldname)
            writer.writeheader()
            for i, w in enumerate(data['TITLE']):
                    p = clf.predizer(w)
                    if p == 1:
                        predito = 'SIM'
                    else:
                        predito = 'NÃO'
                    writer.writerow({'ID': data['ID'][i], 'TITLE':w, 'SMARTPHONE': predito})

    
        
        
    
