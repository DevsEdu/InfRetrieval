import spacy
nlp = spacy.load("pt_core_news_sm")

import pt_core_news_sm
nlp = pt_core_news_sm.load()

import tools
import Preprocess
import similaridade

import numpy as np
import pickle
import json

#Preprocessamento e indexação do modelo vetorial
#V: Vocabulário, IM: Incidence Matrix, II: Inverted Index e DF: Documents Frequency
def vm_indexing(text ,n_docs , doc_id, V, TF, DF, TF_IDF):
    #Tokenizer
    doc = nlp(text)

    #Preprocessamento
    for token in doc:
        #Preprocess token
        p_token = Preprocess.preprocess(token)

        if p_token != -1:

            #Verifica se esta no vocabulario
            if p_token not in V:               
                
                #Adiciona no vocabulario
                V[p_token] = len(V)
                
                #Adiciona uma linha na matriz de incidencia referente ao vocabulario
                TF.append( [0] * n_docs )

                #Adiciona uma linha na matriz td_idf para representação vetorial
                TF_IDF.append( [0] * n_docs )

                #Informa a observacao neste documento
                TF[ V[p_token] ][doc_id] += 1                                                              

                #Inicia a contagem na frequencia de documento
                DF.append(1)

            #Caso ja estaja no vocabulario
            else:
                #Verifica se não foi observado no documento atual
                if TF[ V[p_token] ][doc_id] == 0:
                    #Acrescenta em 1 a frequencia de documento
                    DF[ V[p_token] ] += 1

                #Informa a observacao neste documento
                TF[ V[p_token] ][doc_id] += 1  

#Obter documentos de uma query
def get_query(V, tf_idf, idf):
    q_vector = [0] * len(V)

    words = []
    query = input('O que deseja buscar? ')

    doc =  nlp(query)    

    for token in doc:
        p_token = Preprocess.preprocess( token )

        if p_token != -1:
            q_vector[V[p_token]] += 1

            if V[p_token] not in words:
                words.append(V[p_token])

    for i in words:
        q_vector[i] = q_vector[i] * idf[i]

    return q_vector

def vector_model(paths):
    n_docs = len(paths)
    V = {}
    tf = []
    df = []
    idf = []
    tf_idf = []

    try:
        with open('./VectorModel/td_idf', 'rb') as f:
            tf_idf = pickle.load(f)
        
        with open('./VectorModel/vocabulary', 'r') as f:
            V = json.load(f)
        
        with open('./VectorModel/tf', 'rb') as f:
            tf = pickle.load(f)
        
        with open('./VectorModel/df', 'rb') as f:
            df = pickle.load(f)
        
        with open('./VectorModel/idf', 'rb') as f:
            idf = pickle.load(f)


    except:
        for index, path in enumerate(paths):
                document =  tools.get_document(path)

                if len(document) > 1000000:
                    i = 1000000

                    while True:
                        if document[i] == ' ':
                            break

                        i -= 1

                    
                    pt1 = document[0:i]
                    pt2 = document[i+1:]

                    vm_indexing(pt1, n_docs, index, V, tf, df, tf_idf)
                    vm_indexing(pt2, n_docs, index, V, tf, df, tf_idf)

                else:
                    vm_indexing(document, n_docs, index, V, tf, df, tf_idf)  

        for i in range(len(tf_idf)):
            for j in range(n_docs):
                if j == 0:
                    idf.append( np.log( n_docs/df[i] ) )

                tf_idf[i][j] = tf[i][j] * idf[i]

        tools.save_data(V,'./VectorModel/vocabulary', obj=True)
        tools.save_data(tf_idf,'./VectorModel/td_idf')
        tools.save_data(tf,'./VectorModel/tf')
        tools.save_data(df,'./VectorModel/df')
        tools.save_data(idf,'./VectorModel/idf')

    q_vector = get_query(V, tf_idf, idf)
    tf_idf = np.array(tf_idf)
    
    ranking = []

    for i in range(n_docs):
        d_vector = tf_idf[:,i]

        ranking.append( (paths[i], similaridade.cosseno(q_vector,d_vector)) )

    ranking.sort(key=similaridade.my_comp, reverse=True)

    print('Pontuação utilizando o cosseno score!')
    for item in ranking[0:10]:
        print( f'{item[0]}\t --------\t {item[1]}' )

def main():
    paths = tools.get_paths()

    vector_model(paths)

if __name__ == "__main__":
    main()
    
    

    