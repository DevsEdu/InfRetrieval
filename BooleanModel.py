import spacy
nlp = spacy.load("pt_core_news_sm")

import pt_core_news_sm
nlp = pt_core_news_sm.load()

import Tools
import Preprocess
import Similaridade
import numpy as np

import json
import pickle

#Preprocessamento e indexação do modelo booleano
#V: Vocabulário, IM: Incidence Matrix, II: Inverted Index e DF: Documents Frequency
def bm_indexing(text ,n_docs , doc_id, V, IM, II, DF):
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
                IM.append( [0] * n_docs ) 

                #Adiciona o documento no indice invertido
                II[p_token] = [doc_id]

                #Informa a observacao neste documento
                IM[ V[p_token] ][doc_id] = 1                                                

                #Inicia a contagem na frequencia de documento
                DF[p_token] = 1

            #Caso ja estaja no vocabulario
            else:
                #Verifica se não foi observado no documento atual
                if IM[ V[p_token] ][doc_id] == 0:
                    #Acrescenta em 1 a frequencia de documento
                    DF[p_token] += 1

                    #Adiciona o documento
                    II[p_token].append(doc_id)

                    #Informa a observacao neste documento
                    IM[ V[p_token] ][doc_id] = 1

#Obter documentos de uma query
def get_query(II, V):
    query = input('O que deseja buscar? ')

    doc =  nlp(query)

    word_list = []
    docs = {}

    for token in doc:
        p_token = Preprocess.preprocess( token )

        if p_token != -1:
            word_list.append( V[p_token] )

            if docs == {}:
                docs = set( II[ p_token ] )

            else:
                docs = docs.intersection( set( II[p_token] ) )   



    return docs, word_list

def get_query2(II,V):
    query = input('O que deseja buscar? ')

    operador = []
    resultado = []
    word_list = []

    operacoes = {
        'and' : lambda op1, op2: op1.intersection(op2),
        'or' : lambda op1, op2: op1.union(op2),
        'not' : lambda op1, op2 : op1.difference(op2)
    }

    doc = nlp(query)

    for word in doc:
        if word.text == 'and' or word.text == 'or' or word.text == 'not':
            operador.append(word.text)

        else:
            if word.text == ')':
                op = resultado.pop()
                resultado.pop()

                resultado.append(op)

            else:
                p_token = Preprocess.preprocess( word )

                if p_token == -1:
                    resultado.append(-1)

                else:
                    set_word = set( II[ p_token ] )

                    resultado.append(set_word)

                    word_list.append(V[p_token])

            print(operador, resultado)

            if len(operador) > 0:
                if operador[-1] == 'not' and resultado[-1] != '(' :
                    
                    if len(resultado) > 1 and resultado[-2] != '(':                        
                        op1 = resultado.pop()
                        op = operador.pop()

                        if op1 == -1:
                            resultado.append(-1)

                        else:
                            resultado.append( operacoes[op](op1) )
                    
                    elif len(resultado) == 1:
                        op1 = resultado.pop()
                        op = operador.pop()

                        if op1 == -1:
                            resultado.append(-1)

                        else:
                            resultado.append( operacoes[op](op1) )

                elif resultado[-1] != '(' and resultado[-2] != '(':
                    op1 = resultado.pop()
                    op2 = resultado.pop()

                    op = operador.pop()

                    if op1 != -1 and op2 != -1:
                        resultado.append(operacoes[op](op1,op2))

                    elif op1 == -1 and op2 == -1:
                        resultado.append(-1)
                    
                    elif op1 == -1:
                        resultado.append(op2)
                    
                    else:
                        resultado.append(op1)

                    

    if len(operador) > 0:
        op1 = resultado.pop()
        op2 = resultado.pop()

        op = operador.pop()

        resultado.append(operacoes[op](int(op1),int(op2)))

    return resultado[0], word_list


#Modelo booleano de recuperação de informação
def boolean_model(paths):
    n_docs = len(paths)
    V = {}
    IM = []
    II = {}
    DF = {}

    try:
        with open('./BooleanModel/vocabulary', 'r') as f:
            V = json.load(f)
        
        with open('./BooleanModel/incidentMatrix', 'rb') as f:
            IM = pickle.load(f)

        with open('./BooleanModel/invertedIndex') as f:
            II = json.load(f)
        
        with open('./BooleanModel/docsFrequency') as f:
            DF = json.load(f)

    except:
        for index, path in enumerate(paths):
            document =  Tools.get_document(path)

            if len(document) > 1000000:
                i = 1000000

                while True:
                    if document[i] == ' ':
                        break

                    i -= 1

                
                pt1 = document[0:i]
                pt2 = document[i+1:]

                bm_indexing(pt1, n_docs, index, V, IM, II, DF)
                bm_indexing(pt2, n_docs, index, V, IM, II, DF)

            else:
                bm_indexing(document, n_docs, index, V, IM, II, DF)        


        Tools.save_data(V,'./BooleanModel/vocabulary',obj=True)
        Tools.save_data(IM,'./BooleanModel/incidentMatrix')
        Tools.save_data(II,'./BooleanModel/invertedIndex',obj=True)
        Tools.save_data(DF,'./BooleanModel/docsFrequency',obj=True)

    docs, word_list = get_query2(II, V)

    #print(docs, word_list)

    ranking = []

    for doc in docs:
        indices = np.nonzero( np.array(IM)[:,doc] )[0]

        ranking.append( (paths[doc], Similaridade.dice(set(indices), set(word_list))))

    ranking.sort(key=Similaridade.my_comp, reverse=True)

    print('Pontuação utilizando DICE score!')
    for item in ranking:
        print(f'{item[0]}\t ---\t {item[1]}')       

def main():
    paths = Tools.get_paths()

    boolean_model(paths)

if __name__ == "__main__":
    main()
    
    

    