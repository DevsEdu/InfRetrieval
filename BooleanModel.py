import unicodedata
import spacy
nlp = spacy.load("pt_core_news_sm")
import pt_core_news_sm
nlp = pt_core_news_sm.load()
import os
import json
import pickle

#Preprocessamento e indexação do modelo booleano
#V: Vocabulário, IM: Incidence Matrix, II: Inverted Index e DF: Documents Frequency
def bm_indexing(text ,n_docs , doc_id, V, IM, II, DF):
    #Tokenizer
    doc = nlp(text)

    #Preprocessamento
    for token in doc:
        if not token.is_stop:
            if not token.is_punct:
                if not token.is_space:
                    lemma = token.lemma_.lower()
                    lemma = unicodedata.normalize('NFKD', lemma).encode('ascii', 'ignore').decode('ascii')

                    #Verifica se esta no vocabulario
                    if lemma not in V:                 

                        #Adiciona no vocabulario
                        V[lemma] = len(V)
                        
                        #Adiciona uma linha na matriz de incidencia referente ao vocabulario
                        IM.append( [0] * n_docs ) 

                        #Adiciona o documento no indice invertido
                        II[lemma] = [doc_id]

                        #Informa a observacao neste documento
                        IM[ V[lemma] ][doc_id] = 1                                                

                        #Inicia a contagem na frequencia de documento
                        DF[lemma] = 1

                    #Caso ja estaja no vocabulario
                    else:
                        #Verifica se não foi observado no documento atual
                        if IM[ V[lemma] ][doc_id] == 0:
                            #Acrescenta em 1 a frequencia de documento
                            DF[lemma] += 1

                            #Adiciona o documento
                            II[lemma].append(doc_id)

                            #Informa a observacao neste documento
                            IM[ V[lemma] ][doc_id] = 1

#Diretórios para os documentos
def get_paths():
    paths = []

    #Obtendo todas as pastas
    pastes = os.listdir("./data")

    for paste in pastes:       
        files = os.listdir("./data/" + paste)

        #Obtendo todos os arquivos
        for file in files:
            paths.append("./data/"+paste+"/"+file)

    return paths

#Leitura de todos os documentos
def get_document(path):    

    file = open(path,'r')
    
    document = file.read()

    file.close()
    
    return document.replace('\n',' ')

def save_data(data,name,obj=False):    

    if obj:
        file = open(name, 'w')
        json.dump(data,file)

    else:
        file = open(name, 'wb')
        pickle.dump(data,file)

    file.close()

#Modelo booleano de recuperação de informação
def boolean_model(paths):
    n_docs = len(paths)
    V = {}
    IM = []
    II = {}
    DF = {}

    try:
        with open('vocabulary', 'r') as f:
            V = json.load(f)
        
        with open('incidentMatrix', 'rb') as f:
            IM = pickle.load(f)

        with open('invertedIndex') as f:
            II = json.load(f)
        
        with open('docsFrequency') as f:
            DF = json.load(f)

    except:
        for index, path in enumerate(paths):
            document =  get_document(path)

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


        save_data(V,'vocabulary',obj=True)
        save_data(IM,'incidentMatrix')
        save_data(II,'invertedIndex',obj=True)
        save_data(DF,'docsFrequency',obj=True)

def main():
    paths = get_paths()

    boolean_model(paths)

if __name__ == "__main__":
    main()
    
    

    