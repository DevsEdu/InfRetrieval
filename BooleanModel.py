import unicodedata
import spacy
nlp = spacy.load("pt_core_news_sm")
import pt_core_news_sm
nlp = pt_core_news_sm.load()
import os

#Preprocessamento e indexação do modelo booleano
#V: Vocabulário, IM: Incidence Matrix, II: Inverted Index e DF: Documents Frequency
def bm_indexing(text ,n_docs , doc_id, V, IM, II, DF):
    #Tokenizer
    doc = nlp(text)

    #Preprocessamento
    for token in doc:
        if not token.is_stop:
            if not token.is_punct:
                lemma = token.lemma_.lower()
                lemma = unicodedata.normalize('NFKD', lemma).encode('ascii', 'ignore').decode('ascii')

                #Verifica se esta no vocabulario
                if lemma not in V:                 

                    #Adiciona no vocabulario
                    V[lemma] = len(V)
                    
                    #Adiciona uma linha na matriz de incidencia referente ao vocabulario
                    IM.append( [0] * n_docs ) 

                    #Adiciona o documento no indice invertido
                    II[lemma] = {doc_id}

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
                        II[lemma].add(doc_id)

                        #Informa a observacao neste documento
                        IM[ V[lemma] ][doc_id] = 1


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

def get_corpus(paths):
    corpus = []

    for path in paths:
        file = open(path,'r')
        
        corpus.append(file.read())

        file.close()
    
    return corpus

def boolean_model(corpus):
    n_docs = len(corpus)
    V = {}
    IM = []
    II = {}
    DF = {}    

    for index, document in enumerate(corpus):
        bm_indexing(document, n_docs, index, V, IM, II, DF)

    for item in II:
        print(item, II[item])

def main():
    paths = get_paths()
    corpus = get_corpus(paths)

    boolean_model(corpus)

if __name__ == "__main__":
    main()
    
    

    