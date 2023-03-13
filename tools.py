import os
import json
import pickle

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

#Geração de arquivo com pickle ou json
def save_data(data,name,obj=False):    

    if obj:
        file = open(name, 'w')
        json.dump(data,file)

    else:
        file = open(name, 'wb')
        pickle.dump(data,file)

    file.close()
