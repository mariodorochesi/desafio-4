import numpy as np
import pandas as pd
import math
import os

# Arreglo auxiliar con las categorias ordenadas por su respectivo indice
categories = ['polera', 'pantalon', 'sweater', 'vestido',
            'saco', 'sandalia', 'camisa', 'zapatilla', 'bolso', 'bota']

def process_data(route, training_dp):
    '''
        Metodo que procesa el archivo pasado por ruta.
        Ademas de eso:
            - Normaliza los vectores de caracteristicas
            - Aplica oneshot encoding a los labels de categorias
            - Genera un corte entre train y test
    '''
    print(f'Leyendo archivo {route}...')
    # Se lee el CSV
    print('Separando labels y vectores de caracteristicas...')
    df = pd.read_csv(route, sep=',')
    # Se obtiene la columna de Labels
    labels = df['label']
    # Se borra la columna labels del dataframe
    df = df.drop(['label'], axis=1)
    # Se obtiene la data como un arreglo de numpy
    print('Normalizando vector de caracteristicas...')
    data = df.to_numpy()
    data = data.reshape(data.shape[0],28,28)
    data = data.astype('float32')
    data = data / 255
    labels = labels.to_numpy()
    print('Aplicando Oneshot Encoding a Labels...')
    labels = oneshot_encoding(labels)
    training_cut = math.floor((len(data)-1)*training_dp)
    return data[0:training_cut], labels[0:training_cut], data[training_cut+1:len(data)-1], labels[training_cut+1:len(labels)-1]



def oneshot_encoding(x):
    '''
        Implementacion sencilla que aplica oneshot encoding a un vector
        de vectores de caracteristicas.
    '''
    arreglo = np.zeros((0,10))
    for i in x:
        aux = np.zeros(10)
        aux[i] = 1
        arreglo = np.insert(arreglo,len(arreglo), aux,axis=0)
    return arreglo

def get_category(x):
    return categories[x]

def cls():
    os.system('cls' if os.name=='nt' else 'clear')