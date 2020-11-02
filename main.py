import numpy as np
import pandas as pd
from framework.Network import Network
from framework.Layers import FullyConnectedLayer, ActivationLayer, FlattenLayer
from framework.Activation import Sigmoid, Relu
from framework.Loss import MSE
from framework.utils import process_data, cls
import time
import sys

# Ruta donde se encuentra el archivo con la data para entrenar/testear
route = 'data/fashion-1.csv'
# Porcentaje de data a utilizar en entrenamiento. El sobrante se usa de test
training_data_percent = 0.7
# Numero de epocas a entrenar el modelo
epochs = 100
# Tasa de aprendizaje
lr = 0.05


def menu():
    print('1.- Entrenar Modelo y Probar')
    print('2.- Cargar Modelo Pre-Entrenado y Probar')
    print('0.- Cerrar Programa')

if __name__ == "__main__":
    training_data , training_labels, testing_data, testing_labels = process_data(route, training_data_percent)
    print('Inicializando modelo de Red...')
    network = Network(
        [
            FlattenLayer(size=28),
            FullyConnectedLayer(28 * 28, 256),
            ActivationLayer(Relu.activation, Relu.activation_derivated),
            FullyConnectedLayer(256, 128),
            ActivationLayer(Relu.activation, Relu.activation_derivated),
            FullyConnectedLayer(128, 10)
        ],
        MSE.loss,
        MSE.derivated_loss)

    while True:
        cls()
        menu()
        opcion = input("Ingrese su opcion deseada : ")

        if opcion == '1':
            print('Iniciando proceso de entrenamiento...')
            network.train(training_data, training_labels, epochs, lr)
            network.test(testing_data, testing_labels)
            aux = input("Presione Enter para Continuar")
        elif opcion == '2':
            print('Cargando Modelo Pre-Entrenado...')
            network.load_model()
            network.test(testing_data, testing_labels)
            aux = input("Presione Enter para Continuar")
        else:
            print('Adios <3')
            sys.exit(-1)

    network.load_model()
    network.test(testing_data, testing_labels)