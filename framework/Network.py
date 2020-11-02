import numpy as np
from framework.Layers import FlattenLayer, FullyConnectedLayer, ActivationLayer
from framework.Activation import Relu
import json, codecs
from framework.utils import get_category
class Network:
    '''
        Clase que define las funcionalidades basicas
        de una red neuronal.
    '''
    def __init__(self, layers, loss_function, loss_prime):
        '''
            Constructor de Red Neuronal.
            Recibe : 
            - Lista de capas para definir la estructura de la red.
            - Funcion de perdida y su derivada
        '''
        # Arreglo de Capas de la Red
        self.layers = []
        # Se agregan las capas indicadas en el consturctor
        for layer in layers:
            self.layers.append(layer)
        # Se asigna funciones de perdida
        self.loss = loss_function
        self.loss_prime = loss_prime

    def predict_category(self, input_data):
        '''
            Funcion que predice una categoria dado un vector
            de entrada.

            Para ello lo unico que realiza es una propagacion
            forward y retorna la salida de la ultima capa.
        '''
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output[0]

    def train(self, train_data, train_labels, epochs, lr):
        '''
            Funcion que entrena la red neuronal.

            Para ello recibe la data de entrenamiento y sus respectivas
            labels. Ademas recibe la cantidad de epocas a iterar
            y la tasa de aprendizaje de la red.
        '''
        train_data_lenght = len(train_data)

        '''
            Para cada epoca:
                Para cada muestra del conjunto de entrenamiento:
                    Por cada capa de la red:
                        Hacer forward propagation
                    Por cada capa de la red en orden inverso:
                        Hacer backward propagation
        '''
        for i in range(epochs):
            err = 0
            for j in range(train_data_lenght):

                # Propagacion Forward
                output = train_data[j]
                for layer in self.layers:
                    output = layer.forward(output)

                err = err +  self.loss(train_labels[j], output)

                # Propagacion Backward
                error = self.loss_prime(train_labels[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, lr)

            # Se obtiene el promedio del error para todas las muestras
            err = err / train_data_lenght
            print(f'Epoca {i+1}/{epochs} Error promedio:{err}')

    def test(self, data, labels):
        '''
            Metodo que prueba la precision de la red.

            Para ello entrega un conjunto de datos de entrada con
            sus respectivos labels.
        '''
        right_predictions = 0
        total_data = len(data)
        for test_data , test_label in zip(data,labels):
            prediction = self.predict_category(test_data)
            predicted_category = np.argmax(prediction)
            real_category = np.argmax(test_label)
            if predicted_category == real_category:
                right_predictions += 1
            print(f'Prediccion : {get_category(predicted_category)} Real : {get_category(real_category)}')
        print(f'Aciertos : {right_predictions} Fallos : {total_data - right_predictions}')
        print(f'Exactitud de Red {str(100*round(right_predictions/total_data,2))}%')

    def save_model(self):
        '''
            Metodo que guarda el modelo en un archivo JSON.

            Es una implementacion muy sencilla por lo cual sugiero no 
            tocarla o se rompe.
        '''
        lista_layers = list()
        for layer in self.layers:
            if isinstance(layer, FlattenLayer):
                data = {
                    'type' : 'FlattenLayer',
                    'size' : layer.size[0]
                }
                lista_layers.append(data)
            elif isinstance(layer, FullyConnectedLayer):
                data = {
                    'type' : 'FullyConnectedLayer',
                    'input_size' : layer.input_size,
                    'output_size' : layer.output_size,
                    'weights' : layer.weights.tolist(),
                    'bias' : layer.bias.tolist()
                }
                lista_layers.append(data)
            else:
                data = {
                    'type' : 'ActivationLayer',
                }
                lista_layers.append(data)
        json.dump(lista_layers, codecs.open('model.json', 'w', encoding='utf-8'))

    def load_model(self):
        '''
            Carga un modelo a partir de un archivo JSON

            Es una implementacion muy sencilla y funciona unicamente
            para el modelo definido en main.py
        '''
        i = 0
        with open('model.json') as json_file:
            data = json.load(json_file)
            for layer in data:
                if layer['type'] == 'FlattenLayer':
                    self.layers[i].size = layer['size']
                elif layer['type'] == 'FullyConnectedLayer':
                    self.layers[i].input_size = layer['input_size']
                    self.layers[i].output_size = layer['output_size']
                    self.layers[i].weights = np.array(layer['weights'], dtype='float32')
                    self.layers[i].bias = np.array(layer['bias'], dtype='float32')
                else:
                    self.layers[i].activation = Relu.activation
                    self.layers[i].activation_prime = Relu.activation_derivated
                i = i + 1
