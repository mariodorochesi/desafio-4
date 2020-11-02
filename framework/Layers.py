import numpy as np


class FlattenLayer:
    '''
        Implementacion de FlattenLayer.

        FlatternLayer es un tipo de capa especial muy utilizada por los
        frameworks que se encarga unicamente de ajustar una data de entrada
        a un tamano compatible con las siguientes capas de la red.

        Esto se logra a traves de la funcion reshape de numpy
    '''
    def __init__(self, size):
        self.size = (size, size)
    
    def backward(self, output_error, learning_rate):
        return np.reshape(output_error, self.size)

    def forward(self, input):
        return np.reshape(input, (1, -1))

class FullyConnectedLayer:
    '''
        Clase que implementa una capa donde todas las neuronas
        se encuentran conectadas a la capa anterior.
    '''
    def __init__(self, input_size, output_size):
        '''
            Constructor de la clase.
            Recibe como entrada la cantidad de neuronas
            de entrada y la cantidad de salidas que genera.
        '''
        # Tamano de entrada
        self.input_size = input_size
        # Tamano de salida
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / (input_size + output_size)
        self.bias = np.random.randn(1, output_size) / (input_size + output_size)

    def backward(self, output_error, learning_rate):
        '''
            Metodo que realiza el procedimiento de backward propagation.

            Para ello recibe como entrada el error de la capa siguiente y
            calcula un error de entrada como el producto punto de los pesos
            de la capa por el vector de error. Luego ajusta sus valores
            de peso y bias a travez de la resta del error por el factor
            de aprendizaje.
        '''
        # Se calculan los errores
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        # Se ajusta peso y bias de la capa
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
    def forward(self, input):
        '''
            Realiza el producto punto entre el vector
            de entrada y los pesos de la capa. Luego le
            agrega el sesgo para evitar linealidad
        '''
        self.input = input
        return np.dot(input, self.weights) + self.bias


class ActivationLayer:
    '''
        Capa que sirve para utilizar la funcion de activacion
        cuando se hace una propagacion forward o utilizar
        la derivada de la funcion de activacion cuando se hace
        una propagacion backward.
    '''
    def __init__(self, activation, activation_prime):
        '''
            Recibe como parametros las funciones de activacion a utilizar
            en la capa.
        '''
        self.activation = activation
        self.activation_prime = activation_prime
    
    def backward(self, output_error, learning_rate):
        '''
            Aplica la funcion de activacion a la salida de la capa
            siguiente, y la replica hacia atras.
        '''
        return output_error * self.activation_prime(self.input)

    def forward(self, input):
        '''
            Aplica la funcion de activacion a la entrada de la capa.
        '''
        self.input = input
        return self.activation(input)