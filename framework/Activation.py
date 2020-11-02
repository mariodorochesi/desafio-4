import numpy as np

class Sigmoid:
    '''
        Clase que define la funcion de activacion Sigmoid como
        metodos estaticos
    '''
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def activation_derivated(x):
        return np.exp(-x) / (1 + np.exp(-x))**2

class Relu:
    '''
        Clase que define la funcion de activacion Relu como
        metodos estaticos
    '''
    @staticmethod
    def activation(x):
        return np.maximum(x, 0) 
    @staticmethod
    def activation_derivated(x):
        return np.array(x >= 0).astype('int')