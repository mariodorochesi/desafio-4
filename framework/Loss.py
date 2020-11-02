import numpy as np

class MSE:
    '''
        Clase que define la funcion de perdida
        para el error cuadratico meido junto con su derivada
    '''
    @staticmethod
    def loss(real, prediction):
        return np.mean(np.power(real - prediction, 2))
    @staticmethod
    def derivated_loss(real, prediction):
        return 2 * (real - prediction) / real.size