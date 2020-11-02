# **Desafio 3**

Este repositorio considera la implementacion de una red neuronal propia para el reconocimiento de patrones en un dataset de 70.000 prendas de ropa distribuidas en 10 categorias.

## **Ejecución**
``` 
python3 main.py
```

## **Algoritmo Propuesto**

### **Vector de Caracteristicas**

Para el desarrollo de esta actividad se ha considerado inicialmente cargar los datos del dataset y distriburilos en 2 secciones, una para la data relacionada a los pixeles de la imagenes y otro para los labels de cada una de las prendas.

Posteriormente se normaliza los vector de pixeles a valores entre 0 y 1.

### **Oneshot Encoding**

En este caso cada uno de los labels en el intervalo [0,9] se ha transformado a un vector de binarios que indica a que categoria pertenece mediante la tecnica de oneshot encoding.

## **Modelo Propuesto**

El modelo propuesto consta de 6 capas aunque tecnicamente podria ser considerado un modelo de 3 capas, debido a que algunas capas son de activacion las cuales no es nada mas que aplicar la funcion de activacion en la propagacion sea forward o backward.

- **Flattern Layer** : Se encarga de tomar el input y hacer un reshape de la data tanto en la propagacion backward como forward. Es una capa muy utilizada en los frameworks comunes.
- **FullyConnectedLayer**(28*28, 256): Una capa completamente conectada que toma como entrada todos los pixeles y los lleva a una salida de 256 caracteristicas.
- **ActivationLayer** : Capa que se encarga se aplicar la funcion de activacion entre dos capas.
- **FullyConnectedLayer***(256,128): Toma como entrada las 256 caracteristicas de la capa anterior y las lleva a 128 caracteristicas.
- **ActivationLayer**: Capa que se encarga se aplicar la funcion de activacion entre dos capas.
- **FullyConnectedLayer**(128,10): Toma como entrada las 128 caracteristicas anterior y las lleva finalmente a 10 salidas para determinar a que categoria corresponde.

## **Uso**

Existen 3 opciones en el menu, las cuales se describen a continuacion.

### Opcion 1

La opción 1 del menu entrena el modelo segun los hiperparametros definidos en el codigo.

Este procedimiento tomó casi 1 hora en mi computador.


### Opcion 2

La opción 2 del menu carga el modelo pre-entrenado y lo utiliza para predecir el 30% del dataset.

Finalmente entrega un porcentaje de precision del modelo que anticipo es del **89%**.

### Opcion 3

Cierra el programa :3