
import numpy as np
from sklearn.datasets import make_multilabel_classification
from scipy.spatial import distance
from random import randint
from random import shuffle
class DisturbingNeigbors():
    #Inicializamos variables
    def __init__(self,
                 n_instancias=10,
                 n_caracteristicas=8,
                 n_clases=5):
        super(DisturbingNeigbors, self).__init__(
            n_instancias,
            n_caracteristicas,
            n_clases)
    
   #Calculamos el conjunto de datos
    def conjunto_datos(self):
        X, Y = make_multilabel_classification(n_samples=self.n_instancias, 
                                              n_features=self.n_caracteristicas, n_classes=self.n_clases)
        return X, Y
    
    #Calculamos un array aleatorio boolean que es el que nos indicara que caracteristicas
    #valoraremos a la hora de calcular la distancia al vecino mas cercano
    def aleatorio_Boolean(self):
        RndDimensions=np.random.randint(0, 2,self.n_caracteristicas)
        RndDimensions=RndDimensions.astype(bool)
        return RndDimensions
     
    #Calculamos un array aleatorio para seleccionar unas instancias aleatorias que
    #seran con las que calculemos la distancia al vecino mas cercano    
    def aleatorio_Array(self):
        s=list(range(self.n_instancias))
        random.shuffle(s)
        RndNeighbors=np.array(s[:(int(len(s)/2))])
        return RndNeighbors
    
    #Reducimos los datos obtenidos a las caracteristicas que vamos a evaluar, que
    #seran las que hemos obtenido segun el array aleatorio boolean
    def reducir_Datos(X,RndDimensions):
        X2=X[:, RndDimensions]
        return X2
    
    #Calculamos los vecinos mas cercanos a las instancias escogidas antes
    #aleatoriamente
    def matriz_Distancia(X2,RndNeighbors,mDistancia):
        cont=-1
        for i in X2:
            dist=999
            cont+=1
            cont2=-1
            for j in RndNeighbors:
                cont2+=1
                dist2=distance.euclidean(i,X2[j,:])
                if dist2<dist:
                    dist=dist2
                    a=cont
                    b=cont2
            mDistancia[a][b]=1
        return mDistancia