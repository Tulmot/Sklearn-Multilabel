import numpy as np
from random import shuffle
#from sklearn.datasets import make_multilabel_classification
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.tree import DecisionTreeClassifier
#from sklearn import tree

class DisturbingNeigbors:
    #Inicializamos variables
    def __init__(self,
                 base_estimator=DecisionTreeClassifier,
                 n_vecinos=10,
                 n_caracteristicas=0.5):
        self.base_estimator =base_estimator 
        self.n_vecinos=n_vecinos
        self.n_caracteristicas=n_caracteristicas*n_vecinos
    
   
    
    #Calculamos un array random boolean que es el que nos indicara que 
    #caracteristicas que valoraremos
    def _random_boolean(self,rnd_dimensions):
        rnd_dimensions=np.random.randint(0, 2,self.n_caracteristicas)
        return rnd_dimensions.astype(bool)
     
    #Calculamos un array random para seleccionar unas instancias aleatorias    
    def _random_array(self):
        s=list(range(self.n_vecinos))
        shuffle(s)
        return np.array(s[:(int(len(s)/2))])
    
    #Reducimos los datos obtenidos a las caracteristicas que vamos a evaluar,
    #que seran las que hemos obtenido segun el array random boolean
    def _reduce_data(self,X):
        return X[:, rnd_dimensions]
    
    #Calculamos los vecinos mas cercanos a las instancias escogidas antes
    #aleatoriamente
    def _nearest_neighbor(self,m_reducida):
        m_vecinos=np.zeros((self.n_vecinos,len(rnd_neighbors)))
        cont=-1
        for i in m_reducida:
            dist=999
            cont+=1
            cont2=-1
            for j in rnd_neighbors:
                cont2+=1
                dist2=euclidean_distances([i],[m_reducida[j,:]])
                if dist2<dist:
                    dist=dist2
                    a=cont
                    b=cont2
            m_vecinos[a][b]=1
        return m_vecinos
    
    #CLasificador que vamos a usar para entrenar
    def _clasificate(self,X,Y,m_vecinos):
        m_entrenamiento=np.concatenate((X,m_vecinos),axis=1)
        return self.base_estimator.fit(m_entrenamiento,Y)
    
    #Funcion que llama a los metodos necesarios para devolver el fit
    def fit(self,X,Y):
        rnd_dimensions=_random_boolean(self)
        rnd_neighbors=_random_array(self)
        m_reducida=_reduce_data(self,X)
        m_vecinos=_nearest_neighbor(self,m_reducida)
        return _clasificate(self,X,Y,m_vecinos)
    
    #Recibe la otra parte del conjunto de datos, a partir de la cual vamos
    #a poder predecir luego
    def predict(self,X1):
        m_reducida2=_reduce_data(self,X1)
        m_vecinos2=_nearest_neighbor(self,m_reducida2)
        m_entrenamiento2=np.concatenate((X1,m_vecinos2),axis=1)
        return self.base_estimator.predict(m_entrenamiento2)
    
    
    