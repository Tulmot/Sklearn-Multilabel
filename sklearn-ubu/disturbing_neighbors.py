"""   """

# Author: Eduardo Tubilleja Calvo

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.utils import check_random_state

class DisturbingNeighbors:
    """A Disturbing Neighbors.
    
     Parámetros
    ----------
    base_estimator : Es el clasificador que usaremos para entrenar nuestro
        conjunto de datos, lo que recibe es o vacio o un objeto, si es vacio
        por defecto se usa el DecisionTreeClassifier.
        
    n_neighbors : Son los vecinos molestones que queremos elegir del
        conjunto de datos, por defecto si no se le pasa nada se eligen 10.
        
    n_features : Es el tamanno del sub-espacio aleatorio, según el cual
        se eligen las caracteristicas al azar que vamos a usar para entrenar
        nuestro clasificador, por defecto es 0.5, es decir se coge la mitad 
        de las caracteristicas, si el valor que se le pasa es mayor de 1,
        se coge esa cantidad de caracteristicas.
        
    rnd_dimensions : Un array aleatorio booleano, su tamanno es igual al 
        numero de caracteristicas del conjunto, aunque luego el numero de
        valores TRUE que contendra seran iguales a el valor de la variable
        n_features, los valores TRUE, indican que caracteristicas son las
        elegidas para valorar el conjunto.
        
    rnd_neighbors : Un array aleatorio de enteros,el tamanno de este array 
        depende de la variable n_neighbors, seleccionara filas aleatorias 
        del conjunto de datos, es lo que llamaremos vecinos molestones.
        
    """
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_neighbors=10,
                 n_features=0.5,
                 random_state=None):
        self.base_estimator =base_estimator 
        self.n_neighbors=n_neighbors
        self.n_features=n_features
        self.random_state=random_state
        self.rnd_dimensions=None
        self.rnd_neighbors=None
        
    def _calculate_features(self,X):
        """Calculamos el numero de caracteristicas que usaremos"""
        if  self.n_features<1:
            return X.shape[1]*self.n_features
        else:
            return self.n_features
    
    def _random_boolean(self):
        """Calculamos un array random boolean que es el que nos indicara que 
        caracteristicas que valoraremos"""
        self.rnd_dimensions=self.random_state.randint(0, 2,self.n_features)
        return self.rnd_dimensions.astype(bool)
        
    def _random_array(self,X):
        """Calculamos un array random para seleccionar unas instancias 
        aleatorias""" 
        tam=X.shape[0]
        return self.random_state.choice(tam, self.n_neighbors, replace=False)
        #return np.random_state.randint(0, tam,self.n_neighbors)
    
    def _reduce_data(self,X):
        """Reducimos los datos obtenidos a las caracteristicas que vamos a 
        evaluar, que seran las que hemos obtenido segun el array random 
        boolean"""
        return X[:, self.rnd_dimensions]
    
    def _nearest_neighbor(self,m_reduce):
        """Calculamos los vecinos mas cercanos a las instancias escogidas 
        antes aleatoriamente"""
        m_neighbors=np.zeros((len(m_reduce),len(self.rnd_neighbors)))
        indice_ins=-1
        for instancia in m_reduce:
            dist=math.inf
            indice_ins+=1
            indice_vec=-1
            for j in self.rnd_neighbors:
                indice_vec+=1
                dist_tmp=euclidean_distances(instancia,m_reduce[j,:])
                if dist_tmp<dist:
                    dist=dist_tmp
                    a=indice_ins
                    b=indice_vec
            m_neighbors[a][b]=1
        return m_neighbors
    
    def fit(self,X,Y):
        """Construyendo un ensemble de estimadores del entrenamiento del
        conjunto (X, y).
        
        Parámetros
        ----------
        X : Es una matriz de forma = [n_instances, n_features]
            Es la muestra de entrada para el entranmiento. Solo se acepta
            si es compatible con el estimador base.
            
        y : matriz de la forma = [n_class]
            Los valores que contiene esta matriz son 0 y 1.
            
        Returns
        -------
        self : object
            Returns self.
        """
        self.random_state = check_random_state(self.random_state)
        self.n_features=self._calculate_features(X)
        self.rnd_dimensions=self._random_boolean()
        self.rnd_neighbors=self._random_array(X)
        m_reduce=self._reduce_data(X)
        m_neighbors=self._nearest_neighbor(m_reduce)
        m_train=np.concatenate((X,m_neighbors),axis=1)
        return self.base_estimator.fit(m_train,Y)
    
    def predict(self,X):
        """Predecir clase para X.
        La clase se predice según una muestra de entrada, calcula la clase
        con la mayor probabilidad, que mas media tiene de predicicción.
        
        Parameters
        ----------
        X : Es una matriz de forma = [n_instances, n_features]
            Es la muestra de entrada para el entranmiento. Solo se acepta
            si es compatible con el estimador base.
            
        Returns
        -------
        y : matriz de forma = [n_class]
            Predice las clases.
        """
        m_reduce=self._reduce_data(X)
        m_neighbors=self._nearest_neighbor(m_reduce)
        m_train=np.concatenate((X,m_neighbors),axis=1)
        return self.base_estimator.predict(m_train)
    
    def predict_proba(self,X):
        """Predecir las probabilidades de una clase para X.
        La clase se predice según una muestra de entrada, calcula la clase
        con la mayor probabilidad, que mas media tiene de predicicción.
        
        Parameters
        ----------
        X : Es una matriz de forma = [n_instances, n_features]
            Es la muestra de entrada para el entranmiento. Solo se acepta
            si es compatible con el estimador base.
            
        Returns
        -------
        y : matriz de forma = [n_class]
            Predice las clases.
        """
        m_reduce=self._reduce_data(X)
        m_neighbors=self._nearest_neighbor(m_reduce)
        m_train=np.concatenate((X,m_neighbors),axis=1)
        return self.base_estimator.predict_proba(m_train)
    
    