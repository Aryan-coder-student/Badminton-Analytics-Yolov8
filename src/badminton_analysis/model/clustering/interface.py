from abc import ABC , abstractmethod 
from dataclasses import dataclass 
import numpy as np 

@dataclass
class ClusterModel(ABC):
    embeddings:np.ndarray 

    @abstractmethod
    def fit(self):
        """Fit cluster centroids using the embeddings supplied at construction."""
        pass
        
    @abstractmethod
    def predict(self,embeddings:np.ndarray) -> np.ndarray:
        """Assign embeddings to the nearest fitted centroid."""
        pass
    
    @abstractmethod
    def fit_predict(self,embeddings:np.ndarray) -> np.ndarray:
        """Fit new centroids and return a label for every supplied embedding."""
        pass