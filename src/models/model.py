from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Model(ABC):
    """
    Interface base para os modelos em lote do framework.

    Essa classe define a estrutura obrigatória que qualquer 
    algoritmo lote/offline deve seguir para ser compatível com as fábricas e 
    estratégias do sistema.
    """
    @abstractmethod
    def train(
        self,
        train_sequence: pd.DataFrame | pd.Series,
        test_sequence: pd.DataFrame | pd.Series,
    ):
        pass

    @abstractmethod
    def predict(self, sequence: pd.DataFrame | pd.Series) -> np.ndarray:
        pass

    @abstractmethod
    def plot_results(self):
        pass
