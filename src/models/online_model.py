from abc import ABC, abstractmethod

class OnlineModel(ABC):
    @abstractmethod
    def learn_one(self, features: dict, target: float):
        """
        Atualiza o modelo incrementalmente com um novo par (features, target).
        Isso substitui o método 'train' dos modelos batch.
        """
        pass

    @abstractmethod
    def predict_one(self, features: dict) -> float:
        """
        Faz a previsão para o próximo passo baseado nas features atuais.
        """
        pass
    