from abc import ABC, abstractmethod

class OnlineModel(ABC):
    """
    Interface base (Contrato) para os modelos de aprendizado online (incremental).
    
    Os modelos que implementam esta classe atualizam seus pesos matemáticos a 
    cada nova amostra de monitoramento coletada, permitindo adaptação em tempo 
    real ao envelhecimento do software sem a necessidade de reter todo o histórico 
    na memória.
    """
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
    