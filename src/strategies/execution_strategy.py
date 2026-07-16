# src/strategies/execution_strategy.py

from abc import ABC, abstractmethod

class ExecutionStrategy(ABC):
    @abstractmethod
    def execute(self, context):
        """
        Método principal de execução.
        :param context: A instância da classe Framework que contém o estado e configurações.
        """
        pass