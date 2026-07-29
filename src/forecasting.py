import time
import numpy as np
import pandas as pd

from src.utils import split_sets, normalize
from src.models.model_factory import ModelFactory  # Importar a fábrica

# Classe de Forecasting que utiliza a ModelFactory para criar os modelos.
# É responsável por treinar o modelo, fazer previsões e plotar os resultados. 
class Forecasting:
    """
    Gerenciador e interface unificada para os modelos preditivos (Facade).

    Esta classe encapsula a lógica de inicialização, preparação de dados 
    (normalização e divisão de treino/teste) e interação com os algoritmos de 
    Machine Learning instanciados pela `ModelFactory`. Ela lida automaticamente 
    com as diferenças de preparação de dados entre modelos de aprendizado online 
    e modelos em lote (offline).
    """

    def __init__(
        self,
        sequence: pd.DataFrame,
        model_name: str,
        resources: list[str],
        path_to_save_weights: str | None,
        use_normalization: bool = True,
        path_to_load_model: str | None = None,
    ):
        """
        Inicializa o gerenciador de previsões e instancia o modelo selecionado.

        Modelos de aprendizado online (ex: 'arf', 'arimax') ignoram a etapa de 
        divisão de dados de treino e teste, pois aprendem de forma incremental. 
        Modelos offline aplicam normalização e dividem o DataFrame fornecido em 
        conjuntos de treino (80%) e teste (20%).

        Args:
            sequence (pd.DataFrame): DataFrame contendo os dados brutos de monitoramento.
            model_name (str): Nome ou sigla do algoritmo preditivo a ser utilizado.
            resources (list[str]): Lista de métricas alvo (ex: ['Mem', 'CPU']).
            path_to_save_weights (str | None): Caminho para salvar os pesos do modelo treinados.
            use_normalization (bool, opcional): Flag para habilitar ou desabilitar a 
                                                normalização Min-Max. O padrão é True.
            path_to_load_model (str | None, opcional): Caminho para carregar um arquivo de pesos 
                                                       (.h5) pré-treinado. O padrão é None.
        """
        
        self.resources = resources
        self.normalization_params = {}

        if model_name in ["arf","hat_perceptron", "isoup", "sarimax", "varma", "arimax",
                          "snarimax_ht","snarimax_hat",
                          "snarimax_oxt", "snarimax_arf", "snarimax_amf"]:
                    
            self.train_sequence = None
            self.test_sequence = None
            
            # Utilizar a Factory em vez do método interno
            self.model = ModelFactory.create_model(
                model_name=model_name,
                resources=self.resources,
                normalization_params=self.normalization_params,
                path_to_save_weights=path_to_save_weights,
                path_to_load_model=path_to_load_model
            )
            return

        sequence = sequence[self.resources].copy()
        if use_normalization:
            for resource in self.resources:
                sequence[resource], s_min, s_max = normalize(sequence[resource])
                self.normalization_params[resource] = (s_min, s_max)
        self.train_sequence, self.test_sequence = split_sets(sequence, 0.8)

        # Utilizar a Factory novamente para os modelos offline
        self.model = ModelFactory.create_model(
            model_name=model_name,
            resources=self.resources,
            normalization_params=self.normalization_params,
            path_to_save_weights=path_to_save_weights,
            path_to_load_model=path_to_load_model
        )

    def train(self):
        """
        Executa o treinamento em lote (batch) do modelo instanciado.
        
        Mede e exibe no console o tempo total gasto na fase de treinamento.
        Deve ser utilizado apenas para modelos offline (ex: HLSTM), cujos dados 
        foram divididos na inicialização da classe.
        """

        start_time = time.time()
        self.model.train(self.train_sequence, self.test_sequence)
        end_time = time.time()
        print(f"\nTraining time: {end_time - start_time} seconds\n")

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Realiza uma única previsão com base na sequência de entrada fornecida.

        Args:
            sequence (np.ndarray): Tensor contendo os dados mais recentes do sistema.

        Returns:
            np.ndarray: Array contendo a previsão do próximo passo para os recursos alvo.
        """

        return self.model.predict(sequence)

    def predict_future(
        self, sequence: np.ndarray, n_steps_forecasted: int
    ) -> np.ndarray:
        """
        Realiza múltiplas previsões recursivas projetando um horizonte futuro.

        Utiliza uma abordagem de janela deslizante (sliding window): o modelo prevê 
        o passo temporal t+1, essa previsão é anexada ao final da sequência de entrada 
        (descartando o registro mais antigo), e o processo se repete para gerar o passo t+2, 
        e assim sucessivamente.

        Nota: A implementação atual exige um formato específico de tensor (reshape) 
        otimizado para redes neurais (ex: 1x2x1x2xN).

        Args:
            sequence (np.ndarray): Tensor inicial contendo os dados de entrada.
            n_steps_forecasted (int): Número de passos futuros a serem previstos.

        Returns:
            np.ndarray: Array bidimensional contendo todas as previsões geradas para o horizonte.
        """
        
        predictions = []
        for _ in range(n_steps_forecasted):
            prediction = self.predict(sequence)
            predictions.append(prediction[0])

            # reshape the sequence to append the prediction
            sequence = sequence.reshape((4, len(self.resources)))
            # remove first row and append the prediction to the end
            sequence = np.append(sequence[1:], prediction, axis=0)
            # reshape the sequence to be fed to the model
            sequence = sequence.reshape((1, 2, 1, 2, len(self.resources)))

        return np.array(predictions)

    def plot_results(self):
        """
        Delega a geração de gráficos de resultados para a implementação específica do modelo.
        """
        self.model.plot_results()
