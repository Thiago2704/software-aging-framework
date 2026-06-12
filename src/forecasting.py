import time
import numpy as np
import pandas as pd

from src.utils import split_sets, normalize
from src.models.model_factory import ModelFactory  # Importar a fábrica

# Classe de Forecasting que utiliza a ModelFactory para criar os modelos.
# É responsável por treinar o modelo, fazer previsões e plotar os resultados. 
class Forecasting:
    def __init__(
        self,
        sequence: pd.DataFrame,
        model_name: str,
        resources: list[str],
        path_to_save_weights: str | None,
        use_normalization: bool = True,
        path_to_load_model: str | None = None,
    ):
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
        start_time = time.time()
        self.model.train(self.train_sequence, self.test_sequence)
        end_time = time.time()
        print(f"\nTraining time: {end_time - start_time} seconds\n")

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        return self.model.predict(sequence)

    def predict_future(
        self, sequence: np.ndarray, n_steps_forecasted: int
    ) -> np.ndarray:
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
        self.model.plot_results()
