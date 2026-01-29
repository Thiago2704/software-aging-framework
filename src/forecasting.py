import time

import numpy as np
import pandas as pd

from src.models import HLSTM, MovingAverage, Model, ARMA, VARMA, SARIMAX, AdaptiveRandomForest, HoeffdingAdaptiveTreePerceptron
from src.models import iSOUP
from src.utils import split_sets, normalize


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

        if model_name in ["arf","hat_perceptron", "isoup", "sarimax", "arma", "varma"]:
            if model_name in ["arma", "varma"]:
                for res in self.resources:
                    self.normalization_params[res] = (0.0, 1.0)
                    
            self.train_sequence = None
            self.test_sequence = None
            
            # Inicializa o modelo imediatamente e encerra o __init__
            self.model = self.__get_model(
                model_name, path_to_save_weights, path_to_load_model
            )
            return

        sequence = sequence[self.resources].copy()
        if use_normalization:
            for resource in self.resources:
                sequence[resource], s_min, s_max = normalize(sequence[resource])
                self.normalization_params[resource] = (s_min, s_max)
        self.train_sequence, self.test_sequence = split_sets(sequence, 0.8)

        self.model = self.__get_model(
            model_name, path_to_save_weights, path_to_load_model
        )

    def __get_model(
        self,
        model_name: str,
        path_to_save_weights: str | None,
        path_to_load_model: str | None,
    ) -> Model:
        match model_name:
            case "ma":
                return MovingAverage(normalization_params=self.normalization_params)
            case "h_lstm":
                model = HLSTM(
                    n_features=len(self.resources),
                    normalization_params=self.normalization_params,
                    path_to_save_weights=path_to_save_weights,
                )
                if path_to_load_model:
                    model.load(path_to_load_model)
                return model
            case "arma":
                return ARMA(
                    normalization_params=self.normalization_params,
                    path_to_save_weights=path_to_save_weights   
                )
            case "varma": 
                model = VARMA(
                    normalization_params=self.normalization_params, 
                    path_to_save_weights=path_to_save_weights
                )
                if path_to_load_model:
                    model.load(path_to_load_model) 
                return model
            case "sarimax":
                model = SARIMAX(
                    normalization_params=self.normalization_params, 
                    path_to_save_weights=path_to_save_weights
                )
                if path_to_load_model:
                    model.load(path_to_load_model)
                return model
            case "arf":
                return AdaptiveRandomForest(resources=self.resources)
            case "hat_perceptron":
                return HoeffdingAdaptiveTreePerceptron(resources=self.resources)
            case "isoup": 
                return iSOUP(resources=self.resources)
            case _:
                raise ValueError("Model not found")

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
