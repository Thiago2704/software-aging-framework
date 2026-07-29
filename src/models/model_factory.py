# src/models/model_factory.py

from src.models import (
    HLSTM, MovingAverage, SARIMAX, ARIMAX,
    AdaptiveRandomForest, HoeffdingAdaptiveTreePerceptron, iSOUP,
    SNARIMAX_Tree
)

class ModelFactory:
    """
    Fábrica central para instanciação de modelos preditivos.

    Implementa o padrão de projeto Factory Method para centralizar e isolar 
    a lógica de criação dos modelos.

    Para adicionar um novo modelo ao framework:
    1. Crie a classe do modelo herdando de `Model` ou `OnlineModel`.
    2. Importe a classe neste arquivo.
    3. Adicione um novo `case` no método `create_model`.
    """
    @staticmethod
    def create_model(
        model_name: str,
        resources: list[str],
        normalization_params: dict,
        path_to_save_weights: str | None = None,
        path_to_load_model: str | None = None
    ):
        match model_name:
            case "ma":
                return MovingAverage(normalization_params=normalization_params)
            
            case "h_lstm":
                model = HLSTM(
                    n_features=len(resources),
                    normalization_params=normalization_params,
                    path_to_save_weights=path_to_save_weights,
                )
                if path_to_load_model:
                    model.load(path_to_load_model)
                return model
            
            case "sarimax":
                return SARIMAX(resources=resources)
            
            case "arimax":
                model = ARIMAX(
                    normalization_params=normalization_params,
                    path_to_save_weights=path_to_save_weights
                )
                if path_to_load_model:
                    model.load(path_to_load_model)
                return model
            
            case "arf":
                return AdaptiveRandomForest(resources=resources)
            
            case "hat_perceptron":
                return HoeffdingAdaptiveTreePerceptron(resources=resources)
            
            case "isoup":
                return iSOUP(resources=resources)
            
            case ("snarimax_hat" | "snarimax_ht" | 
                  "snarimax_oxt" | "snarimax_arf" | "snarimax_amf"):
                return SNARIMAX_Tree(
                    resources=resources,
                    tree_type=model_name
                )
            
            case _:
                raise ValueError(f"O modelo '{model_name}' não foi encontrado na fábrica.")