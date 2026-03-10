import numpy as np
import copy
from river import time_series
from river import multioutput
from river import base

from src.models.online_model import OnlineModel

class SNARIMAX_Adapter(base.Regressor):
    def __init__(self, p=1, d=1, q=1):
        self.p = p
        self.d = d
        self.q = q
        self.model = time_series.SNARIMAX(p=self.p, d=self.d, q=self.q)
        
    def learn_one(self, x, y):
        self.model.learn_one(y=y, x=x)
        return self
        
    def predict_one(self, x):
        try:
            res = self.model.forecast(horizon=1, xs=[x] if x else None)
            return res[0]
        except Exception:
            return 0.0

class ARIMAX(OnlineModel):
    def __init__(self, normalization_params: dict[str, tuple[float, float]] = None, path_to_save_weights: str | None = None):
        self.normalization_params = normalization_params if normalization_params else {}
        self.path_to_save_weights = path_to_save_weights
        
        # Define a ordem causal
        base_order = ['CPU', 'Mem', 'Swap', 'DiskSpace']
        if normalization_params:
            self.resources = [r for r in base_order if r in normalization_params.keys()]
            for r in normalization_params.keys():
                if r not in self.resources:
                    self.resources.append(r)
        else:
            self.resources = base_order

        # realiza normalização dos dados
        self.max_values = {res: 1.0 for res in self.resources}

        # Instancia o motor e a cadeia
        base_model = SNARIMAX_Adapter(p=1, d=1, q=1)
        self.model = multioutput.RegressorChain(
            model=base_model,
            order=self.resources
        )

    def learn_one(self, features: dict, targets: dict):
        """Treina o modelo convertendo os dados para o 'mundo miniatura'."""
        
        # Atualiza o registo dos maiores valores vistos
        for res in self.resources:
            val = abs(features.get(res, 0.0))
            if val > self.max_values[res]:
                self.max_values[res] = val

        # Normaliza
        y_norm = {res: features.get(res, 0.0) / self.max_values[res] for res in self.resources}
        x_norm = {} 
        
        # Treina
        self.model.learn_one(x=x_norm, y=y_norm)

    def predict_one(self, features: dict) -> dict:
        """Prevê e expande o valor de volta à escala real."""
        x_norm = {}
        pred_norm = self.model.predict_one(x=x_norm)
        
        # Desnormaliza
        pred_real = {k: max(0.0, v * self.max_values[k]) for k, v in pred_norm.items()}
        return pred_real

    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 1000):
        steps_to_failure = -1
        predictions_path = []
        
        sim_model = copy.deepcopy(self.model)
        current_x_norm = {}
        
        for i in range(max_horizon):
            # Prevê usando o modelo normalizado
            pred_norm = sim_model.predict_one(x=current_x_norm)
            
            # Desnormaliza para verificar os Limites de Falha
            pred_real = {k: max(0.0, v * self.max_values[k]) for k, v in pred_norm.items()}
            predictions_path.append(pred_real)
            
            # Verifica se alguma variável ultrapassou o threshold real
            failed = False
            for res in self.resources:
                if pred_real.get(res, 0.0) >= thresholds.get(res, float('inf')):
                    steps_to_failure = i
                    failed = True
                    break
            
            if failed:
                break
                
            # O Clone aprende do próprio erro
            sim_model.learn_one(x=current_x_norm, y=pred_norm)
            
        return steps_to_failure, predictions_path