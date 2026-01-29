import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

# Ignorar avisos de convergência para não poluir o log em tempo real
warnings.simplefilter('ignore', ConvergenceWarning)

from src.models.model import Model
from src.utils import denormalize 

class SARIMAX(Model):
    def __init__(self, normalization_params: dict[str, tuple[float, float]] = None, path_to_save_weights: str | None = None):
        self.normalization_params = normalization_params if normalization_params else {}
        self.resources = list(normalization_params.keys()) if normalization_params else []
        self.path_to_save_weights = path_to_save_weights
        
        # Buffer para armazenar histórico recente de cada recurso separadamente
        self.window_size = 60 
        self.history = {res: [] for res in self.resources}
        self.fitted_models = {}
        
        # Controle de re-treino
        self.retrain_interval = 10 
        self.steps_since_retrain = 0

        # Hiperparâmetros SARIMAX 
        self.order = (1, 1, 1) 
        # IMPORTANTE: Seasonal order (s=60) exige 60 pontos MÍNIMO para rodar.
        # Se window_size < 60, vai dar erro. Vamos usar s=0 (sem sazonalidade) no online para ser rápido,
        # ou garantir que o buffer tenha tamanho suficiente.
        self.seasonal_order = (0, 0, 0, 0) 

    def learn_one(self, features: dict, targets: dict):
        """Simula aprendizado online."""
        if not self.resources:
            self.resources = list(features.keys())
            self.history = {res: [] for res in self.resources}

        # Adiciona ao histórico
        for res in self.resources:
            if res in features:
                self.history[res].append(features[res])
                if len(self.history[res]) > self.window_size:
                    self.history[res].pop(0)

        self.steps_since_retrain += 1

        # Re-treina periodicamente
        if self.steps_since_retrain >= self.retrain_interval:
            # Só treina se tiver dados mínimos (ex: 20 pontos)
            if len(self.history[self.resources[0]]) > 20:
                self._retrain_models()
            self.steps_since_retrain = 0

    def _retrain_models(self):
        """Reajusta um modelo para cada recurso."""
        for res in self.resources:
            try:
                data = self.history[res]
                # Cria e treina o modelo do zero com o buffer atual
                model = sm.tsa.statespace.SARIMAX(
                    data,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    trend='t',
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.fitted_models[res] = model.fit(disp=False)
            except Exception:
                pass

    def predict_one(self, features: dict) -> dict:
        """Prevê o próximo passo."""
        pred_dict = {}
        for res in self.resources:
            if res in self.fitted_models:
                try:
                    # Forecast de 1 passo
                    val = self.fitted_models[res].forecast(steps=1)[0]
                    pred_dict[res] = max(0, val)
                except:
                    pred_dict[res] = features.get(res, 0)
            else:
                pred_dict[res] = features.get(res, 0)
        return pred_dict

    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 1000):
        """Projeta o futuro."""
        steps_to_failure = -1
        predictions_path = []

        if not self.fitted_models:
            return -1, []

        try:
            # Gera previsões para cada recurso separadamente
            forecasts = {}
            for res in self.resources:
                if res in self.fitted_models:
                    forecasts[res] = self.fitted_models[res].forecast(steps=max_horizon)
                else:
                    # Se não tem modelo, projeta constante
                    forecasts[res] = [current_features[res]] * max_horizon

            # Combina os resultados passo a passo
            for i in range(max_horizon):
                step_pred = {}
                failed = False
                
                for res in self.resources:
                    val = max(0, forecasts[res][i])
                    step_pred[res] = val
                    
                    limit = thresholds.get(res, float('inf'))
                    if val >= limit:
                        steps_to_failure = i
                        failed = True
                
                predictions_path.append(step_pred)
                if failed:
                    break
                    
        except Exception as e:
            print(f"SARIMAX forecast error: {e}")
            return -1, []

        return steps_to_failure, predictions_path

    # Métodos base (placeholders)
    def train(self, tr, te): pass
    def predict(self, seq): return np.array([])
    def load(self, p): pass
    def plot_results(self): pass
    def get_metrics(self): return {}