import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import RegressionResultsWrapper
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Ignora avisos de convergência para manter o log limpo durante o treino rápido
warnings.simplefilter('ignore', ConvergenceWarning)

from src.models.model import Model
from src.utils import denormalize 

class ARMA(Model):
    def __init__(self, normalization_params: dict[str, tuple[float, float]], path_to_save_weights: str | None = None, order=(2, 0, 1)):
        self.normalization_params = normalization_params
        self.resources = list(normalization_params.keys())
        self.path_to_save_weights = path_to_save_weights
        
        # Configuração do Modelo
        self.order = order
        
        self.window_size = 60  # Tamanho do buffer (quantos passos passados ele lembra para treinar)
        self.history = {res: [] for res in self.resources} # Buffer para guardar dados recentes
        self.fitted_models = {} # Cache dos modelos já treinados
        
        # Controle de Re-treino (para economizar CPU)
        self.retrain_interval = 10  # Re-treina a cada 10 novos dados recebidos
        self.steps_since_retrain = 0

    def learn_one(self, features: dict, targets: dict):
        """
        Método essencial para o Framework Online.
        Recebe 1 dado (passo atual), atualiza o buffer e decide se re-treina.
        """
        # Atualiza o Buffer (Janela Deslizante)
        for res in self.resources:
            if res in features:
                val = features[res]
                self.history[res].append(val)
                
                # Mantém o tamanho fixo (remove o dado mais antigo se encher)
                if len(self.history[res]) > self.window_size:
                    self.history[res].pop(0)

        self.steps_since_retrain += 1

        # Re-treina periodicamente (Estratégia Pseudo-Online)
        if self.steps_since_retrain >= self.retrain_interval:
            # Só treina se tiver dados suficientes (ex: > 10 pontos)
            if len(self.history[self.resources[0]]) > 10:
                self._retrain_models()
            self.steps_since_retrain = 0

    def _retrain_models(self):
        """Método interno que re-treina os modelos usando apenas os dados do Buffer."""
        for res in self.resources:
            try:
                data = self.history[res]
                # Cria e ajusta o modelo ARMA com os dados da janela atual
                model = ARIMA(data, order=self.order)
                self.fitted_models[res] = model.fit()
            except Exception:
                # Se falhar (ex: dados constantes), mantém o modelo anterior ou ignora
                pass

    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 1000):
        """
        Faz a previsão de longo prazo usando o modelo já treinado (em cache).
        Retorna: (steps_to_failure, curves_dict)
        """
        steps_to_failure = -1
        future_curves = {}

        # Se o modelo ainda não foi treinado (início da simulação), retorna vazio
        if not self.fitted_models:
            return -1, {}

        try:
            # Gera previsões longas para cada recurso
            for res in self.resources:
                if res in self.fitted_models:
                    # Usa o modelo em cache
                    # prevê do próximo passo até max_horizon
                    future_curves[res] = self.fitted_models[res].forecast(steps=max_horizon)
                else:
                    future_curves[res] = [current_features.get(res, 0)] * max_horizon

            # Verifica onde cruza a linha de falha
            for i in range(max_horizon):
                failed = False
                for res in self.resources:
                    val = future_curves[res][i] 
                    
                    limit = thresholds.get(res, float('inf'))
                    
                    if val >= limit:
                        steps_to_failure = i
                        failed = True
                        break 
                
                if failed:
                    break 
                    
        except Exception as e:
            print(f"ARMA forecast error: {e}")
            return -1, {}

        return steps_to_failure, future_curves
    
    def predict_one(self, current_features: dict) -> dict:
        """
        Prevê apenas o próximo passo (t+1). Útil para gráficos em tempo real.
        """
        predictions = {}
        if not self.fitted_models:
            return {}

        for res in self.resources:
            if res in self.fitted_models:
                try:
                    val = self.fitted_models[res].forecast(steps=1)
                    predictions[res] = val[0]
                except Exception:
                    predictions[res] = 0.0
        return predictions

    def train(self, tr, te): pass 
    def predict(self, seq): return np.array([])
    def load(self, p): pass
    def plot_results(self): pass