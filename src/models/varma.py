import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning

# Filtros para limpar o log 
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', EstimationWarning)
warnings.filterwarnings("ignore", module="statsmodels")

from src.models.model import Model

class VARMA(Model):
    def __init__(self, normalization_params: dict[str, tuple[float, float]], path_to_save_weights: str | None = None, order=(1, 0)):
        self.normalization_params = normalization_params
        # Garante a ordem fixa das colunas (Importante para VARMA)
        self.resources = list(normalization_params.keys()) 
        self.path_to_save_weights = path_to_save_weights
        
        # Configuração do Modelo
        # Nota: (1, 0) é mais estável para online. (2, 1) pode quebrar com poucos dados.
        self.order = order 
        
        self.window_size = 60  # Tamanho do histórico recente
        self.history = []      # Lista de listas [[cpu, mem], [cpu, mem]...]
        self.fitted_model = None 
        
        # Controle de Re-treino
        # Número de passos entre re-treinos
        self.retrain_interval = 10 
        self.steps_since_retrain = 0

    def learn_one(self, features: dict, targets: dict):
        """
        Recebe um dicionário {'CPU': 0.5, 'Mem': 0.3}, converte para lista ordenada
        e atualiza o buffer.
        """
        # Converte dict para lista na ordem correta [Valor_Recurso1, Valor_Recurso2...]
        row = []
        for res in self.resources:
            row.append(features.get(res, 0.0))
        
        # Atualiza Buffer
        self.history.append(row)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        self.steps_since_retrain += 1

        # Re-treina se tiver dados suficientes
        if self.steps_since_retrain >= self.retrain_interval:
            # VARMA precisa de dados mínimos para rodar (min 15 pontos)
            if len(self.history) > 15: 
                self._retrain_model()
            self.steps_since_retrain = 0

    def _retrain_model(self):
        """Treina UM modelo que entende todas as variáveis juntas."""
        try:
            # Converte histórico para DataFrame (VARMAX exige isso ou array 2D)
            df_history = pd.DataFrame(self.history, columns=self.resources)
            
            # Cria o modelo VARMAX
            model = VARMAX(df_history, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
            
            # Treina
            self.fitted_model = model.fit(disp=False, maxiter=100)
            
        except Exception:
            pass

    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 1000):
        """
        Projeta o futuro de TODAS as variáveis simultaneamente.
        """
        steps_to_failure = -1
        future_curves = {res: [] for res in self.resources}

        if self.fitted_model is None:
            return -1, {}

        try:
            # Forecast Multivariado 
            forecast_df = self.fitted_model.forecast(steps=max_horizon)
            
            # Converte para dicionário de listas para facilitar a verificação
            # {'CPU': [0.1, 0.2...], 'Mem': [0.5, 0.51...]}
            predictions_dict = forecast_df.to_dict(orient='list')

            # Verifica falha
            for i in range(max_horizon):
                failed = False
                for res in self.resources:
                    # Pega o valor previsto no passo i
                    val = predictions_dict[res][i]
                    limit = thresholds.get(res, float('inf'))
                    
                    if val >= limit:
                        steps_to_failure = i
                        failed = True
                        break 
                
                if failed:
                    break
            
            return steps_to_failure, predictions_dict

        except Exception as e:
            print(f"VARMA forecast error: {e}")
            return -1, {}

    def predict_one(self, current_features: dict) -> dict:
        """Prevê apenas o próximo passo."""
        if self.fitted_model is None:
            # Cold Start: Retorna o valor atual como chute
            return current_features
        
        try:
            forecast_df = self.fitted_model.forecast(steps=1)
            # Pega a primeira linha e converte para dict
            return forecast_df.iloc[0].to_dict()
        except:
            return current_features

    # Métodos obrigatórios da classe base (Offline) 
    def train(self, tr, te): pass
    def predict(self, seq): return np.array([])
    def load(self, p): pass
    def plot_results(self): pass