import numpy as np
import copy
from river import time_series
from river import tree
from river import forest
from river import multioutput
from src.models.online_model import OnlineModel
from river import base

# classe wrapper para adaptar o SNARIMAX do River, que é um modelo de série temporal, 
# para funcionar como um regressor normal dentro da cadeia de regressão multioutput. 
# Ele encapsula o SNARIMAX e traduz as chamadas de aprendizado e previsão 
# para o formato esperado pelo modelo de série temporal.
class SNARIMAX_Wrapper:
    def __init__(self, p, d, q, regressor):
        self.model = time_series.SNARIMAX(
            p=p, d=d, q=q, 
            regressor=regressor
        )
    
    def learn_one(self, x, y):
        # O SNARIMAX espera (y, x) ao contrário do padrão do River
        self.model.learn_one(y, x)
        return self
        
    def predict_one(self, x):
        # Converte a chamada de predict_one para forecast(1)
        try:
            res = self.model.forecast(horizon=1, xs=[x])
            return res[0]
        except:
            return 0.0
        
# wrapper para o AMF, para evitar que o modelo retorne None quando estiver em dúvida,
# e retorne 0.0 em vez disso, para evitar conflito com o SNARIMAX que espera um valor numérico
class AMF_Wrapper(base.Regressor):
    def __init__(self, regressor):
        self.regressor = regressor

    def learn_one(self, x, y):
        self.regressor.learn_one(x, y)
        return self

    def predict_one(self, x):
        pred = self.regressor.predict_one(x)
        # Se a floresta não souber o que prever, devolve 0.0 em vez de colapsar o sistema
        return 0.0 if pred is None else pred

class SNARIMAX_Tree(OnlineModel):
    def __init__(self, resources: list[str], tree_type: str = 'HAT', p: int = 12, d: int = 1, q: int = 1):
        self.resources = resources
        self.step_count = 0
        
        # Limites Físicos
        self.max_values = {
            'CPU': 100.0,
            'Mem': 16000000.0,
            'Swap': 8000000.0,
            'DiskSpace': 500000000.0
        }

        # Escolhe o modelo
        match tree_type.upper(): 
            # Árvores  
            case 'SNARIMAX_HT': 
                base_regressor = tree.HoeffdingTreeRegressor(
                    grace_period=20, # Número de amostras para considerar uma divisão
                    leaf_prediction='mean'
                )
            case 'SNARIMAX_HAT': 
                base_regressor = tree.HoeffdingAdaptiveTreeRegressor(
                    grace_period=20, 
                    leaf_prediction='adaptive',
                    seed=42
                )  

            # Florestas
            case 'SNARIMAX_ARF': # Adaptive Random Forest
                # árvore base: HAT (Adaptive Hoeffding Tree)
                base_regressor = forest.ARFRegressor(
                    n_models=10, # Número de árvores na floresta
                    grace_period=20, 
                    leaf_prediction='adaptive',
                    seed=42
                )
            case 'SNARIMAX_AMF': # Aggregated Mondrian Forest
                # árvore base: árvore de Mondrian
                amf_base = forest.AMFRegressor(
                    n_estimators=10, # Número de árvores na floresta
                    step=0.1, # Passo de aprendizado
                    seed=42
                )
                # Envolve o AMF para garantir que ele nunca retorne None
                base_regressor = AMF_Wrapper(amf_base) 
            case 'SNARIMAX_OXT': # Online Extra Trees
                # árvore base: HT (Hoeffding Tree)
                base_regressor = forest.OXTRegressor(
                    n_models=10,
                    grace_period=20,
                    leaf_prediction='adaptive',
                    seed=42
                )
            case _: 
                raise ValueError(f"Motor'{tree_type}' não suportado.")

        # ====================================================================
        # Encapsulamento Universal no SNARIMAX
        # ====================================================================
        # INJEÇÃO DO ADAPTADOR: passa o Wrapper em vez do SNARIMAX puro
        wrapped_snarimax = SNARIMAX_Wrapper(
            p=p, d=d, q=q, 
            regressor=base_regressor
        )
        
        self.model = multioutput.RegressorChain(
            model=wrapped_snarimax,
            order=self.resources
        )

    def learn_one(self, features: dict, targets: dict):
        self.step_count += 1
        
        # Alvos normalizados
        y_norm = {res: targets.get(res, 0.0) / self.max_values[res] for res in self.resources}
        
        # Relógio logarítmico
        x_norm = {'time_step': np.log1p(self.step_count) / 10.0}
        
        self.model.learn_one(x=x_norm, y=y_norm)

    def predict_one(self, features: dict) -> dict:
        x_norm = {'time_step': np.log1p(self.step_count + 1) / 10.0}
        
        pred_norm = self.model.predict_one(x=x_norm)
        
        # Desnormaliza
        return {k: max(0.0, v * self.max_values[k]) for k, v in pred_norm.items()}

    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 1000):
        predictions_path = []
        steps_to_failure = -1
        
        sim_model = copy.deepcopy(self.model)
        future_step = self.step_count

        for i in range(max_horizon):
            future_step += 1
            x_future = {'time_step': np.log1p(future_step) / 10.0}
            
            pred_norm = sim_model.predict_one(x=x_future)
            pred_real = {k: max(0.0, v * self.max_values[k]) for k, v in pred_norm.items()}
            predictions_path.append(pred_real)

            failed = False
            for res in self.resources:
                if pred_real.get(res, 0.0) >= thresholds.get(res, float('inf')):
                    if steps_to_failure == -1:
                        steps_to_failure = i + 1
                    failed = True
                    break
            
            if failed: 
                break
                
            sim_model.learn_one(x=x_future, y=pred_norm)
            
        return steps_to_failure, predictions_path

    def get_metrics(self) -> dict:
        return {}