import copy
from river import time_series
from river import multioutput
from river import preprocessing
from river import compose
from river import base
import numpy as np
from src.models.online_model import OnlineModel

class SNARIMAXAdapter(base.Regressor):
    def __init__(self, p, d, q, sp, sd, sq, m):
        self.p, self.d, self.q = p, d, q
        self.sp, self.sd, self.sq, self.m = sp, sd, sq, m
        self.model = time_series.SNARIMAX(
            p=p, d=d, q=q, sp=sp, sd=sd, sq=sq, m=m
        )

    def learn_one(self, x, y):
        self.model.learn_one(y=y, x=x)
        return self

    def predict_one(self, x):
        try:
            return self.model.forecast(horizon=1, xs=[x] if x else None)[0]
        except Exception:
            return 0.0

class SARIMAX(OnlineModel):
    def __init__(self, resources: list[str], p=1, d=1, q=1, m=48, sp=1, sd=0, sq=1):
        self.resources = list(resources)
        base_model = SNARIMAXAdapter(p, d, q, sp, sd, sq, m)
        
        # O SNARIMAX do River configurado com Sazonalidade (m=48 para 24h se passo=30min)
        # o modelo atual não possui sazonalidade

        # normaliza as variáveis 'x'
        normalized_model = compose.Pipeline(
            preprocessing.StandardScaler(),
            preprocessing.TargetStandardScaler(regressor=base_model)
        )

        # RegressorChain permite que o modelo seja multivariado (correlação entre variáveis)
        # Cada recurso na lista aprende com o histórico dos recursos anteriores
        self.model = multioutput.RegressorChain(
            model=normalized_model,
            order=list(self.resources) 
        )
        

    def learn_one(self, features: dict, targets: dict):
        """
        No SARIMAX online, 'features' são variáveis exógenas (X) 
        e 'targets' são os valores reais dos recursos (Y).
        """
        if len(targets) > 0:
            # O RegressorChain do River gerencia a distribuição dos alvos
            # No SNARIMAX, o 'x' em learn_one são as variáveis exógenas
            self.model.learn_one(x=features, y=targets)

    def predict_one(self, features: dict) -> dict:
        """Prevê o próximo passo (t+1)"""
        prediction = self.model.predict_one(x=features)
        
        if not prediction:
            return {r: 0.0 for r in self.resources}
            
        # Filtra apenas os recursos que queremos prever, ignorando variáveis exógenas
        return {res: max(0.0, prediction.get(res, 0.0)) for res in self.resources}

    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 336):
        """
        Simulação recursiva. 
        """
        predictions_path = []
        steps_to_failure = -1
        
        # Criamos uma cópia do modelo atual para simular o futuro sem sujar o aprendizado real
        sim_model = copy.deepcopy(self.model)
        
        # Para a simulação, precisamos de uma entrada inicial
        # Mantemos exógenas (como Fragmentação) constantes durante a projeção
        next_exog = current_features.copy()
        
        for i in range(max_horizon):
            # Prever o próximo estado (t + i + 1)
            prediction = sim_model.predict_one(next_exog)
            
            if not prediction:
                prediction = {r: 0.0 for r in self.resources}

            # Garante valores físicos reais (sem RAM negativa) e filtra apenas recursos previstos
            prediction = {res: max(0.0, prediction.get(res, 0.0)) for res in self.resources}
            predictions_path.append(prediction)
            
            # Verificar Falha nos Thresholds
            is_failure = False
            for res in self.resources:
                limit = thresholds.get(res, float('inf'))
                val_pred = prediction.get(res, 0)
                
                if val_pred >= limit:
                    steps_to_failure = i + 1
                    is_failure = True
                    break
            
            if is_failure:
                break

            # Recursão
            sim_model.learn_one(x=next_exog, y=prediction)
            
            # Atualiza exógenas se houver dependência, ou mantém para o próximo forecast
        return steps_to_failure, predictions_path
    