from river import tree
from river import compose
from river import preprocessing
from src.models.online_model import OnlineModel

class iSOUP(OnlineModel):
    def __init__(self, resources: list[str], seed: int = 42):
        self.resources = resources
        
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            tree.iSOUPTreeRegressor(
                leaf_prediction='model',  # model ou adaptive
                #quantidade de amostras que o modelo vê antes de decidir se deve criar um novo galho
                # reduzir grace period se o passo for grande, 
                #ex: passo de 60seg, 300x60 = 18.000seg= 5hrs
                #o modelo só vai aprender novas regras a cada 5hrs
                #considerar reduzir para 50
                grace_period=50
            )
        )

    def learn_one(self, features: dict, targets: dict):
        if len(targets) > 0:
            self.model.learn_one(features, targets)

    def predict_one(self, features: dict) -> dict:
        return self.model.predict_one(features)
        
    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 100):
        """
        Simula o futuro passo-a-passo recursivamente.
        Entrada: Estado Atual (t) -> Prevê (t+1) -> Usa (t+1) como entrada para prever (t+2)...
        """
        predictions_path = []
        
        next_input = current_features.copy()
        
        steps_to_failure = -1
        
        for i in range(max_horizon):
            # Prever o próximo estado
            prediction = self.model.predict_one(next_input)
            
            # Se a previsão vier vazia (modelo frio), retorna zeros
            if not prediction:
                prediction = {r: 0.0 for r in self.resources}

            # Garante valores não negativos
            for key in prediction:
                prediction[key] = max(0.0, prediction[key])

            predictions_path.append(prediction)
            
            # Verificar Falha 
            is_failure = False
            for res in self.resources:
                # Pega o limite do dicionário (ex: {'Mem': 8000, 'CPU': 90})
                limit = thresholds.get(res, float('inf'))
                val_pred = prediction.get(res, 0)
                
                if val_pred >= limit:
                    steps_to_failure = i + 1 # Falha prevista no passo i+1
                    is_failure = True
                    break # Sai do loop de recursos
            
            if is_failure:
                break # Sai do loop de horizonte 

            # Atualizar a Entrada para o próximo passo (Recursão)
            for res in self.resources:
                if res in prediction:
                    next_input[res] = prediction[res]

        return steps_to_failure, predictions_path

    def get_metrics(self) -> dict:
        return {}