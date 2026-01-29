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
                leaf_prediction='model', 
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
    
    def predict_until_failure(self, current_features: dict, memory_threshold: float, max_horizon: int = 1000):
        predictions_path = []
        next_features = current_features.copy()
        steps_to_failure = -1
        
        # Quantos passos o modelo olha para trás
        NUM_LAGS = 3  # Exemplo: Se usa lag_1, lag_2 e lag_3
        
        for i in range(max_horizon):
            # Prever
            prediction = self.model.predict_one(next_features)

            for key in prediction:
                prediction[key] = max(0, prediction[key])

            predictions_path.append(prediction)
            
            # Verificar Falha (Supondo que a chave seja 'Mem')
            predicted_mem = prediction.get('Mem', 0)
            if predicted_mem >= memory_threshold:
                steps_to_failure = i
                break
            
            # Atualizar a esteira (shift)
            # Primeiro movemos os antigos para trás (do último para o primeiro)
            # Ex: lag_3 = lag_2, lag_2 = lag_1
            # Isso é vital! Se atualizar o lag_1 primeiro, você perde o valor dele antes de passar pro lag_2.
            for lag in range(NUM_LAGS, 1, -1):
                # Se tiver features de CPU também, tem que fazer para CPU
                if f'Mem_lag_{lag-1}' in next_features:
                    next_features[f'Mem_lag_{lag}'] = next_features[f'Mem_lag_{lag-1}']
                
                if f'CPU_lag_{lag-1}' in next_features: # Exemplo se tiver CPU
                    next_features[f'CPU_lag_{lag}'] = next_features[f'CPU_lag_{lag-1}']

            # Inserir a nova previsão na "boca" da esteira (lag_1)
            next_features['Mem_lag_1'] = predicted_mem
            # Se o modelo também prevê CPU e usa lag de CPU, atualize também:
            if 'CPU' in prediction:
                 next_features['CPU_lag_1'] = prediction['CPU']

        return steps_to_failure, predictions_path

    def get_metrics(self) -> dict:
        return {}