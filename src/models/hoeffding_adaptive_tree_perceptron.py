from river import tree
from src.models.online_model import OnlineModel

class HoeffdingAdaptiveTreePerceptron(OnlineModel):
    def __init__(self, resources: list[str], seed: int = 42):
        self.resources = resources
        
        # Cria um modelo HAT separado para cada recurso
        self.models = {}

        for res in resources:
            if res == "CPU":
                # Para CPU usa apenas a MÉDIA. 
                # não explode e ignora picos falsos.
                strategy = "mean"
            else:
                # Para Memória usa ADAPTIVE.
                strategy = "adaptive"
            self.models[res] = tree.HoeffdingAdaptiveTreeRegressor(
                seed=seed,
                leaf_prediction=strategy, # Usa a estratégia definida acima
                grace_period=200 # Um valor mais alto ajuda a estabilizar
            )

    def learn_one(self, features: dict, targets: dict):
        """
        Treina os modelos.
        """
        for res in self.resources:
            if res in targets:
                self.models[res].learn_one(features, targets[res])

    def predict_one(self, features: dict) -> dict:
        """
        Retorna previsões.
        """
        predictions = {}
        for res in self.resources:
            predictions[res] = self.models[res].predict_one(features)
        
        return predictions
    
    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 1000):
        """
        Simula o futuro recursivamente.
        """
        predictions_path = []
        next_features = current_features.copy()
        steps_to_failure = -1

        for i in range(max_horizon):
            step_prediction = {}
            
            # Prever cada recurso
            for res in self.resources:
                # O modelo prevê baseado no estado anterior (next_features)
                pred = self.models[res].predict_one(next_features)
                
                # Trava de segurança (não negativo)
                step_prediction[res] = max(0, pred)

            predictions_path.append(step_prediction)

            # Verificar Falha 
            failed = False
            for res, limit in thresholds.items():
                # Se algum recurso passar do limite, marca a falha
                if step_prediction.get(res, 0) >= limit:
                    steps_to_failure = i
                    failed = True
                    break
            
            if failed:
                break

            # Recursão
            # As features para o próximo passo são as previsões atuais
            next_features = step_prediction.copy()

        return steps_to_failure, predictions_path

    def get_metrics(self) -> dict:
        return {}