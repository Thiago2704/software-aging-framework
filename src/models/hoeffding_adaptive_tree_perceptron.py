from river import tree
from river import multioutput
from src.models.online_model import OnlineModel
from river import compose
from river import preprocessing

class HoeffdingAdaptiveTreePerceptron(OnlineModel):
    def __init__(self, resources: list[str], seed: int = 42):
        self.resources = resources
        print("ordem dos recursos no modelo:", self.resources)

        # Inicia com 1.0 para evitar divisão por zero
        self.target_max = {res: 1.0 for res in resources}
        
        # Inserindo o StandardScaler 
        base_tree = compose.Pipeline(
            preprocessing.StandardScaler(),
            tree.HoeffdingAdaptiveTreeRegressor(
                seed=seed,
                leaf_prediction='adaptive', 
                grace_period=40
            )
        )

        self.model = multioutput.RegressorChain(
            model=base_tree,
            order=resources 
        )

    def learn_one(self, features: dict, targets: dict):
        """
        Treina o modelo multivariado aplicando o Fator de Esquecimento.
        """
        target_values = {}
        for res in self.resources:
            if res in targets:
                valor_real = targets[res]
                
                # 1. FATOR DE ESQUECIMENTO (DECAY)
                # Usamos 0.995 (ou 0.999) para descer o teto devagar e não chocar os pesos do Perceptron
                self.target_max[res] = max(valor_real, self.target_max[res] * 0.995)
                
                # 2. Normaliza o alvo garantindo que não divide por zero
                teto_atual = max(self.target_max[res], 1e-6)
                target_values[res] = valor_real / teto_atual
        
        if target_values:
            self.model.learn_one(features, target_values)

    def predict_one(self, features: dict) -> dict:
        """
        Retorna previsões desnormalizadas (em valores reais como KB ou %).
        """
        previsoes_normalizadas = self.model.predict_one(features)
        
        # Se vier vazio (cold start)
        if not previsoes_normalizadas:
             return {r: 0.0 for r in self.resources}
        
        # 3. Desnormaliza multiplicando pelo teto elástico atual
        previsoes_reais = {}
        for res, valor_norm in previsoes_normalizadas.items():
            teto_atual = self.target_max.get(res, 1.0)
            previsoes_reais[res] = valor_norm * teto_atual
            
        return previsoes_reais
    
    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 100):
        predictions_path = []
        next_features = current_features.copy()
        steps_to_failure = -1

        for i in range(max_horizon):
            # No modelo multivariado, predict_one já retorna o dicionário completo desnormalizado
            step_prediction = self.predict_one(next_features)
            
            # Garantir valores não negativos e preencher vazios se cold start
            if not step_prediction:
                step_prediction = {r: 0.0 for r in self.resources}
            step_prediction = {k: max(0, v) for k, v in step_prediction.items()}

            predictions_path.append(step_prediction)

            # Verificação de falha em qualquer um dos recursos
            failed = False
            for res, limit in thresholds.items():
                if step_prediction.get(res, 0) >= limit:
                    steps_to_failure = i + 1  # Ajustado para ficar igual ao iSOUP
                    failed = True
                    break
            
            if failed:
                break

            # Feedback loop
            next_features.update(step_prediction)

        return steps_to_failure, predictions_path

    def get_metrics(self) -> dict:
        return {}