from river import tree
from river import multioutput
from src.models.online_model import OnlineModel

class HoeffdingAdaptiveTreePerceptron(OnlineModel):
    def __init__(self, resources: list[str], seed: int = 42):
        self.resources = resources
        
        # O RegressorChain permite que o modelo aprenda a relação entre as variáveis.
        # Ele treina um modelo para cada alvo, mas o modelo N usa as saídas dos 
        # modelos 1 a N-1 como features adicionais.
        self.model = multioutput.RegressorChain(
            model=tree.HoeffdingAdaptiveTreeRegressor(
                seed=seed,
                leaf_prediction='adaptive', # Adaptativo para captar mudanças de conceito
                # perido que o modelo deve esperar antes de considerar criar um novo nó
                # valores baixos podem causar overfitting, valores altos podem atrasar a adaptação
                # 50 x 30min = 25hrs 
                #criará regras novas a cada 25 horas de dados
                grace_period=50 
            ),
            order=resources # Define a ordem de dependência entre os recursos
        )

    def learn_one(self, features: dict, targets: dict):
        """
        Treina o modelo multivariado de uma vez.
        'targets' deve ser um dicionário com as chaves definidas em self.resources.
        """
        # Filtramos o target para garantir que apenas os recursos monitorados entrem
        target_values = {res: targets[res] for res in self.resources if res in targets}
        
        if target_values:
            self.model.learn_one(features, target_values)

    def predict_one(self, features: dict) -> dict:
        """
        Retorna previsões para todos os recursos simultaneamente.
        """
        return self.model.predict_one(features)
    
    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 100):
        """
        Simula o futuro recursivamente considerando a correlação entre variáveis.
        """
        predictions_path = []
        next_features = current_features.copy()
        steps_to_failure = -1

        for i in range(max_horizon):
            # No modelo multivariado, predict_one já retorna o dicionário completo
            step_prediction = self.model.predict_one(next_features)
            
            # Garantir valores não negativos 
            step_prediction = {k: max(0, v) for k, v in step_prediction.items()}

            predictions_path.append(step_prediction)

            # Verificação de falha em qualquer um dos recursos
            failed = False
            for res, limit in thresholds.items():
                if step_prediction.get(res, 0) >= limit:
                    steps_to_failure = i
                    failed = True
                    break
            
            if failed:
                break

            # Feedback loop: as previsões de agora tornam-se as entradas do próximo passo
            next_features.update(step_prediction)

        return steps_to_failure, predictions_path

    def get_metrics(self) -> dict:
        # River mantém métricas internas se você adicionar um objeto river.metrics
        return {}