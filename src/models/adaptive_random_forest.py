from river.forest import ARFRegressor
from river import compose, preprocessing
from src.models.online_model import OnlineModel
from collections import deque
import numpy as np

class AdaptiveRandomForest(OnlineModel):
    def __init__(self, resources: list[str], n_models: int = 10, seed: int = 42, window_size: int = 60):
        
        self.resources = list(resources)
        self.window_size = window_size
        # Cria um modelo ARF separado para cada recurso (CPU, Mem)

        self.models = {}
        for res in self.resources:
            base_arf = ARFRegressor(n_models=n_models, seed=seed)
            # Impede que a árvore sofra overflow calculando variância em milhões
            self.models[res] = compose.Pipeline(
                preprocessing.TargetStandardScaler(regressor=base_arf)
            )
        # BUFFER INTERNO: Cada modelo gerencia seu histórico
        # Buffer agora começa vazio e adapta-se dinamicamente a variáveis auxiliares
        self.rolling_windows = {}

    def __extract_features(self, raw_features: dict) -> dict:
        """
        Método privado que transforma dados brutos em estatísticas (Média, Max, Std).
        """
        enriched_features = {}
        
        # Atualiza as janelas, criando novas automaticamente se aparecer 'Frag_1' ou 'DiskIO'
        for key, value in raw_features.items():
            if key not in self.rolling_windows:
                self.rolling_windows[key] = deque(maxlen=self.window_size)
            self.rolling_windows[key].append(value)
            
        # Calcula as estatísticas
        for res in self.resources:
            # Converte deque para lista para cálculo
            data = list(self.rolling_windows[res])
            
            if not data: # Proteção se estiver vazio
                enriched_features[f"{res}_mean"] = 0.0
                enriched_features[f"{res}_max"] = 0.0
                enriched_features[f"{res}_std"] = 0.0
                continue

            # Cria as 3 visões
            enriched_features[f"{res}_mean"] = np.mean(data)
            enriched_features[f"{res}_max"] = np.max(data)
            enriched_features[f"{res}_std"] = np.std(data)
            
        return enriched_features

    def learn_one(self, features: dict, target: dict):
        """
        Treina os modelos.
        features: Dicionário com os dados de entrada (ex: {'CPU': 10.5, 'Mem': 500})
        target: Dicionário com o valor real que aconteceu (resposta)
        """
        enriched_features = self.__extract_features(features)
        
        for res in self.resources:
            if res in target:
                # O modelo de CPU aprende a prever CPU usando todas as features
                self.models[res].learn_one(enriched_features, target[res])

    def predict_one(self, features: dict) -> dict:
        """
        Retorna previsões para todos os recursos.
        """
        enriched_features = {}
        for key, value in features.items():
            # Pega o histórico existente
            data = list(self.rolling_windows.get(key, []))
            # Adiciona o dado ATUAL temporariamente para o cálculo
            data.append(value)

            enriched_features[f"{key}_mean"] = np.mean(data)
            enriched_features[f"{key}_max"] = np.max(data)
            enriched_features[f"{key}_std"] = np.std(data)

        predictions = {}
        for res in self.resources:
            pred = self.models[res].predict_one(enriched_features)
            # Garantir que a previsão não seja negativa quando a arvore não tiver aprendido o suficiente
            predictions[res] = max(0.0, pred if pred is not None else 0.0)

        
        return predictions
    
    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 336):
        """
        Realiza a previsão recursiva a longo prazo simulando o futuro.
        """
        steps_to_failure = -1
        predictions_path = []

        # clonar as janelas (Para não sujar o histórico real com previsões)
        # Precisamos de cópias independentes para simular o futuro
        simulated_windows = {
            key: list(window) for key, window in self.rolling_windows.items()
        }
        
        # Se as janelas estiverem vazias (início da execução), preenche com o valor atual
        for key, value in current_features.items():
            if key not in simulated_windows or not simulated_windows[key]:
                simulated_windows[key] = [value]

        # Mantém variáveis exógenas fixas durante a simulação
        next_exog = current_features.copy()

        # loop de simulação para cada passo futuro
        for i in range(max_horizon):
            
            # Calcular Features Baseadas na Janela Simulada
            step_features = {}
            for key, data in simulated_windows.items():
                step_features[f"{key}_mean"] = np.mean(data)
                step_features[f"{key}_max"] = np.max(data)
                step_features[f"{key}_std"] = np.std(data)

            # Prever o Próximo Passo
            step_prediction = {}
            for res in self.resources:
                pred = self.models[res].predict_one(step_features)
                # Garantir que a previsão não seja negativa
                step_prediction[res] = max(0, pred) 

            predictions_path.append(step_prediction)

            # Verificar Falha (Threshold)
            # Verifica se algum recurso estourou o limite
            failed = False
            for res in self.resources:
                limit = thresholds.get(res, float('inf'))
                if step_prediction[res] >= limit:
                    steps_to_failure = i + 1
                    failed = True
                    break
            
            if failed:
                break

            # Atualizar a janela simulada (Recursão)
            # A previsão de agora vira o passado do próximo passo
            for key in simulated_windows.keys():
                # Se for CPU/Memória, adiciona a previsão. Se for Frag_1, mantém o valor exógeno inicial
                val = step_prediction.get(key, next_exog.get(key, 0.0))
                simulated_windows[key].append(val)
                # Mantém o tamanho fixo da janela 
                if len(simulated_windows[key]) > self.window_size: # Assumindo window_size=60
                    simulated_windows[key].pop(0)

        return steps_to_failure, predictions_path