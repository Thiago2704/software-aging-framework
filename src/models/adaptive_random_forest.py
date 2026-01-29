from river.forest import ARFRegressor
from src.models.online_model import OnlineModel
from collections import deque
import numpy as np

class AdaptiveRandomForest(OnlineModel):
    def __init__(self, resources: list[str], n_models: int = 10, seed: int = 42, window_size: int = 60):
        self.resources = resources
        self.window_size = window_size
        # Cria um modelo ARF separado para cada recurso (CPU, Mem)
        self.models = {
            res: ARFRegressor(
                n_models=n_models,
                seed=seed
            ) 
            for res in resources
        }
        # BUFFER INTERNO: Cada modelo gerencia seu histórico
        self.rolling_windows = {
            res: deque(maxlen=window_size) 
            for res in resources
        }

    def __extract_features(self, raw_features: dict) -> dict:
        """
        Método privado que transforma dados brutos em estatísticas (Média, Max, Std).
        """
        enriched_features = {}
        
        # Atualiza a janela com os dados novos
        for res, value in raw_features.items():
            self.rolling_windows[res].append(value)
            
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

    def learn_one(self, features: dict, targets: dict):
        """
        Treina os modelos.
        features: Dicionário com os dados de entrada (ex: {'CPU': 10.5, 'Mem': 500})
        targets: Dicionário com os valores reais que aconteceram (resposta)
        """
        enriched_features = self.__extract_features(features)
        
        for res in self.resources:
            if res in targets:
                # O modelo de CPU aprende a prever CPU usando todas as features
                self.models[res].learn_one(enriched_features, targets[res])

    def predict_one(self, features: dict) -> dict:
        """
        Retorna previsões para todos os recursos.
        """
        enriched_features = {}
        for res in self.resources:
            # Pega o histórico existente
            data = list(self.rolling_windows[res])
            # Adiciona o dado ATUAL temporariamente para o cálculo
            data.append(features[res])

            enriched_features[f"{res}_mean"] = np.mean(data)
            enriched_features[f"{res}_max"] = np.max(data)
            enriched_features[f"{res}_std"] = np.std(data)

        predictions = {}
        for res in self.resources:
            predictions[res] = self.models[res].predict_one(enriched_features)
        
        return predictions
    
    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 1000):
        """
        Realiza a previsão recursiva a longo prazo simulando o futuro.
        """
        steps_to_failure = -1
        predictions_path = []

        # clonar as janelas (Para não sujar o histórico real com previsões)
        # Precisamos de cópias independentes para simular o futuro
        simulated_windows = {
            res: list(self.rolling_windows[res]) 
            for res in self.resources
        }
        
        # Se as janelas estiverem vazias (início da execução), preenche com o valor atual
        for res in self.resources:
            if not simulated_windows[res]:
                simulated_windows[res] = [current_features[res]]

        # loop de simulação para cada passo futuro
        for i in range(max_horizon):
            
            # Calcular Features Baseadas na Janela Simulada
            step_features = {}
            for res in self.resources:
                data = simulated_windows[res]
                step_features[f"{res}_mean"] = np.mean(data)
                step_features[f"{res}_max"] = np.max(data)
                step_features[f"{res}_std"] = np.std(data)

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
                    steps_to_failure = i
                    failed = True
                    break
            
            if failed:
                break

            # Atualizar a janela simulada (Recursão)
            # A previsão de agora vira o passado do próximo passo
            for res in self.resources:
                simulated_windows[res].append(step_prediction[res])
                # Mantém o tamanho fixo da janela 
                if len(simulated_windows[res]) > self.window_size: # Assumindo window_size=60
                    simulated_windows[res].pop(0)

        return steps_to_failure, predictions_path