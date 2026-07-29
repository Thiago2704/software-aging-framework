import numpy as np
import copy
from river import time_series
from river import tree
from river import forest
from river import multioutput
from src.models.online_model import OnlineModel
from river import base

class SNARIMAX_Wrapper:
    """
    Adaptador (Wrapper) para o modelo SNARIMAX da biblioteca River.

    O SNARIMAX nativo do River possui uma interface voltada para séries temporais 
    (esperando argumentos na ordem y, x e utilizando o método `forecast`). 
    Este adaptador converte essa interface para o padrão de um regressor comum 
    (ordem x, y e método `predict_one`), permitindo que o SNARIMAX seja embutido 
    dentro de cadeias de regressão (RegressorChain).

    Nota sobre RegressorChain:
        É uma técnica de regressão multivariada onde a previsão de uma variável 
        é utilizada como feature (dica) para prever a próxima variável na cadeia. 
        Isso ajuda o modelo a aprender a correlação entre os recursos do sistema 
        (ex: entender que picos de Memória podem influenciar o uso da CPU).
    """
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
        
class AMF_Wrapper(base.Regressor):
    """
    Adaptador de segurança para o modelo Aggregated Mondrian Forest (AMF).

    Florestas AMF no River podem retornar `None` durante as primeiras iterações 
    se não tiverem confiança suficiente para prever. Este wrapper atua como um *fail-safe*, 
    convertendo `None` em `0.0`.
    """
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
    """
    Modelo híbrido de aprendizado online combinando SNARIMAX e Árvores/Florestas.

    Esta classe orquestra a predição multivariada de consumo de recursos de software. 
    Ela encadeia as previsões (RegressorChain) garantindo que a predição de um 
    recurso (ex: Memória) ajude na predição do próximo (ex: CPU). Além disso, 
    gerencia automaticamente a normalização interna dos dados utilizando limites 
    físicos pré-estabelecidos e alimenta os modelos com um relógio logarítmico 
    para simular o tempo contínuo de envelhecimento (aging).
    """
    def __init__(self, resources: list[str], tree_type: str = 'SNARIMAX_HAT', p: int = 12, d: int = 1, q: int = 1):
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

        # Injeta a árvore base escolhida dentro do SNARIMAX, que é envelopado 
        # pelo Wrapper e finalmente colocado na RegressorChain
        wrapped_snarimax = SNARIMAX_Wrapper(
            p=p, d=d, q=q, 
            regressor=base_regressor
        )
        
        self.model = multioutput.RegressorChain(
            model=wrapped_snarimax,
            order=self.resources
        )

    def learn_one(self, features: dict, targets: dict):
        """
        Aprende um único passo utilizando normalização interna.

        Converte os alvos absolutos para o intervalo [0, 1] com base nos 
        limites de `max_values` e cria uma feature sintética de tempo (relógio 
        logarítmico) para sinalizar a evolução contínua temporal ao modelo.

        Args:
            features (dict): Dicionário de features originais (ignorado, substituído pelo tempo interno).
            targets (dict): Os valores reais e absolutos do sistema no passo atual.
        """
        self.step_count += 1
        
        # Alvos normalizados
        y_norm = {res: targets.get(res, 0.0) / self.max_values[res] for res in self.resources}
        
        # Relógio logarítmico
        x_norm = {'time_step': np.log1p(self.step_count) / 10.0}
        
        self.model.learn_one(x=x_norm, y=y_norm)

    def predict_one(self, features: dict) -> dict:
        """
        Gera a previsão para o próximo passo temporal.

        Args:
            features (dict): Features atuais do sistema (ignoradas a favor do relógio interno).

        Returns:
            dict: Os valores previstos desnormalizados e absolutos, com piso garantido em 0.0.
        """
        x_norm = {'time_step': np.log1p(self.step_count + 1) / 10.0}
        
        pred_norm = self.model.predict_one(x=x_norm)
        
        # Desnormaliza
        return {k: max(0.0, v * self.max_values[k]) for k, v in pred_norm.items()}

    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 1000):
        """
        Simula ativamente o futuro gerando previsões recursivas até atingir o limite estipulado.

        Clona o estado atual do modelo (deep copy) e entra num loop onde as previsões 
        geradas são realimentadas no próprio clone como se fossem a realidade. O loop 
        para quando a previsão de qualquer recurso cruza a linha vermelha (threshold) 
        ou o horizonte máximo é alcançado.

        Args:
            current_features (dict): Estado base do sistema para iniciar a simulação.
            thresholds (dict): Limites críticos para cada recurso (ex: {'Mem': 14000000}).
            max_horizon (int, opcional): Limite máximo de passos futuros a serem simulados.

        Returns:
            tuple:
                - steps_to_failure (int): O número de passos no futuro até a provável falha 
                                          (-1 se o sistema não falhar dentro do max_horizon).
                - predictions_path (list[dict]): A trajetória completa das previsões a cada passo.
        """
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