import numpy as np
import copy
from river import time_series
from river import multioutput
from river import base
from river import optim
from src.models.online_model import OnlineModel
from river import linear_model 

class SNARIMAX_Adapter(base.Regressor):
    """
    Adaptador (Wrapper) para inicialização do SNARIMAX na biblioteca River.

    Além de converter a interface de série temporal (y, x) para o padrão de 
    regressão comum (x, y) exigido pelo `RegressorChain`, esse adaptador injeta 
    um otimizador SGD (Stochastic Gradient Descent) com uma taxa de aprendizado 
    suave (0.01) no modelo linear base. Isso garante maior estabilidade na 
    convergência dos pesos ao longo do tempo.
    """

    # p=4 (olha 2h para trás em passos de 30min), d=0 (estabiliza), q=1 (corrige erro recente)
    def __init__(self, p=1, d=1, q=1):
        """
        Args:
            p (int, opcional): Ordem da parte Autorregressiva (AR). Quantos passos no passado observar.
            d (int, opcional): Grau de Diferenciação (I). Quantas vezes diferenciar para estabilizar a série.
            q (int, opcional): Ordem da Média Móvel (MA). Tamanho da janela para correção de erros recentes.
        """
        self.p = p
        self.d = d
        self.q = q
        
        # Define o otimizador com a taxa suave de aprendizagem
        optimizer = optim.SGD(0.01)
        
        # Cria o "motor matemático" (Regressor) usando o otimizador 
        regressor = linear_model.LinearRegression(optimizer=optimizer)
        
        # Entrega o motor configurado para a classe SNARIMAX
        self.model = time_series.SNARIMAX(
            p=self.p, 
            d=self.d, 
            q=self.q, 
            regressor=regressor
        )
        
    def learn_one(self, x, y):
        """Ajusta a ordem dos parâmetros (y, x) exigida pelo SNARIMAX nativo."""
        self.model.learn_one(y=y, x=x)
        return self
        
    def predict_one(self, x):
        """
        Converte a chamada de predição unitária para o método `forecast` do SNARIMAX.
        Retorna 0.0 em caso de instabilidade matemática interna.
        """
        try:
            res = self.model.forecast(horizon=1, xs=[x] if x else None)
            return res[0]
        except Exception:
            return 0.0

class ARIMAX(OnlineModel):
    """
    Modelo de aprendizado online baseado em ARIMAX Multivariado (Regressor Chain).

    Utiliza uma cadeia de regressão para capturar a 
    causalidade entre diferentes métricas físicas do sistema (ex: a previsão 
    da Memória ajuda a prever o Swap) e emprega uma escala de tempo logarítmica 
    para estabilizar o treinamento em execuções de longa duração.
    """

    def __init__(self, normalization_params: dict[str, tuple[float, float]] = None, path_to_save_weights: str | None = None):
        self.normalization_params = normalization_params if normalization_params else {}
        self.path_to_save_weights = path_to_save_weights
        
        # relógio contínuo para modelar o tempo (crucial para o Software Aging)
        # Ele é incrementado a cada passo de aprendizado, mas não é resetado, 
        # permitindo que o modelo capture a evolução temporal real do sistema.
        self.step_count = 0
        
        # Define a ordem causal
        base_order = ['CPU', 'Mem', 'Swap', 'DiskSpace']
        if normalization_params:
            self.resources = [r for r in base_order if r in normalization_params.keys()]
            for r in normalization_params.keys():
                if r not in self.resources:
                    self.resources.append(r)
        else:
            self.resources = base_order

        # Define os valores máximos para normalização (teto fixo para evitar saturação e manter a escala consistente)
        self.max_values = {
            'CPU': 100.0,               
            'Mem': 16000000.0,          
            'Swap': 8000000.0,          
            'DiskSpace': 500000000.0   
        }
        # Instancia o motor e a cadeia
        base_model = SNARIMAX_Adapter(p=12, d=1, q=1)
        self.model = multioutput.RegressorChain(
            model=base_model,
            order=self.resources
        )

    def learn_one(self, features: dict, targets: dict):
        """
        Processa uma única amostra e atualiza os pesos do modelo.

        Aplica normalização estática (dividindo pelo teto físico) e converte o 
        contador de passos em uma variável temporal logarítmica, permitindo que 
        o modelo compreenda a passagem do tempo sem sofrer saturação numérica.
        """
        self.step_count += 1
        
        # Normaliza usando o teto fixo (os valores vão variar naturalmente entre 0.0 e 1.0)
        y_norm = {res: features.get(res, 0.0) / self.max_values[res] for res in self.resources}
        
        #escalando o tempo usando logaritmo para desacelerar o crescimento e evitar saturação rápida
        x_norm = {'time_step': np.log1p(self.step_count) / 10.0}
        
        # Treina
        self.model.learn_one(x=x_norm, y=y_norm)

    def predict_one(self, features: dict) -> dict:
        """Prevê e expande o valor de volta à escala real."""
        x_norm = {'time_step': np.log1p(self.step_count + 1) / 10.0}
        pred_norm = self.model.predict_one(x=x_norm)
        
        # Desnormaliza
        pred_real = {k: max(0.0, v * self.max_values[k]) for k, v in pred_norm.items()}
        return pred_real

    def predict_until_failure(self, current_features: dict, thresholds: dict, max_horizon: int = 1000):
        """Avança no escuro até que uma variável atinja o threshold real."""
        steps_to_failure = -1
        predictions_path = []
        
        sim_model = copy.deepcopy(self.model)
        future_step = self.step_count
        
        for i in range(max_horizon):
            # O relógio virtual continua a contar no futuro 
            future_step += 1
            current_x_norm = {'time_step': np.log1p(future_step) / 10.0}
            
            # Prevê usando o modelo normalizado
            pred_norm = sim_model.predict_one(x=current_x_norm)
            
            # Desnormaliza para verificar os Limites de Falha reais
            pred_real = {k: max(0.0, v * self.max_values[k]) for k, v in pred_norm.items()}
            predictions_path.append(pred_real)
            
            # Verifica se alguma variável ultrapassou o threshold (A Falha)
            failed = False
            for res in self.resources:
                if pred_real.get(res, 0.0) >= thresholds.get(res, float('inf')):
                    if steps_to_failure == -1:
                        steps_to_failure = i + 1
                    failed = True
                    break
            
            if failed:
                break
                
            # O Clone aprende da sua própria previsão para manter a inércia correta
            sim_model.learn_one(x=current_x_norm, y=pred_norm)
            
        return steps_to_failure, predictions_path