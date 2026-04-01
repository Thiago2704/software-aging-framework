import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt


def normalize(sequence: pd.DataFrame | np.ndarray) -> tuple[pd.DataFrame, float, float]:
    # Normalize data
    s_min = min(sequence)
    s_max = max(sequence)
    sequence = (sequence - s_min) / (s_max - s_min)
    sequence = sequence.replace(np.nan, 0)

    return sequence, s_min, s_max


def denormalize(
    sequence: pd.DataFrame | np.ndarray, s_min: float, s_max: float
) -> pd.DataFrame:
    sequence = sequence * (s_max - s_min) + s_min

    return sequence


def split_sets(
    sequence: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(sequence) * train_ratio)
    train, test = sequence[:train_size], sequence[train_size:]

    return train, test


def split_sequence(sequence, n_steps):
    x, y = [], []
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_idx = i + n_steps
        # Check if we are beyond the sequence
        if end_idx > len(sequence) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x), np.array(y)


def split_multivariate_sequences(sequences, n_steps):
    x, y = list(), list()
    for i in range(len(sequences)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

def generate_individual_plots(resources: list, timestamps: list, history_real: dict, history_pred: dict, model_name: str, base_path: str, is_replay_mode: bool):
    """
    Gera e salva gráficos individuais em estilo Dashboard Dark Mode para cada recurso monitorizado.
    """
    print("\nGerando gráficos de execução online (Estilo Dashboard)...")
    
    largura = 6
    altura = 4
    cor_real = "#1125bc"
    cor_pred = "#c80707"

    # Descubra quantos minutos vale cada passo (se mudou o resample para 1min, coloque 1. Se for 30min, coloque 30)
    minutos_por_passo = 30 
    
    # Cria uma nova lista convertendo os passos (timestamps) para horas
    tempo_em_horas = [(t * minutos_por_passo) / 60 for t in timestamps]
    
    for res in resources:
        fig, ax = plt.subplots(figsize=(largura, altura))
        #cor do fundo

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        resources_names ={
            'CPU': "CPU",
            'Mem': "Memory",
            'Swap': "Swap",
            'DiskSpace': "Disk Space",
        }

        texto_y = resources_names.get(res, res)
        fator_conversao = 1024 if res in ['Mem', 'Swap', 'DiskSpace'] else 1

        # Cria listas temporárias com os valores convertidos apenas para desenhar o gráfico
        y_real_plot = [v / fator_conversao for v in history_real[res]]
        y_pred_plot = [v / fator_conversao for v in history_pred[res]]
        
        # Plot das linhas 
        ax.plot(tempo_em_horas, y_real_plot, label=f'Real {texto_y}', color=cor_real, linewidth=1.8)
        ax.plot(tempo_em_horas, y_pred_plot, label=f'Predicted {texto_y}', color=cor_pred, linestyle='--', linewidth=1.8)

        # Zoom Dinâmico
        valores_reais = [v for v in y_real_plot if v > 0]
        valores_pred = [v for v in y_pred_plot if v > 0]
        todos_valores = valores_reais + valores_pred
        
        if todos_valores:
            min_y, max_y = min(todos_valores), max(todos_valores)
            amplitude = max_y - min_y
            margem = amplitude * 0.1 if amplitude > 0 else min_y * 0.05
            ax.set_ylim(bottom=max(0, min_y - margem), top=max_y + margem)
        
        # Estilização
        ax.tick_params(colors="#000000", labelsize=10)

        # Legenda do Eixo X (Tempo)
        ax.set_xlabel("Time (Hours)", color="#000000", fontsize=11, fontweight='bold', labelpad=10)
        
        # Legenda do Eixo Y (Recurso dinâmico)
        #ax.set_ylabel(f"Consumo de {res}", color="#000000", fontsize=11, fontweight='bold', labelpad=10)

        # Dicionário de Legendas
        legendas_y = {
            'CPU': "CPU utilization (%)",
            'Mem': "Memory Usage (MB)",
            'Swap': "Swap Usage (MB)",
            'DiskSpace': "Disk Space Used (MB)",
        }
                
        texto_y = legendas_y.get(res, f"Consumo de {res}")
        # Aplica a legenda
        ax.set_ylabel(texto_y, color="#000000", fontsize=11, fontweight='bold', labelpad=10)
        
        for spine in ax.spines.values():
            spine.set_color('#333333')
   
        #ax.grid(True, color='#333333', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.legend(facecolor='white', edgecolor='#cccccc', labelcolor='black', loc='upper left')
    
        plt.tight_layout()
        
        # Resolução do caminho de salvamento
        if is_replay_mode:
            path_to_save = os.path.join(base_path, f"replay_analysis_graph_{res}.png")
        else: 
            path_to_save = base_path.replace(".csv", f"_{res}.png")
            
        plt.savefig(path_to_save, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Gráfico salvo em: {path_to_save}")
        
        plt.close(fig)


def calculate_metrics(real_values: list, pred_values: list) -> dict:
    """Calcula MAD, MSD e MAPE para duas listas de valores."""
    y_true = np.array(real_values)
    y_pred = np.array(pred_values)
    
    # MAD (Mean Absolute Deviation)
    mad = np.mean(np.abs(y_true - y_pred))
    
    # MSD (Mean Squared Deviation)
    msd = np.mean(np.square(y_true - y_pred))
    
    # MAPE (Mean Absolute Percentage Error)
    # Adiciona um epsilon super pequeno para evitar divisão por zero
    epsilon = np.finfo(np.float64).eps
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, epsilon))) * 100
    
    return {
        "MAD": round(mad, 4),
        "MSD": round(msd, 4),
        "MAPE": round(mape, 2) # Retorna em porcentagem (ex: 5.43%)
    }

class DataAggregator:
    def __init__(self, resources: list[str], window_size: int):
        self.resources = resources
        self.window_size = window_size
        self._buffer = {res: [] for res in self.resources}

    def add_data(self, raw_data: dict):
        """Adiciona um snapshot (dado bruto) ao buffer."""
        for res in self.resources:
            if res in raw_data:
                self._buffer[res].append(raw_data[res])

    def is_ready(self) -> bool:
        """Verifica se o buffer atingiu o tamanho da janela."""
        # Basta checar um dos recursos, pois todos crescem juntos
        return len(self._buffer[self.resources[0]]) >= self.window_size

    def get_aggregated_data(self) -> dict:
        """Calcula a MÉDIA, retorna o dado limpo e ESVAZIA o buffer."""
        aggregated = {}
        for res in self.resources:
            # Calcula a média 
            if self._buffer[res]:
                aggregated[res] = np.mean(self._buffer[res])
            else:
                aggregated[res] = 0.0 # Fallback caso vazio
            
            # Limpar o buffer automaticamente aqui
            self._buffer[res] = []
            
        return aggregated