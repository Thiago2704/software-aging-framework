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

def generate_individual_plots(resources: list, timestamps: list, history_real: dict, history_pred: dict, model_name: str, base_path: str, is_replay_mode: bool, split_step: int, metricas_erro: dict):
    """
    Gera gráficos com linha de corte (split) e banda de confiança (margem de erro).
    """
    print("\nGerando gráficos de Previsão de Longo Prazo (Horizonte Escuro)...")
    
    largura = 8
    altura = 5
    cor_real = "#1125bc"
    cor_pred = "#c80707"

    minutos_por_passo = 10
    tempo_em_horas = [(t * minutos_por_passo) / 60 for t in timestamps]
    #tempo em minutos
    #tempo_em_horas = [t * minutos_por_passo for t in timestamps]

    # Momento exato onde a previsão começou (Linha Vertical)
    tempo_split = tempo_em_horas[split_step - 1] if split_step - 1 < len(tempo_em_horas) else tempo_em_horas[-1]
    
    for res in resources:
        fig, ax = plt.subplots(figsize=(largura, altura))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        fator_conversao = 1024 if res in ['Mem', 'Swap', 'DiskSpace'] else 1 

        # Quantos passos a linha vermelha (previsão) conseguiu dar
        passos_previstos = len(history_pred.get(res, []))
        
        # Descobre onde a linha azul deve parar
        if passos_previstos > 0:
            # Para no exato momento: Início da previsão + Passos que sobreviveu
            limite_final = (split_step -1) + passos_previstos
        else:
            # Se não houver previsão, desenha tudo
            limite_final = len(history_real[res])
            
        # Faz o corte (slice) nas listas originais para o tamanho exato
        y_real_plot = [v / fator_conversao for v in history_real[res][:limite_final]]
        tempo_real_plot = tempo_em_horas[:limite_final]
        
        # Plot da linha REAL contínua 
        ax.plot(tempo_real_plot, y_real_plot, label=f'Real {res}', color=cor_real, linewidth=1.8)
        
        # Prepara a linha de PREVISÃO (começa apenas a partir do split_step)
        if len(history_pred[res]) > 0:
            
            tempo_pred = tempo_em_horas[split_step - 1 : split_step - 1 + len(history_pred[res])]
            y_pred_plot = [v / fator_conversao for v in history_pred[res]]

            # INTERVALOS DE CONFIANÇA 80% E 95%
            if res in metricas_erro:
                mad = metricas_erro[res]['MAD'] / fator_conversao
                
                y_upper_95, y_lower_95 = [], []
                y_upper_80, y_lower_80 = [], []
                
                num_passos = len(y_pred_plot)
                
                for step, p in enumerate(y_pred_plot):
                    # a incerteza cresce com a raiz quadrada do tempo
                    fator_incerteza = (step / (num_passos - 1)) ** 0.5 if num_passos > 1 else 0
                    
                    # Multiplicadores para simular Intervalos de Confiança baseados no MAD
                    # (Aproximação baseada na Distribuição Normal: MAD * 1.25 = Desvio Padrão)
                    # 95% = 1.96 * Desvio Padrão  -> aprox. 2.45 * MAD
                    # 80% = 1.28 * Desvio Padrão  -> aprox. 1.60 * MAD
                    
                    erro_80 = mad * 1.60 * fator_incerteza
                    erro_95 = mad * 2.45 * fator_incerteza
                    
                    # Limites 95% (Cinza Claro - Mais largo)
                    y_upper_95.append(p + erro_95)
                    y_lower_95.append(max(0, p - erro_95))
                    
                    # Limites 80% (Cinza Escuro - Mais estreito)
                    y_upper_80.append(p + erro_80)
                    y_lower_80.append(max(0, p - erro_80))
                
                # Cores do Intervalo de Confiança
                cor_ci_95 = "#E0E0E0" # Cinza claro
                cor_ci_80 = "#BDBDBD" # Cinza mais escuro
                
                # DESENHAR PRIMEIRO AS SOMBRAS (Para ficarem no fundo)
                # Desenha a sombra mais larga (95%)
                ax.fill_between(tempo_pred, y_lower_95, y_upper_95, color=cor_ci_95, label='95% Confidence Interval')
                
                # Desenha a sombra mais estreita (80%) por cima da clara
                ax.fill_between(tempo_pred, y_lower_80, y_upper_80, color=cor_ci_80, label='80% Confidence Interval')

            # DESENHAR A LINHA VERMELHA POR ÚLTIMO (Para ficar destacada por cima dos cinzas)
            ax.plot(tempo_pred, y_pred_plot, label=f'Predicted {res}', color=cor_pred, linewidth=2.0)

        # --- LINHA VERTICAL PONTILHADA ---
        ax.axvline(x=tempo_split, color='gray', linestyle=':', linewidth=2, label='Forecast Start')

        # Zoom Dinâmico e Estilização
        ax.tick_params(colors="#000000", labelsize=10)
        ax.set_xlabel("Time (hours)", color="#000000", fontsize=11, fontweight='bold', labelpad=10)
        
        legendas_y = {
            'CPU': "CPU utilization (%)",
            'Mem': "Memory Usage (MB)",
            'Swap': "Swap Usage (MB)",
            'DiskSpace': "Disk Space Used (MB)",
        }
        texto_y = legendas_y.get(res, f"Consumo de {res}")
        ax.set_ylabel(texto_y, color="#000000", fontsize=11, fontweight='bold', labelpad=10)
        
        for spine in ax.spines.values():
            spine.set_color('#333333')
   
        ax.legend(facecolor='white', edgecolor='#cccccc', labelcolor='black', loc='upper left')
        plt.tight_layout()
        
        if is_replay_mode:
            path_to_save = os.path.join(base_path, f"horizon_graph_{res}.png")
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

def save_metrics_to_txt(model_name: str, split_step: int, horizon: int, metrics_dict: dict, base_dir: str):
    """
    Salva as métricas de erro num ficheiro .txt formatado.
    """
    if not metrics_dict:
        return
        
    caminho_txt = os.path.join(base_dir, f"metricas_erro_{model_name}.txt")
    
    with open(caminho_txt, "w", encoding="utf-8") as f_out:
        f_out.write("="*50 + "\n")
        f_out.write(f"RELATORIO DE PRECISAO - MODELO: {model_name.upper()}\n")
        f_out.write(f"Split Step: {split_step} | Horizonte: {horizon}\n")
        f_out.write("="*50 + "\n\n")
        
        for res, metricas in metrics_dict.items():
            f_out.write(f"Recurso: {res}\n")
            f_out.write(f"  - MAD  (Erro Absoluto): {metricas['MAD']}\n")
            f_out.write(f"  - MSD  (Erro Quadratico): {metricas['MSD']}\n")
            f_out.write(f"  - MAPE (Erro Percentual): {metricas['MAPE']}%\n")
            f_out.write("-" * 50 + "\n")
            
    print(f"\n[+] Arquivo de métricas salvo com sucesso em: {caminho_txt}")

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