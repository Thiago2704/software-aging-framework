import numpy as np
import pandas as pd


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