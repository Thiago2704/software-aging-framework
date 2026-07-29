import pandas as pd
import os

def load_system_metrics(folder_path: str, resample_rule='30min'):
    """
    Lê, processa e unifica arquivos CSV de métricas de sistema de um diretório.

    A função procura por logs padronizados de recursos (CPU, Memória, Disco) e 
    logs de fragmentação de memória, converte os timestamps para um índice 
    temporal unificado (DatetimeIndex) e combina todos os dados num único DataFrame. 
    Valores ausentes são preenchidos por propagação (forward-fill) e zeros.

    Se uma regra de reamostragem (resample) for fornecida, os dados serão agrupados 
    no intervalo especificado, calculando a média ('mean') e o valor máximo ('max') 
    para cada métrica.

    Args:
        folder_path (str): Caminho para a pasta contendo os arquivos CSV de log 
                           (ex: 'cpu.csv', 'memory.csv', 'fragmentation_0.csv').
        resample_rule (str, opcional): Regra de frequência temporal suportada pelo 
                                       Pandas (ex: '30min', '1H'). O padrão é '30min'.
                                       Se passado como None, a agregação temporal é ignorada.

    Returns:
        pd.DataFrame: DataFrame unificado e indexado cronologicamente contendo todas as 
                      métricas processadas. Se houve reamostragem, as colunas terão 
                      sufixos '_mean' e '_max' (ex: 'cpu_total_mean', 'mem_used_max').

    Raises:
        ValueError: Se nenhum arquivo CSV válido for encontrado e processado com sucesso 
                    dentro da pasta especificada.
    """

    print(f"Lendo logs de: {folder_path}")
    dfs = []
    
    # Lista de arquivos 
    files_std = [
        ("cpu.csv", 'cpu'),
        ("memory.csv", 'mem'),
        ("disk_write_read.csv", 'disk_io'),
        ("disk.csv", 'disk_space')
    ]

    for filename, type_tag in files_std:
        path = os.path.join(folder_path, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=';', index_col=False)
                df.columns = [c.strip() for c in df.columns]
                
                # Converte data manualmente (Formato ISO)
                df['date_time'] = pd.to_datetime(df['date_time'], format='%Y-%m-%d %H:%M:%S')
                df = df.set_index('date_time')

                if type_tag == 'cpu':
                    df['cpu_total'] = df['usr'] + df['sys']
                    dfs.append(df[['cpu_total', 'iowait']])
                
                elif type_tag == 'mem':
                    df = df.rename(columns={'used': 'mem_used', 'swap': 'swap_used'})
                    dfs.append(df[['mem_used', 'swap_used']])
                
                elif type_tag == 'disk_io':
                    df = df.rename(columns={'tps': 'disk_tps'})
                    dfs.append(df[['disk_tps']])
                
                elif type_tag == 'disk_space':
                    df = df.rename(columns={'used': 'disk_space_used'})
                    dfs.append(df[['disk_space_used']])
            except Exception as e:
                print(f"Erro ao ler {filename}: {e}")

    # lida com arquivo de fragmentação
    for order in ['0', '1']:
        fname = f"fragmentation_{order}.csv"
        path = os.path.join(folder_path, fname)
        
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=';', index_col=False)
                df.columns = [c.strip() for c in df.columns]
                
                # Converte data Unix 
                df['datetime'] = pd.to_datetime(df['datetime'], format='%a %b %d %H:%M:%S %Y')
                
                # Agrupa e SOMA ocorrências
                df_grouped = df.groupby('datetime')['process_occurrences'].sum()
                
                df_final = pd.DataFrame(df_grouped)
                df_final.columns = [f'frag_order_{order}_intensity']
                
                dfs.append(df_final)
            except Exception as e:
                 print(f"Erro ao ler {fname}: {e}")


    if not dfs:
        raise ValueError("Nenhum arquivo CSV válido encontrado na pasta!")

    # Junta tudo
    full_df = pd.concat(dfs, axis=1)
    full_df = full_df.sort_index()
    
    # Preenche buracos 
    full_df = full_df.ffill().fillna(0)

    # Agregação Temporal
    if resample_rule:
        rule = resample_rule.lower().replace('h', 'h') 
        full_df = full_df.resample(rule).agg(['mean', 'max'])
        
        # Achata colunas
        full_df.columns = ['_'.join(col).strip() for col in full_df.columns.values]
        full_df = full_df.dropna()

    print(f"Leitura concluída! Shape final: {full_df.shape}")
    
    return full_df