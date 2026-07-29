import os
import time
import pandas as pd
import psutil

from src.strategies.execution_strategy import ExecutionStrategy
from src.utils import DataAggregator, calculate_metrics, generate_individual_plots, save_metrics_to_txt
from src.data_loader import load_system_metrics

class OnlineLearningStrategy(ExecutionStrategy):
    """
    Estratégia de execução para modelos preditivos de aprendizado online (incremental).

    Esta classe implementa o ciclo contínuo de coleta de dados, treinamento e predição
    para lidar com o envelhecimento de software (Software Aging). 
    
    A execução suporta dois modos de operação:
    1. Modo Replay: Consome dados de logs históricos (CSV) para simular a passagem do tempo.
    2. Modo Live: Monitora processos do sistema operacional em tempo real utilizando um DataAggregator.

    O ciclo de vida da estratégia é dividido em três fases principais:
    - Aprendizado (Warmup): O modelo consome dados reais passo a passo até atingir o `split_step`.
    - Predição (No escuro): Ao atingir o `split_step`, o modelo projeta o comportamento futuro
      dos recursos durante o `horizonte_de_previsao` estipulado.
    - Observação e Validação: Continua coletando dados reais para comparar com a projeção feita,
      gerando relatórios de métricas de erro (MAD, MSD, MAPE) e gráficos comparativos ao final.
    """

    def execute(self, context):
        print(f"\nIniciando Aprendizado Online com {context.model_name}...")
        print("\n" + "="*40)
        print(f"DEBUG DE LIMITES (O que o Python enxerga):")
        print(f"Dicionario Completo: {context.thresholds_by_resource}")
        print("="*40 + "\n")

        is_replay_mode = os.path.isdir(context.directory_path) and os.path.exists(os.path.join(context.directory_path, "cpu.csv"))
    
        # Variáveis de Controle para o Aprendizado Online
        data_stream = None
        aggregator = None

        if is_replay_mode:
            print(f" Modo leitura: Lendo logs historicos de {context.directory_path}")
            # Carrega dados já agregados (ex: Média Horária)
            full_df = load_system_metrics(context.directory_path)
            data_stream = full_df.iterrows()
            print(f"Dados carregados: {len(full_df)} amostras.")
        else:
            print("Modo live: Iniciando monitoramento em tempo real...")
            context.monitor_process.start()
            timeout = 10
            start_time = time.time()
            while not os.path.exists(context.filename) and (time.time() - start_time) < timeout:
                time.sleep(0.5)
                
            AGGREGATION_WINDOW = 5
            aggregator = DataAggregator(context.resources, AGGREGATION_WINDOW)

        last_observation = None
        learning_step = 0
        warmup_steps = 0
        
        timestamps = []
        history_real = {res: [] for res in context.resources}
        history_pred = {res: [] for res in context.resources}

        aux_metrics = ['DiskIO', 'Frag_1', 'IOWait'] 
        history_aux = {res: [] for res in aux_metrics}

        running = True

        print(f"Monitoramento iniciado. Aquecendo modelo por {warmup_steps} segundos...")

        SPLIT_STEP = context.split_step
        HORIZONTE_DE_PREVISAO = context.horizonte_de_previsao
        todas_metricas_erro = {}

        try: 
            while running:
                features_mean = None

                if is_replay_mode:
                    try:
                        timestamp, row = next(data_stream)
                        # Mapeia colunas do Loader (ex: 'mem_used_mean') para o Modelo (ex: 'Mem')
                        features_mean = {}
                        if 'Mem' in context.resources: features_mean['Mem'] = row.get('mem_used_mean', 0)
                        if 'CPU' in context.resources: features_mean['CPU'] = row.get('cpu_total_mean', 0)
                        if 'Swap' in context.resources: features_mean['Swap'] = row.get('swap_used_mean', 0)
                        if 'DiskSpace' in context.resources: features_mean['DiskSpace'] = row.get('disk_space_used_mean', 0)
                        
                        # Adiciona exógenas extras se disponíveis (Fragmentação)
                        # Nota: Se o modelo não usar, ele ignora, ou você ajusta self.resources

                        features_mean['Frag_1'] = row['frag_order_1_intensity_mean']
                        features_mean['DiskIO'] = row.get('disk_tps_mean', 0)
                        features_mean['IOWait'] = row.get('iowait_mean', 0)
                    except StopIteration:
                        print("\nFim dos dados históricos.")
                        running = False
                        break
                else:
                    time.sleep(context.monitoring_interval_in_seconds)
                    # Ler dado mais recente
                    try:
                        df = pd.read_csv(context.filename)
                        if df.empty: continue
                        current_row = df.iloc[-1]
                        raw_features = {res: current_row[res] for res in context.resources}
                    except Exception:
                        continue

                    aggregator.add_data(raw_features)
                    if aggregator.is_ready():
                        features_mean = aggregator.get_aggregated_data()

                # Se por algum motivo não tiver features, pula
                if features_mean is None: continue

                timestamps.append(len(timestamps)+1)
                for res in context.resources:
                    history_real[res].append(features_mean[res])

                # MONITORAMENTO E APRENDIZADO SILENCIOSO
                if learning_step < SPLIT_STEP:
                    if last_observation is not None:
                        context.forecasting.model.learn_one(last_observation, features_mean)
                    last_observation = features_mean.copy()
                    learning_step += 1
                    print(f"\rMonitorando e Aprendendo... {learning_step}/{SPLIT_STEP}", end="")
                    continue
                
                # MOMENTO DA PREVISÃO NO ESCURO
                elif learning_step == SPLIT_STEP:
                    print(f"\n\n[!] Passo {SPLIT_STEP} atingido! Parando aprendizado e prevendo futuro...")
                    
                    # se estiver em modo replay, ignora os limites dos recursos para evitar que o modelo pare de prever
                    if is_replay_mode:
                        if context.model_name in context.online_models:
                            threshold_arg = {res: float('inf') for res in context.resources}
                        else:
                            threshold_arg = float('inf')
                        print(" -> Modo Replay: Limites originais ignorados para desenhar TODO o horizonte.")
                    else:
                        threshold_arg = context.thresholds_by_resource if context.model_name in context.online_models else context.thresholds_by_resource.get('Mem', float('inf'))
                    
                    # Gera o forecast completo para o futuro
                    steps_to_fail, path = context.forecasting.model.predict_until_failure(
                        last_observation, 
                        threshold_arg,
                        max_horizon=HORIZONTE_DE_PREVISAO 
                    )
                    
                    # Salva o caminho previsto no history_pred inteiro de uma vez
                    if isinstance(path, dict):# Para VARMA
                        for res in context.resources:
                            history_pred[res] = path.get(res, [])
                    elif isinstance(path, list): # Para River
                        for res in context.resources:
                            history_pred[res] = [step.get(res, 0) for step in path]
                            
                    learning_step += 1
                    
                    # Se não for modo replay, parar de monitorar depois da previsão
                    if not is_replay_mode:
                        running = False

                # APENAS OBSERVAÇÃO DA REALIDADE
                else:
                    learning_step += 1
                    print(f"\rObservando realidade para comparação... {learning_step}", end="")
                    
                    # Se já observou realidade suficiente para cobrir todo o horizonte previsto, encerra.
                    if learning_step >= SPLIT_STEP + HORIZONTE_DE_PREVISAO:
                        print("\nFim do horizonte de previsão atingido!")
                        running = False

        except KeyboardInterrupt:
            print("\nMonitoramento interrompido pelo usuario.")

        finally: 
            if not is_replay_mode:
                context.monitor_process.terminate()

            print("\n" + "="*50)
            print(f"RELATORIO DE PRECISAO DO HORIZONTE DE PREVISAO: {context.model_name.upper()}")
            print("="*50)
            
            # Calcula as métricas apenas para o período de previsão (comparando o futuro real com o previsto)
            for res in context.resources:
                reais_futuro = history_real[res][SPLIT_STEP : SPLIT_STEP + HORIZONTE_DE_PREVISAO]
                preds_futuro = history_pred[res][:len(reais_futuro)]
                
                if len(reais_futuro) > 0 and len(preds_futuro) > 0:
                    metricas = calculate_metrics(reais_futuro, preds_futuro)
                    todas_metricas_erro[res] = metricas 
                    
                    print(f"Recurso: {res}")
                    print(f"  - MAD  (Erro Absoluto): {metricas['MAD']}")
                    print(f"  - MSD  (Erro Quadratico): {metricas['MSD']}")
                    print(f"  - MAPE (Erro Percentual): {metricas['MAPE']}%")
                    print("-" * 50)
            
            if context.output_directory:
                    base_dir = context.output_directory
            else:
                base_dir = context.directory_path if is_replay_mode else os.path.dirname(context.filename)
            
            # chamada do utils
            save_metrics_to_txt(
                model_name=context.model_name,
                split_step=SPLIT_STEP,
                horizon=HORIZONTE_DE_PREVISAO,
                metrics_dict=todas_metricas_erro,
                base_dir=base_dir
            )

            # Gerar e Salvar o Gráfico
            if context.save_plot and len(timestamps) > 0:
                # Define qual diretório base enviar para o utilitário
                if context.output_directory:
                    base_dir = context.output_directory
                else:
                    base_dir = context.directory_path if is_replay_mode else context.filename
                
                # Chama a função externa para fazer o trabalho visual
                generate_individual_plots(
                    resources=context.resources,
                    timestamps=timestamps,
                    history_real=history_real,
                    history_pred=history_pred,
                    model_name=context.model_name,
                    base_path=base_dir,
                    is_replay_mode=is_replay_mode,
                    split_step=SPLIT_STEP,      
                    metricas_erro=todas_metricas_erro 
                )