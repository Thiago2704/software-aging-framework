import subprocess
import sys
import time
from multiprocessing import Queue

import pandas as pd
import psutil
import yaml
import os
from matplotlib import pyplot as plt

from src.forecasting import Forecasting
from src.monitor import ResourceMonitorProcess
from src.utils import DataAggregator, normalize, denormalize, calculate_metrics, generate_individual_plots, save_metrics_to_txt
from src.data_loader import load_system_metrics


class Framework:
    def __init__(
        self,
        run_monitoring: bool,
        resources_to_predict: list[str],
        monitoring_time_in_seconds: int,
        monitoring_interval_in_seconds: int,
        directory_path: str,
        model: str,
        path_to_load_weights: str | None,
        path_to_save_weights: str | None,
        save_plot: bool,
        run_in_real_time: bool,
        process_name: str,
        memory_threshold: float,
        cpu_threshold: float,
        disk_threshold: float,
        number_of_predictions: int,
        start_command: str,
        restart_command: str | None,
        normalization_log_path: str,

        # NOVAS VARIÁVEIS PARA AUTOMAÇÃO
        split_step: int = 300,
        horizonte_de_previsao: int = 96,
        output_directory: str = None
    ):
        self.run_monitoring = run_monitoring
        self.resources = resources_to_predict
        self.monitoring_time_in_seconds = monitoring_time_in_seconds
        self.monitoring_interval_in_seconds = monitoring_interval_in_seconds
        self.directory_path = directory_path
        self.model_name = model
        self.path_to_load_weights = path_to_load_weights
        self.path_to_save_weights = path_to_save_weights
        self.save_plot = save_plot
        self.run_in_real_time = run_in_real_time
        self.process_name = process_name
        self.thresholds_by_resource = {
            "Mem": memory_threshold,
            "CPU": cpu_threshold,
            "Disk": disk_threshold,
            "DiskSpace": disk_threshold, 
            "Swap": 8500000, # valor arbitrário alto para Swap, trocar no futuro
        }
        self.number_of_predictions = number_of_predictions
        self.start_command = start_command
        self.restart_command = restart_command
        self.forecasting: Forecasting | None = None
        self.monitor_process: ResourceMonitorProcess | None = None
        self.error_queue = Queue()
        self.normalization_log_path = normalization_log_path
        self.online_models = ["arf", "hat_perceptron", "isoup", "arimax", "sarimax", "varma",
                              "snarimax_ht","snarimax_hat",
                              "snarimax_oxt", "snarimax_arf", "snarimax_amf"] # Lista de modelos que usam aprendizado online
        
        # variaveis auxiliares para automação com script
        self.split_step = split_step
        self.horizonte_de_previsao = horizonte_de_previsao
        self.output_directory = output_directory


        if self.model_name in self.online_models:
            # Cria nome de arquivo novo para o log
            self.filename = self.__create_filename(self.directory_path)
            
            # Inicializa o Monitor
            self.monitor_process = ResourceMonitorProcess(
                self.monitoring_interval_in_seconds,
                self.process_name,
                self.filename,
                self.error_queue,
            )
            
            # Inicializa o Forecasting imediatamente (com DataFrame vazio)
            self.forecasting = Forecasting(
                sequence=pd.DataFrame(), 
                model_name=self.model_name,
                resources=self.resources,
                path_to_save_weights=None,
                use_normalization=False, 
                path_to_load_model=None
            )
        elif self.run_in_real_time or self.run_monitoring:
            self.path_to_save_weights = self.__create_weights_filename(
                self.path_to_save_weights
            )
            self.filename = self.__create_filename(self.directory_path)
            self.monitor_process = ResourceMonitorProcess(
                self.monitoring_interval_in_seconds,
                self.process_name,
                self.filename,
                self.error_queue,
            )
        else:
            self.filename = self.directory_path

    @staticmethod
    def __create_filename(directory_path: str) -> str:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        return f"{directory_path}/log_{current_time}.csv"

    @staticmethod
    def __create_weights_filename(directory_path: str | None) -> str | None:
        if directory_path:
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            return f"{directory_path}/log_{current_time}.h5"
        return None

    def run(self):
        if self.model_name in ["arf","hat_perceptron", "isoup", "arimax", "sarimax", "varma",
                               "snarimax_ht","snarimax_hat", 
                               "snarimax_oxt", "snarimax_arf", "snarimax_amf"]:
            self.__run_online_learning()
            return
        elif self.run_in_real_time:
            self.__run_real_time()
        else:
            self.__run_experiment()

    def __run_experiment(self):
        if self.run_monitoring:
            self.monitor_process.start()

            time.sleep(1)
            if self.error_queue.qsize() > 0:
                print("\nError monitoring process\n")
                return

            self.__countdown()
            self.monitor_process.terminate()

        dataframe = pd.read_csv(self.filename)

        self.forecasting = Forecasting(
            dataframe, self.model_name, self.resources, self.path_to_save_weights
        )
        self.forecasting.train()
        self.__plot_graph()

    def __run_real_time(self):
        self.monitor_process.start()
        time.sleep(1)

        if self.run_monitoring:
            if self.error_queue.qsize() > 0:
                print("\nError monitoring process\n")
                return

            self.__countdown()

            dataframe = pd.read_csv(self.filename)

            if dataframe.shape[0] < 4:
                print(
                    "\nNot enough monitoring data for forecasting, monitor for longer time\n"
                )
                return

            self.forecasting = Forecasting(
                dataframe, self.model_name, self.resources, self.path_to_save_weights
            )
            self.forecasting.train()

        elif self.path_to_load_weights:
            dataframe = pd.read_csv(self.normalization_log_path)
            self.forecasting = Forecasting(
                dataframe,
                self.model_name,
                self.resources,
                self.path_to_save_weights,
                False,
                self.path_to_load_weights,
            )
        else:
            print(
                "\nUnable to run if monitoring has not been run or model path has not been passed\n"
            )
            self.monitor_process.terminate()
            return

        # dictionary to store predictions over time (all predictions of number_of_predictions)
        predictions_over_time = {
            f"{resource}_n{i + 1}": []
            for resource in self.resources
            for i in range(self.number_of_predictions)
        }
        running = True

        while running:
            time.sleep(self.monitoring_interval_in_seconds)

            # collect real-time monitoring data
            current_data = pd.read_csv(self.filename)
            current_data = current_data[self.resources]

            # check if the current data has enough rows for forecasting
            if current_data.shape[0] < 4:
                continue

            n_steps = 2
            n_seq = 2
            normalization_params = {}

            for resource in self.resources:
                current_data[resource], s_min, s_max = normalize(current_data[resource])
                normalization_params[resource] = (s_min, s_max)

            # the last 4 rows of the current data are used for forecasting (n_steps = 4 or n_seq = 2 and n_steps = 2)
            reshaped_current_data = current_data[-4:].values.reshape(
                (1, n_seq, 1, n_steps, len(self.resources))
            )

            # perform forecasting using the trained model
            predictions = self.forecasting.predict_future(
                reshaped_current_data, self.number_of_predictions
            )

            flag_list = []

            # compare predictions with thresholds and update flag_list and plot the results
            for idx, resource in enumerate(self.resources):
                s_min, s_max = normalization_params[resource]
                denormalized_predictions = denormalize(
                    predictions[:, idx], s_min, s_max
                )

                for i, pred_value in enumerate(denormalized_predictions):
                    predictions_over_time[f"{resource}_n{i + 1}"].append(pred_value)

                    if pred_value > self.thresholds_by_resource[resource]:
                        flag_list.append(1)
                    else:
                        flag_list.append(0)

            # check if rejuvenation should be triggered
            if flag_list.count(1) > 0:
                print("\nActivated Rejuvenation\n")
                print("Flag list:", flag_list)

                for process in psutil.process_iter(attrs=["pid", "name"]):
                    if self.process_name.lower() in process.info["name"].lower():
                        self.__restart_process(
                            process, self.start_command, self.restart_command
                        )
                        running = False
                        break

        self.monitor_process.terminate()

        # save the predictions over time in a csv file
        predictions_over_time_df = pd.DataFrame(predictions_over_time)
        predictions_over_time_df.to_csv(
            self.filename.replace(".csv", "_predictions.csv"), index=False
        )

    def __restart_process(
        self, process: psutil.Process, start_command: str, restart_command: str | None
    ):
        if restart_command is not None:
            subprocess.Popen(restart_command, shell=True)
        else:
            process.terminate()  # Terminate the process
            process.wait()  # Wait for the process to exit

        # Start the process again
        subprocess.Popen(start_command, shell=True)

        self.monitor_process.terminate()

    def __print_progress_bar(self, current_second, text):
        progress_bar_size = 50
        current_progress = (current_second + 1) / self.monitoring_time_in_seconds
        sys.stdout.write(
            f"\r{text}: [{'=' * int(progress_bar_size * current_progress):{progress_bar_size}s}] "
            f"{current_second + 1}/{self.monitoring_time_in_seconds} seconds"
        )
        sys.stdout.flush()

    def __countdown(self):
        for current_second in range(self.monitoring_time_in_seconds):
            self.__print_progress_bar(current_second, "Monitoring")
            time.sleep(self.monitoring_interval_in_seconds)
        print()

    def __plot_graph(self):
        self.forecasting.plot_results()

        if self.save_plot:
            path_to_save = self.filename.replace(".csv", ".png")
            plt.savefig(path_to_save, dpi=300)

    def __run_online_learning(self):
            print(f"\nIniciando Aprendizado Online com {self.model_name}...")

            print("\n" + "="*40)
            print(f"DEBUG DE LIMITES (O que o Python enxerga):")
            print(f"Dicionario Completo: {self.thresholds_by_resource}")
            print("="*40 + "\n")

            is_replay_mode = os.path.isdir(self.directory_path) and os.path.exists(os.path.join(self.directory_path, "cpu.csv"))
        
            # Variáveis de Controle
            data_stream = None
            aggregator = None

            if is_replay_mode:
                print(f" Modo leitura: Lendo logs historicos de {self.directory_path}")
                # Carrega dados já agregados (ex: Média Horária)
                full_df = load_system_metrics(self.directory_path)
                data_stream = full_df.iterrows()
                print(f"Dados carregados: {len(full_df)} amostras.")
            
            else:
                print("Modo live: Iniciando monitoramento em tempo real...")
                self.monitor_process.start()
                # Espera o arquivo ser criado
                timeout = 10
                start_time = time.time()
                while not os.path.exists(self.filename) and (time.time() - start_time) < timeout:
                    time.sleep(0.5)
                    
                AGGREGATION_WINDOW = 5  # segundos
                aggregator = DataAggregator(self.resources, AGGREGATION_WINDOW)

            
            last_observation = None
            learning_step = 0
            warmup_steps = 0
            
            # Inicializar listas para o gráfico
            timestamps = []
            history_real = {res: [] for res in self.resources}
            history_pred = {res: [] for res in self.resources}

            aux_metrics = ['DiskIO', 'Frag_1', 'IOWait'] 
            history_aux = {res: [] for res in aux_metrics}

            running = True

            print(f"Monitoramento iniciado. Aquecendo modelo por {warmup_steps} segundos...")

            SPLIT_STEP = self.split_step # Passo onde o modelo para de aprender e começa a prever o futuro 
            HORIZONTE_DE_PREVISAO = self.horizonte_de_previsao # Quantos passos para o futuro ele vai adivinhar

            todas_metricas_erro = {} # Guardará os cálculos de erro

            try: 
                while running:
                    features_mean = None

                    if is_replay_mode:
                        try:
                            timestamp, row = next(data_stream)
                            
                            # Mapeia colunas do Loader (ex: 'mem_used_mean') para o Modelo (ex: 'Mem')
                            features_mean = {}
                            if 'Mem' in self.resources: features_mean['Mem'] = row.get('mem_used_mean', 0)
                            if 'CPU' in self.resources: features_mean['CPU'] = row.get('cpu_total_mean', 0)
                            if 'Swap' in self.resources: features_mean['Swap'] = row.get('swap_used_mean', 0)
                            if 'DiskSpace' in self.resources: features_mean['DiskSpace'] = row.get('disk_space_used_mean', 0)
                            
                            # Adiciona exógenas extras se disponíveis (Fragmentação)
                            # Nota: Se o modelo não usar, ele ignora, ou você ajusta self.resources
                            
                            features_mean['Frag_1'] = row['frag_order_1_intensity_mean']
                            features_mean['DiskIO'] = row.get('disk_tps_mean', 0)
                            features_mean['IOWait'] = row.get('iowait_mean', 0)

                            # Simulação de tempo (opcional)
                            # time.sleep(0.05) 
                            
                        except StopIteration:
                            print("\nFim dos dados históricos.")
                            running = False
                            break
                    
                    else:
                        time.sleep(self.monitoring_interval_in_seconds)
                        # Ler dado mais recente
                        try:
                            df = pd.read_csv(self.filename)
                            if df.empty: continue
                            current_row = df.iloc[-1]
                            raw_features = {res: current_row[res] for res in self.resources}
                        except Exception:
                            continue

                        aggregator.add_data(raw_features)

                        if aggregator.is_ready():

                            features_mean = aggregator.get_aggregated_data()

                    # Se por algum motivo não tiver features, pula
                    if features_mean is None: continue

                    timestamps.append(len(timestamps)+1)
                    for res in self.resources:
                        history_real[res].append(features_mean[res]) # Salva a realidade continuamente

                    # MONITORAMENTO E APRENDIZADO SILENCIOSO 
                    if learning_step < SPLIT_STEP:
                        if last_observation is not None:
                            self.forecasting.model.learn_one(last_observation, features_mean)
                        last_observation = features_mean.copy()
                        learning_step += 1
                        print(f"\rMonitorando e Aprendendo... {learning_step}/{SPLIT_STEP}", end="")
                        continue
                    
                    # MOMENTO DA PREVISÃO NO ESCURO 
                    elif learning_step == SPLIT_STEP:
                        print(f"\n\n[!] Passo {SPLIT_STEP} atingido! Parando aprendizado e prevendo futuro...")
                        
                        # se estiver em modo replay, ignora os limites dos recursos para evitar que o modelo pare de prever
                        if is_replay_mode:
                            if self.model_name in ["arf", "hat_perceptron", "arimax", "sarimax", "varma", "isoup",
                                                   "snarimax_ht","snarimax_hat",
                                                   "snarimax_oxt", "snarimax_arf", "snarimax_amf"]:
                                threshold_arg = {res: float('inf') for res in self.resources}
                            else:
                                threshold_arg = float('inf')
                            print(" -> Modo Replay: Limites originais ignorados para desenhar TODO o horizonte.")
                        else:
                            threshold_arg = self.thresholds_by_resource if self.model_name in ["arf", "hat_perceptron", "arimax", "sarimax", "varma", "isoup",
                                                                                               "snarimax_ht","snarimax_hat", 
                                                                                               "snarimax_oxt", "snarimax_arf", "snarimax_amf"] else self.thresholds_by_resource.get('Mem', float('inf'))
                        # Gera o forecast completo para o futuro
                        steps_to_fail, path = self.forecasting.model.predict_until_failure(
                            last_observation, 
                            threshold_arg,
                            max_horizon=HORIZONTE_DE_PREVISAO 
                        )
                        
                        # Salva o caminho previsto no history_pred inteiro de uma vez!
                        if isinstance(path, dict): # Para VARMA
                            for res in self.resources:
                                history_pred[res] = path.get(res, [])
                        elif isinstance(path, list): # Para River
                            for res in self.resources:
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
                    self.monitor_process.terminate()

                print("\n" + "="*50)
                print(f"RELATORIO DE PRECISAO DO HORIZONTE DE PREVISAO: {self.model_name.upper()}")
                print("="*50)
                
                # Calcula as métricas APENAS para o período de previsão (comparando o futuro real com o previsto)
                for res in self.resources:
                    reais_futuro = history_real[res][SPLIT_STEP : SPLIT_STEP + HORIZONTE_DE_PREVISAO]
                    preds_futuro = history_pred[res][:len(reais_futuro)] # Garante mesmo tamanho
                    
                    if len(reais_futuro) > 0 and len(preds_futuro) > 0:
                        metricas = calculate_metrics(reais_futuro, preds_futuro)
                        todas_metricas_erro[res] = metricas # Guarda para passar para o gráfico
                        
                        print(f"Recurso: {res}")
                        print(f"  - MAD  (Erro Absoluto): {metricas['MAD']}")
                        print(f"  - MSD  (Erro Quadratico): {metricas['MSD']}")
                        print(f"  - MAPE (Erro Percentual): {metricas['MAPE']}%")
                        print("-" * 50)

                if self.output_directory:
                    base_dir = self.output_directory
                else:
                    base_dir = self.directory_path if is_replay_mode else os.path.dirname(self.filename)
                
                # chamada do utils 
                save_metrics_to_txt(
                    model_name=self.model_name,
                    split_step=SPLIT_STEP,
                    horizon=HORIZONTE_DE_PREVISAO,
                    metrics_dict=todas_metricas_erro,
                    base_dir=base_dir
                )

                # Gerar e Salvar o Gráfico
                if self.save_plot and len(timestamps) > 0:
                    # Define qual diretório base enviar para o utilitário
                    if self.output_directory:
                        base_dir = self.output_directory
                    else:
                        base_dir = self.directory_path if is_replay_mode else self.filename
                    
                    # Chama a função externa para fazer o trabalho visual
                    generate_individual_plots(
                        resources=self.resources,
                        timestamps=timestamps,
                        history_real=history_real,
                        history_pred=history_pred,
                        model_name=self.model_name,
                        base_path=base_dir,
                        is_replay_mode=is_replay_mode,
                        split_step=SPLIT_STEP,      
                        metricas_erro=todas_metricas_erro 
                    )

    def __trigger_rejuvenation(self):
        for process in psutil.process_iter(attrs=["pid", "name"]):
            if self.process_name.lower() in process.info["name"].lower():
                self.__restart_process(
                    process, self.start_command, self.restart_command
                )
                break

class FrameworkConfig:
    def __init__(self,
        split_step_override=None,
        horizonte_override=None,
        output_dir_override=None
        ):

        with open("config.yaml", "r", encoding="utf-8") as yml_file:
            config = yaml.load(yml_file, Loader=yaml.FullLoader)

        # Permite sobrescrever parâmetros via linha de comando, para automação com scripts
        if split_step_override is not None:
            config["general"]["split_step"] = split_step_override
            
        if horizonte_override is not None:
            config["general"]["horizonte_de_previsao"] = horizonte_override
            
        if output_dir_override is not None:
            config["general"]["output_directory"] = output_dir_override

        framework = Framework(
            **config["general"], **config["monitoring"], **config["real_time"]
        )
        framework.run()
