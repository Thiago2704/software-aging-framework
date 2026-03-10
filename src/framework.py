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
from src.utils import DataAggregator, normalize, denormalize
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
        self.online_models = ["arf", "hat_perceptron", "isoup", "arimax", "sarimax", "varma"]


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
        if self.model_name in ["arf","hat_perceptron", "isoup", "arimax", "sarimax", "varma"]:
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

            # --- ADICIONE ESTE PRINT AQUI ---
            print("\n" + "="*40)
            print(f"DEBUG DE LIMITES (O que o Python vê):")
            print(f"Dicionário Completo: {self.thresholds_by_resource}")
            print("="*40 + "\n")
            # --------------------------------

            is_replay_mode = os.path.isdir(self.directory_path) and os.path.exists(os.path.join(self.directory_path, "cpu.csv"))
        
            # Variáveis de Controle
            data_stream = None
            aggregator = None

            if is_replay_mode:
                print(f" Modo leitura: Lendo logs históricos de {self.directory_path}")
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
            warmup_steps = 30 
            
            # Inicializar listas para o gráfico
            timestamps = []
            history_real = {res: [] for res in self.resources}
            history_pred = {res: [] for res in self.resources}

            aux_metrics = ['DiskIO', 'Frag_1', 'IOWait'] 
            history_aux = {res: [] for res in aux_metrics}

            running = True

            print(f"Monitoramento iniciado. Aquecendo modelo por {warmup_steps} segundos...")

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

                    # Se por algum motivo não tivermos features, pula
                    if features_mean is None: continue

                    # APRENDER
                    if last_observation is not None:
                        self.forecasting.model.learn_one(last_observation, features_mean)
                        learning_step += 1 

                    # Atualiza o buffer
                    last_observation = features_mean

                    # Warm-up check
                    if learning_step < warmup_steps:
                        print(f"\rAquecendo... {learning_step}/{warmup_steps} amostras aprendidas.", end="")
                        continue

                    if self.model_name in ["arf", "hat_perceptron", "arimax", "sarimax", "varma", "isoup"]:
                            # Modelo Novo: Aceita dict
                            threshold_arg = self.thresholds_by_resource
                    else:
                            # Modelos Antigos: Esperam float
                            threshold_arg = self.thresholds_by_resource.get('Mem', float('inf'))
                    # PREVER
                    steps_to_fail, path = self.forecasting.model.predict_until_failure(
                        features_mean, 
                        threshold_arg,
                        max_horizon=336 # Olha 100 blocos para frente
                    )

                    if isinstance(path, dict) and path:
                            # Se for ARMA (retorna Dict {'CPU': [val1, val2]}), pega o primeiro valor
                            immediate_pred = {r: path[r][0] for r in self.resources if r in path}
                    elif isinstance(path, list) and path:
                            # Se for Outros Modelos (retorna List [{'CPU': val1}, ...]), pega o índice 0
                            immediate_pred = path[0]
                    else:
                            # Se estiver vazio
                            immediate_pred = {r: 0 for r in self.resources}
                    
                    # Guardar dados para o gráfico
                    timestamps.append(len(timestamps) + 1)
                    for res in self.resources:
                        immediate_pred[res] = max(0, immediate_pred.get(res, 0))
                        history_real[res].append(features_mean[res])
                        history_pred[res].append(immediate_pred[res])

                    # Guardar métricas auxiliares
                    for res in aux_metrics:
                        val = features_mean.get(res, 0)
                        history_aux[res].append(val)

                    # Verificar Thresholds
                    flag_list = []
                    print(f"\nStatus Real: {features_mean} | Previsto: {immediate_pred}") 

                    if steps_to_fail != -1:
                        print(f"\nActivated Rejuvenation (Falha prevista em {steps_to_fail} blocos)\n")
                        if not is_replay_mode:
                            self.__trigger_rejuvenation()
                            running = False
                    
                    for res in self.resources:
                        pred_value = immediate_pred[res]
                        if pred_value > self.thresholds_by_resource[res]:
                            flag_list.append(1)
                            print(f"ALERTA: {res} previsto ({pred_value:.2f}) > Limite")
                        else:
                            flag_list.append(0)
                    
                    if 1 in flag_list:
                        print("\nActivated Rejuvenation (Online Model Predicted Failure)\n")
                        #Só reinicia se não for modo Replay
                        if not is_replay_mode:
                            for process in psutil.process_iter(attrs=["pid", "name"]):
                                if self.process_name.lower() in process.info["name"].lower():
                                    self.__restart_process(
                                        process, self.start_command, self.restart_command
                                    )
                                    running = False
                                    break
            
            except KeyboardInterrupt:
                print("\nMonitoramento interrompido pelo usuário.")

            finally: 
                if not is_replay_mode:
                    self.monitor_process.terminate()

                # Gerar e Salvar o Gráfico
                if self.save_plot and len(timestamps) > 0:
                    print("\nGerando gráfico de execução online...")
                    plt.figure(figsize=(12, 6))
                    
                    for idx, res in enumerate(self.resources):
                        plt.subplot(len(self.resources), 1, idx + 1)
                        plt.plot(timestamps, history_real[res], label=f'Real {res}', color='blue')
                        plt.plot(timestamps, history_pred[res], label=f'Predição {res} ({self.model_name})', color='green', linestyle='--')
                        plt.title(f"{res} - Real vs Online Prediction")
                        plt.legend()
                        plt.grid(True)
                    
                    plt.tight_layout()
                    if is_replay_mode:
                        path_to_save = os.path.join(self.directory_path, "replay_analysis_graph.png")
                    else: 
                        path_to_save = self.filename.replace(".csv", ".png")
                    plt.savefig(path_to_save)
                    print(f"Gráfico salvo em: {path_to_save}")

    def __trigger_rejuvenation(self):
        for process in psutil.process_iter(attrs=["pid", "name"]):
            if self.process_name.lower() in process.info["name"].lower():
                self.__restart_process(
                    process, self.start_command, self.restart_command
                )
                break

class FrameworkConfig:
    def __init__(self):
        with open("config.yaml", "r") as yml_file:
            config = yaml.load(yml_file, Loader=yaml.FullLoader)

        framework = Framework(
            **config["general"], **config["monitoring"], **config["real_time"]
        )
        framework.run()
