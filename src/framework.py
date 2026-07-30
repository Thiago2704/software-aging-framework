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

# Importando estratégias específicas
from src.strategies.online_learning_strategy import OnlineLearningStrategy
from src.strategies. offline_experiment_strategy import OfflineExperimentStrategy
from src.strategies.experiment_strategy import ExperimentStrategy

class Framework:

    """
    Motor central de orquestração do Software Aging Framework.

    Atua como o Contexto no padrão de projeto Strategy. E responsável por 
    inicializar os processos de monitoramento de recursos, instanciar os modelos 
    preditivos (via Factory) e, com base nas configurações, selecionar e delegar 
    a execução para a estratégia adequada (Online, Offline).
    Também fornece métodos utilitários globais, como reiniciar processos e desenhar logs.
    """

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
        self.online_models = ["arimax", "sarimax",
                              "snarimax_ht","snarimax_hat",
                              "snarimax_oxt", "snarimax_arf", "snarimax_amf"] # Lista de modelos que usam aprendizado online
        
        # variaveis auxiliares para automação com script
        self.split_step = split_step
        self.horizonte_de_previsao = horizonte_de_previsao
        self.output_directory = output_directory

        # Define a estratégia atual
        self.strategy = self.__choose_strategy()

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

    def __choose_strategy(self):
        """Seleciona e retorna a classe de estratégia apropriada."""
        if self.model_name in self.online_models:
            return OnlineLearningStrategy()
        elif self.run_in_real_time:
            return OfflineExperimentStrategy()
        else:
            return ExperimentStrategy()
 
    def run(self):
        """Delega a execução para a estratégia instanciada."""
        self.strategy.execute(self)

    def restart_process(
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

    def print_progress_bar(self, current_second, text):
        progress_bar_size = 50
        current_progress = (current_second + 1) / self.monitoring_time_in_seconds
        sys.stdout.write(
            f"\r{text}: [{'=' * int(progress_bar_size * current_progress):{progress_bar_size}s}] "
            f"{current_second + 1}/{self.monitoring_time_in_seconds} seconds"
        )
        sys.stdout.flush()

    def countdown(self):
        for current_second in range(self.monitoring_time_in_seconds):
            self.__print_progress_bar(current_second, "Monitoring")
            time.sleep(self.monitoring_interval_in_seconds)
        print()

    def plot_graph(self):
        self.forecasting.plot_results()

        if self.save_plot:
            path_to_save = self.filename.replace(".csv", ".png")
            plt.savefig(path_to_save, dpi=300)

    def trigger_rejuvenation(self):
        for process in psutil.process_iter(attrs=["pid", "name"]):
            if self.process_name.lower() in process.info["name"].lower():
                self.__restart_process(
                    process, self.start_command, self.restart_command
                )
                break

class FrameworkConfig:
    """
    Carregador e tradutor do arquivo YAML para o motor central.

    Lê as configurações de `config.yaml` e permite a injeção via código de 
    parâmetros chave.
    """
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
