import time
from datetime import datetime
from multiprocessing import Process, Queue

import pandas as pd
import psutil


class ResourceMonitor:
    """
    Monitor de recursos do sistema operacional focado em um processo específico.

    Esta classe utiliza a biblioteca `psutil` para rastrear continuamente o uso de 
    CPU, Memória (RSS) e Disco associados a um processo alvo, salvando as leituras 
    em um arquivo CSV.
    """
    class ProcessNotFound(Exception):
        #Exceção levantada quando o processo alvo não é encontrado no sistema.
        pass

    def __init__(self, interval_in_seconds: int, process_name: str, filename: str):
        self.interval_in_seconds = interval_in_seconds
        self.filename = filename
        self.process_name = process_name
        self.process = None

    def monitor(self):
        """
        Inicia o loop contínuo de monitoramento.

        Busca o processo pelo nome, cria o arquivo CSV de destino com os cabeçalhos 
        e entra num laço infinito (while True) coletando e apensando as métricas a 
        cada intervalo definido.

        Raises:
            ProcessNotFound: Se o processo especificado não estiver em execução no momento da chamada.
        """

        self.process = self.__get_process()
        self.__create_file()

        while True:
            cpu_percent = self.process.cpu_percent()
            mem_info = self.process.memory_info()
            mem_usage = mem_info.rss / (1024**1)  # Memory usage in KB
            # TODO: monitor the process disk usage
            disk_usage = psutil.disk_usage("/").used / (
                1024**1
            )  # Disk usage in in KB

            timestamp = datetime.now()
            data = (timestamp, cpu_percent, mem_usage, disk_usage)

            dataframe = pd.DataFrame(
                [data], columns=["Timestamp", "CPU", "Mem", "Disk"]
            )
            dataframe.to_csv(self.filename, mode="a", index=False, header=False)

            time.sleep(self.interval_in_seconds)

    def __get_process(self):
        """
        Busca a instância do processo no sistema operacional pelo nome.

        Returns:
            psutil.Process: Instância do processo alvo para coleta de métricas.

        Raises:
            ProcessNotFound: Se nenhum processo contendo o nome procurado for encontrado.
        """

        for process in psutil.process_iter(attrs=["pid", "name"]):
            if self.process_name.lower() in process.info["name"].lower():
                return psutil.Process(process.info["pid"])
        raise self.ProcessNotFound(f"Process '{self.process_name}' not found.")

    def __create_file(self):
        dataframe = pd.DataFrame(columns=["Timestamp", "CPU", "Mem", "Disk"])
        dataframe.to_csv(self.filename, index=False)


class ResourceMonitorProcess(Process):
    """
    Processo isolado (Multiprocessing) para execução do ResourceMonitor.

    Garante que o loop infinito de coleta de métricas não bloqueie a linha de 
    execução principal (Main Thread) do framework. Qualquer exceção gerada 
    durante o monitoramento é capturada e enviada de volta ao processo pai 
    através de uma fila (Queue).
    """

    def __init__(
        self, interval_in_seconds: int, process_name: str, filename: str, queue: Queue
    ):
        """
        Inicializa o processo paralelo de monitoramento.

        Args:
            interval_in_seconds (int): Intervalo de coleta das métricas.
            process_name (str): Nome do processo do sistema operacional a ser monitorado.
            filename (str): Caminho do arquivo de saída (CSV).
            queue (Queue): Fila de comunicação inter-processos para envio de mensagens de erro.
        """

        super(ResourceMonitorProcess, self).__init__()
        self.resource_monitor = ResourceMonitor(
            interval_in_seconds, process_name, filename
        )
        self.queue = queue

    def run(self):
        """
        Método invocado automaticamente ao chamar `.start()` na instância do processo.
        """

        try:
            self.resource_monitor.monitor()
        except Exception as e:
            self.queue.put(str(e))
            raise e
