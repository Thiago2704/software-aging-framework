import time
import pandas as pd
from src.forecasting import Forecasting
from src.strategies.execution_strategy import ExecutionStrategy

# Essa classe representa a estratégia de execução para experimentos. 
# Ela herda da classe ExecutionStrategy e implementa o método execute, 
# que é responsável por executar a estratégia de experimento com base no 
# contexto fornecido.
class ExperimentStrategy(ExecutionStrategy):
    def execute(self, context):
        if context.run_monitoring:
            context.monitor_process.start()

            time.sleep(1)
            if context.error_queue.qsize() > 0:
                print("\nError monitoring process\n")
                return

            context.countdown()
            context.monitor_process.terminate()

        dataframe = pd.read_csv(context.filename)

        context.forecasting = Forecasting(
            dataframe, context.model_name, context.resources, context.path_to_save_weights
        )
        context.forecasting.train()
        context.plot_graph()