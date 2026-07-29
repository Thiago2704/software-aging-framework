import time
import psutil
import pandas as pd
from src.forecasting import Forecasting
from src.utils import normalize, denormalize
from src.strategies.execution_strategy import ExecutionStrategy

class OfflineExperimentStrategy(ExecutionStrategy):
    """
    Estratégia de execução para inferência em tempo real utilizando modelos pré-treinados (Offline).

    Esta classe é responsável por monitorar o sistema vivo e aplicar modelos de aprendizado 
    em lote (batch learning, como Redes Neurais/LSTM) para prever anomalias.

    O fluxo de operação ocorre nas seguintes etapas:
    1. Inicialização/Setup: Acumula dados iniciais para um treino rápido ou carrega os 
       pesos de um modelo previamente treinado a partir de um arquivo `.h5`.
    2. Coleta Contínua: Em intervalos regulares, lê as últimas métricas gravadas pelo monitor.
    3. Pré-processamento: Aplica normalização (Min-Max) e remodela (reshape) os dados 
       na forma de janelas deslizantes requeridas pelo modelo.
    4. Inferência e Mitigação: Realiza a previsão para os próximos N passos. Se alguma 
       previsão ultrapassar os limites (thresholds) definidos, aciona o rejuvenescimento 
       (reiniciando o processo alvo).
    5. Exportação: Salva todas as previsões geradas num ficheiro CSV de log.
    """

    def execute(self, context):
        context.monitor_process.start()
        time.sleep(1)

        if context.run_monitoring:
            if context.error_queue.qsize() > 0:
                print("\nError monitoring process\n")
                return

            context.countdown()
            dataframe = pd.read_csv(context.filename)

            if dataframe.shape[0] < 4:
                print("\nNot enough monitoring data for forecasting, monitor for longer time\n")
                return

            context.forecasting = Forecasting(
                dataframe, context.model_name, context.resources, context.path_to_save_weights
            )
            context.forecasting.train()

        elif context.path_to_load_weights:
            dataframe = pd.read_csv(context.normalization_log_path)
            context.forecasting = Forecasting(
                dataframe,
                context.model_name,
                context.resources,
                context.path_to_save_weights,
                False,
                context.path_to_load_weights,
            )
        else:
            print("\nUnable to run if monitoring has not been run or model path has not been passed\n")
            context.monitor_process.terminate()
            return

        # dictionary to store predictions over time (all predictions of number_of_predictions)
        predictions_over_time = {
            f"{resource}_n{i + 1}": []
            for resource in context.resources
            for i in range(context.number_of_predictions)
        }
        running = True

        while running:
            time.sleep(context.monitoring_interval_in_seconds)

            # collect real-time monitoring data
            current_data = pd.read_csv(context.filename)
            current_data = current_data[context.resources]

            # check if the current data has enough rows for forecasting
            if current_data.shape[0] < 4:
                continue

            n_steps = 2
            n_seq = 2
            normalization_params = {}

            for resource in context.resources:
                current_data[resource], s_min, s_max = normalize(current_data[resource])
                normalization_params[resource] = (s_min, s_max)

            # the last 4 rows of the current data are used for forecasting (n_steps = 4 or n_seq = 2 and n_steps = 2)
            reshaped_current_data = current_data[-4:].values.reshape(
                (1, n_seq, 1, n_steps, len(context.resources))
            )

            # perform forecasting using the trained model
            predictions = context.forecasting.predict_future(
                reshaped_current_data, context.number_of_predictions
            )

            flag_list = []
            
            # compare predictions with thresholds and update flag_list and plot the results
            for idx, resource in enumerate(context.resources):
                s_min, s_max = normalization_params[resource]
                denormalized_predictions = denormalize(predictions[:, idx], s_min, s_max)

                for i, pred_value in enumerate(denormalized_predictions):
                    predictions_over_time[f"{resource}_n{i + 1}"].append(pred_value)

                    if pred_value > context.thresholds_by_resource[resource]:
                        flag_list.append(1)
                    else:
                        flag_list.append(0)
            # check if rejuvenation should be triggered
            if flag_list.count(1) > 0:
                print("\nActivated Rejuvenation\n")
                print("Flag list:", flag_list)

                for process in psutil.process_iter(attrs=["pid", "name"]):
                    if context.process_name.lower() in process.info["name"].lower():
                        context.restart_process(
                            process, context.start_command, context.restart_command
                        )
                        running = False
                        break
        
        context.monitor_process.terminate()
        # save the predictions over time in a csv file
        predictions_over_time_df = pd.DataFrame(predictions_over_time)
        predictions_over_time_df.to_csv(
            context.filename.replace(".csv", "_predictions.csv"), index=False
        )