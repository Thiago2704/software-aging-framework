import argparse
from src.framework import FrameworkConfig

"""
Ponto de entrada principal do Software Aging Framework.

Este script atua como a interface de linha de comando (CLI) para inicializar
e orquestrar o framework. Ele delega a execução para o `FrameworkConfig`, 
que lê as configurações padrão do arquivo `config.yaml`.

Adicionalmente, este script permite a injeção (sobrescrita) dinâmica de 
parâmetros-chave através do terminal. Isso facilita a automação de testes 
e a criação de cenários de experimentação em lote (orquestrados por 
scripts externos como o `run_experiments.py`).

Uso básico (utiliza apenas o config.yaml):
    python main.py

Uso avançado (sobrescrevendo parâmetros via CLI):
    python main.py --split_step 300 --horizonte 96 --output_dir ./resultados/exp1
"""
if __name__ == "__main__":
    # Configura o "escutador" de linha de comando
    parser = argparse.ArgumentParser(description="Laboratório de Envelhecimento de Software")

    parser.add_argument('--split_step', type=int, default=None, help='Passo de corte temporal')
    parser.add_argument('--horizonte', type=int, default=None, help='Horizonte de previsão')
    parser.add_argument('--output_dir', type=str, default=None, help='Pasta para salvar os resultados')

    # Lê o que foi digitado
    args = parser.parse_args()

    # Instancia as configurações do Framework. 
    # Caso um parâmetro não tenha sido passado via CLI, ele será None, 
    # e o FrameworkConfig utilizará o valor padrão definido no config.yaml.
    FrameworkConfig(
        split_step_override=args.split_step,
        horizonte_override=args.horizonte,
        output_dir_override=args.output_dir
    )