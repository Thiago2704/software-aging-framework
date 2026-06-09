import argparse
from src.framework import FrameworkConfig

# Ponto de entrada.
# Pode ser executado manualmente ou através de scripts de automação (run_experiments.py).
# aceita argumentos via linha de comando (CLI) para injetar dinamicamente 
# os cenários de teste (tamanho do histórico, horizonte futuro e local de salvamento).
# os cenarios são definidos em run_experiments.py, que chama este main.py passando os parâmetros via CLI.
if __name__ == "__main__":
    # Configura o "escutador" de linha de comando
    parser = argparse.ArgumentParser(description="Laboratório de Envelhecimento de Software")

    parser.add_argument('--split_step', type=int, default=None, help='Passo de corte temporal')
    parser.add_argument('--horizonte', type=int, default=None, help='Horizonte de previsão')
    parser.add_argument('--output_dir', type=str, default=None, help='Pasta para salvar os resultados')

    # Lê o que foi digitado
    args = parser.parse_args()

    # Chama a sua classe passando as variáveis (mesmo que sejam None)
    FrameworkConfig(
        split_step_override=args.split_step,
        horizonte_override=args.horizonte,
        output_dir_override=args.output_dir
    )