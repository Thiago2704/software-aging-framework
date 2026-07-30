# Software Aging Framework

## 📖Visão Geral
O **Software Aging Framework** é um ambiente de experimentação e monitoramento criado para executar, avaliar e comparar diferentes modelos de previsão do envelhecimento de software. O repositório suporta a execução de testes em tempo real e em lote (offline), permitindo análises comparativas de métricas e plotagem de resultados. 

---

## 📂Estrutura do Repositório
A arquitetura do projeto está dividida em módulos focados em dados, modelagem preditiva, estratégias de execução e visualização:

```text
software-aging-framework/
├── .gitignore
├── README.md
├── config.yaml                     # Arquivo de configuração de parâmetros globais e de modelos
├── main.py                         # Ponto de entrada principal da aplicação
├── plot_real_time_experiments.py   # Script para geração de gráficos de experimentos em tempo real
├── requirements.txt                # Dependências do projeto (Python)
├── run_experiments.py              # Script automatizado para rodar a bateria de testes
└── src/                            # Código-fonte principal
    ├── __init__.py
    ├── data_loader.py              # Responsável por carregar e pré-processar os logs no modo replay.
    ├── forecasting.py              # Módulo de lógica de previsão
    ├── framework.py                # Orquestrador principal do framework
    ├── monitor.py                  # Ferramenta para monitoramento do estado do sistema
    ├── utils.py                    # Funções utilitárias e auxiliares
    ├── models/                     # Implementações de Modelos 
    └── strategies/                 # Padrões de execução (Online e Offline)
```

---

## 🏗️Módulos e Componentes Principais

### 1. 🧠Modelos Preditivos (src/models/)
Este diretório concentra a implementação algorítmica. O sistema usa uma abordagem de fábrica (model_factory.py) para instanciar diversos algoritmos, facilitando a introdução de novos métodos. Os modelos incluídos são:

*   Modelos de Séries Temporais Estatísticos: arimax, sarimax, moving average.
*   Modelos de Árvores: Hoeffding Tree Regressor(HT), Hoeffding Adaptive Tree Regressor(HAT).
*   Modelos de Florestas: Adaptive Random Forest(ARF), Aggregated Mondrian Forest(AMF), Online Extra Trees(OXT).
*   Modelos de Redes Neurais: h_lstm.py (Long Short-Term Memory adaptada).
*   Estruturas Base: model.py e online_model.py 

### 2. ⚙️Estratégias de Execução (src/strategies/)
O framework separa a lógica do modelo da forma como o experimento é conduzido, usando o padrão Strategy:
*   execution_strategy.py e experiment_strategy.py: Classes base para as estratégias.
*   offline_experiment_strategy.py: Executa simulações tradicionais de aprendizado em lote.
*   online_learning_strategy.py: Permite a avaliação contínua dos preditores, para fluxos de dados de envelhecimento em sistemas de longa duração e aprendizado online.

### 3. 🛠️Core e Utilitários (src/)
*   data_loader.py: Garante que os dados(logs do modo replay) sejam formatados adequadamente para os modelos.
*   forecasting.py e framework.py: Conectam o carregamento de dados aos modelos escolhidos e repassam os resultados para avaliação.
*   monitor.py: Monitora a degradação e métricas do próprio experimento ou do ambiente simulado.

### 4. Orquestração e Visualização (Raiz)
*   main.py e run_experiments.py: Podem ser configurados através do config.yaml para iterar sobre múltiplos modelos e conjuntos de dados, gerando saídas de forma automatizada.
*   plot_real_time_experiments.py: Consome os resultados das previsões e do envelhecimento para gerar recursos visuais.

---

## 🚀Sugestão de Guia de Uso (Setup)

1. Instalação das Dependências:
   Crie um ambiente virtual (venv) e instale as bibliotecas necessárias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # ou .\venv\Scripts\activate no Windows
   pip install -r requirements.txt
   ```

2. Configuração:
   Edite o arquivo config.yaml para definir hiperparâmetros dos modelos e apontar os caminhos das bases de dados.

3. Execução:
   Para rodar a suíte completa:
   ```bash
   python run_experiments.py
   ```