# Indicador Argus de Risco-País (IARP) v1.0
=========================================

O **IARP** é um modelo econométrico de código aberto projetado para estimar um proxy para o risco-país do Brasil. Utilizando dados macroeconômicos de fontes públicas (BCB, FRED), o projeto constrói um indicador mensal robusto e interpretável, validado através de uma rigorosa suíte de testes.

## Features
--------

-   **Pipeline Completo:** Automação de ponta a ponta, desde a coleta de dados e engenharia de features até a modelagem e geração do indicador final.

-   **Validação Robusta:** Inclui uma suíte de testes completa para garantir a qualidade do modelo, cobrindo sanidade dos dados, multicolinearidade, data leakage, benchmarking e testes de estresse.

-   **Análise de Drivers:** Oferece interpretabilidade ao decompor a previsão mais recente, mostrando quais variáveis mais contribuíram para o aumento ou redução do risco.

-   **Estrutura Modular:** O código é organizado em módulos com responsabilidades claras, facilitando a manutenção e a extensão do projeto.

## Estrutura do Projeto
--------------------

```
iarp-risco-pais/
│
├── data/                 # Arquivos de dados de entrada (EMBI+, CDS)
├── output/               # Local onde o indicador final é salvo
├── src/                  # Código fonte principal
│   ├── __init__.py
│   ├── analysis.py       # Análise de drivers e plotagem
│   ├── config.py         # Configurações centrais (paths, chaves de API)
│   ├── data_loader.py    # Funções de carregamento e preparação de dados
│   └── model_pipeline.py # Funções de treino e seleção de features
│
├── testes/               # Suíte de testes de validação do modelo
│   └── test_validation.py
│
├── .env                  # Arquivo para armazenar a chave da API do FRED (não versionado)
├── main.py               # Script principal para EXECUTAR o pipeline completo
├── requirements.txt      # Dependências do projeto
└── LICENSE               # Licença MIT

```

## Instalação e Setup
------------------

1.  **Clone o repositório:**

    ```
    git clone https://github.com/seravatpendragon/brazil-risk-proxy.git
    cd brazil-risk-proxy

    ```

2.  **Crie um ambiente virtual e instale as dependências:**

    ```
    python -m venv .venv
    source .venv/bin/activate  # No Windows: .venv\Scripts\activate
    pip install -r requirements.txt

    ```

3.  **Configure a Chave de API:**

    -   Renomeie o arquivo `.env.example` para `.env`.

    -   Abra o arquivo `.env` e insira sua chave da API do FRED:

        ```
        FRED_API_KEY="SUA_CHAVE_API_AQUI"

        ```

## Como Usar
---------

Para executar o pipeline completo (carregamento de dados, treinamento, validação e análise), basta rodar o script principal:

```
python main.py

```

Ao final da execução, o indicador final será salvo em `output/iarp_indicator.csv` e um gráfico de comparação será exibido.

## Licença
-------

Este projeto é licenciado sob a Licença MIT. Veja o arquivo [LICENSE](https://github.com/seravatpendragon/brazil-risk-proxy/blob/main/LICENSE "null") para mais detalhes.
