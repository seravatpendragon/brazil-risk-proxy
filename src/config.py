# ===================================================================================
# MÓDULO DE CONFIGURAÇÃO CENTRAL - IARP
#
# Descrição:
# Centraliza todas as configurações do projeto, como caminhos de arquivos,
# parâmetros de modelo, códigos de séries e chaves de API.
# ===================================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# --- 1. Carregamento de Variáveis de Ambiente ---
# Carrega as variáveis do arquivo .env para o ambiente
load_dotenv()


# --- 2. Chaves de API ---
# Busca a chave da API do FRED do ambiente.
# Garante que o script pare se a chave não for encontrada.
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("A FRED_API_KEY não foi encontrada. Verifique seu arquivo .env")


# --- 3. Caminhos Principais ---
# Define o caminho raiz do projeto de forma dinâmica para funcionar em qualquer máquina
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
EMBI_PATH = DATA_DIR / "ipeadata-embi.csv"
CDS_PATH = DATA_DIR / "Brasil-CDS-5-Anos-USD-Visão-Geral.csv"


# --- 4. Códigos das Séries Macroeconômicas ---
# Dicionários que mapeiam nomes legíveis para os códigos das APIs
BCB_SERIES = {
    'selic_mensal': 4390,
    'ipca': 433,
    'cambio': 3698,
    'reservas_int': 3546,
    'divida_bruta_pib': 4536,
    'ibc_br': 24363
}

FRED_SERIES = {
    'vix': 'VIXCLS',
    'ust10y': 'DGS10',
    'usd_broad': 'TWEXBGSMTH',
    'brent': 'WCOILBRENTEU',
    'iron_ore': 'PIORECRUSDM',
    'soybeans': 'PSOYBUSDM'
}

# --- 5. Parâmetros de Modelagem e Validação ---
# Seleção de Features
STABILITY_THRESHOLD = 0.4
N_BOOTSTRAP_RUNS = 20
MIN_STABLE_FEATURES = 15

# Validação Cruzada
N_CV_SPLITS = 5

