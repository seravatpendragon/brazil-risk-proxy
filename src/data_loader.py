# ===================================================================================
# MÓDULO DE CARREGAMENTO E PREPARAÇÃO DE DADOS - IARP
#
# Descrição:
# Contém funções para baixar dados de fontes externas (FRED, BCB/SGS) e
# carregar dados locais (CSV), além de construir o alvo híbrido e a matriz
# de features para o modelo.
# ===================================================================================

import os
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Módulos do projeto
from src.config import FRED_API_KEY, BCB_SERIES, FRED_SERIES, EMBI_PATH, CDS_PATH

def fred_series(series_id: str, start: str = '2014-01-01', end: str = '2025-12-31') -> pd.Series:
    """Busca uma série temporal da API do FRED."""
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}&observation_end={end}"
        r = requests.get(url, timeout=30); r.raise_for_status()
        obs = r.json().get('observations', [])
        if not obs: return pd.Series(dtype=float, name=series_id)
        df = pd.DataFrame(obs)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        s = df.dropna(subset=['value']).set_index('date')['value'].resample('MS').last()
        s.name = series_id
        return s
    except Exception as e:
        print(f"    ⚠️ Falha ao baixar série {series_id} do FRED: {e}")
        return pd.Series(dtype=float, name=series_id)

def sgs_series(code: int, start_date='01/01/2014', end_date='31/12/2025') -> pd.Series:
    """Busca uma série temporal da API do SGS/BCB."""
    try:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}"
        resp = requests.get(url, timeout=30); resp.raise_for_status()
        data = resp.json()
        if not data: return pd.Series(dtype=float, name=str(code))
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        s = df.dropna().set_index('data')['valor'].resample('MS').last()
        s.name = str(code)
        return s
    except Exception as e:
        print(f"    ⚠️ Falha ao baixar série {code} do SGS/BCB: {e}")
        return pd.Series(dtype=float, name=str(code))

def load_and_prep_data():
    """
    Orquestra o carregamento de todos os dados, cria features e o alvo híbrido.
    Retorna a matriz de features, o alvo híbrido e as séries originais de EMBI e CDS.
    """
    print("\n[FASE 1 de 4] Carregando dados e construindo features...")

    # 1. Carregar Alvo
    try:
        embi_df = pd.read_csv(EMBI_PATH); embi = embi_df.iloc[:, [0, 1]]; embi.columns = ['date', 'embi']
        embi['date'] = pd.to_datetime(embi['date']); embi = embi.set_index('date')['embi'].resample('MS').mean()
        embi = embi[embi.index <= '2024-07-31']
    except Exception as e:
        print(f"    ❌ ERRO ao processar arquivo EMBI: {e}")
        return None, None, None, None

    try:
        cds_df = pd.read_csv(CDS_PATH); cds_df['Data'] = pd.to_datetime(cds_df['Data'], format='%d.%m.%Y')
        cds_df['Último'] = cds_df['Último'].str.replace(',', '.').astype(float)
        cds = cds_df.set_index('Data')['Último'].resample('MS').last()
    except Exception as e:
        print(f"    ❌ ERRO ao processar arquivo CDS: {e}")
        return None, None, None, None

    # Calibração e criação do alvo híbrido
    transition_date = '2024-07-31'
    overlap_idx = embi.index.intersection(cds.index)
    overlap_idx = overlap_idx[(overlap_idx >= '2023-01-01') & (overlap_idx <= transition_date)]
    reg_df = pd.DataFrame({'cds': cds.loc[overlap_idx], 'embi': embi.loc[overlap_idx]}).dropna()
    lr = LinearRegression().fit(reg_df['cds'].values.reshape(-1, 1), reg_df['embi'].values)
    cds_future = cds[cds.index > transition_date]
    cds_adjusted = pd.Series(lr.predict(cds_future.values.reshape(-1, 1)), index=cds_future.index)
    hybrid_target = pd.concat([embi[embi.index <= transition_date], cds_adjusted]).sort_index().rename('y')

    # 2. Criar Features
    df = pd.DataFrame({**{k: sgs_series(v) for k, v in BCB_SERIES.items()}, **{k: fred_series(v) for k, v in FRED_SERIES.items()}}).sort_index().asfreq('MS')
    out = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            out[f'{col}_var_m'] = df[col].pct_change() * 100
            out[f'{col}_var_12m'] = df[col].pct_change(12) * 100
            out[f'{col}_vol_6m'] = out[f'{col}_var_m'].rolling(6).std()

    if 'selic_mensal' in df.columns and 'ipca' in df.columns:
        ipca_12m = df['ipca'].rolling(12).sum()
        out['taxa_real'] = df['selic_mensal'] - ipca_12m

    features_to_lag = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    X_lags = [out[features_to_lag].shift(lag).rename(columns=lambda c: f"{c}_lag{lag}") for lag in [0, 1, 2]]
    X_final = pd.concat(X_lags, axis=1).sort_index().asfreq('MS')

    print("    ✅ Dados carregados e features construídas.")
    return X_final, hybrid_target, embi, cds

