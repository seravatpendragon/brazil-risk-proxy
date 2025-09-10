# ===================================================================================
# MÓDULO DE TESTES DE VALIDAÇÃO - IARP
#
# Descrição:
# Contém a suíte de testes para validar o modelo IARP, incluindo testes de
# sanidade dos dados, multicolinearidade, leakage, benchmarking e robustez.
# ===================================================================================

import numpy as np
import pandas as pd
from numpy.linalg import cond
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from statsmodels.tsa.api import SimpleExpSmoothing

from src.model_pipeline import run_temporal_cv

def test_data_sanity(X_final):
    print("\n--- 1. TESTES DE INGESTÃO & SANIDADE DOS DADOS ---")
    idx = X_final.index
    print(f"\n[1.1] Consistência Temporal:")
    print(f"      Índice é monotônico (ordenado)? {idx.is_monotonic_increasing}")
    print(f"      Frequência inferida: {pd.infer_freq(idx)}")
    print("      Interpretação: Esperamos True e 'MS' (Month Start).")

    numeric_cols = X_final.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        summary_df = pd.DataFrame({
            'na_pct': X_final[numeric_cols].isna().mean()*100,
            'q01': X_final[numeric_cols].quantile(0.01),
            'median': X_final[numeric_cols].median(),
            'q99': X_final[numeric_cols].quantile(0.99),
        }).sort_values('na_pct', ascending=False)
        print("\n[1.2] Análise de Dados Faltantes e Extremos (Top 10):")
        print(summary_df.head(10))
        print("      Interpretação: Colunas com alto % de NAs podem ser menos confiáveis.")
    else:
        print("\n[1.2] Não há colunas numéricas para analisar.")

def test_multicollinearity(X_final, stable_features):
    print("\n--- 2. TESTES DE ESTABILIDADE DAS FEATURES ---")
    X_st = X_final[stable_features].select_dtypes(include=np.number)
    X_st_cleaned = X_st.dropna()
    if X_st_cleaned.shape[1] > 1 and X_st_cleaned.shape[0] > 0:
        condition_number = cond(X_st_cleaned.values)
        print(f"\n[2.1] Checagem de Multicolinearidade:")
        print(f"      Número de Condição das features estáveis: {condition_number:.2e}")
        print("      Interpretação: >1e3 indica multicolinearidade. O modelo Ridge já mitiga isso.")
    else:
        print("\n[2.1] Não há features numéricas suficientes para calcular multicolinearidade.")

def test_leakage_shuffle(X_final, y, stable_features):
    print("\n--- 3. TESTES DE LEAKAGE (VAZAMENTO DE DADOS) ---")
    print("\n[3.1] Teste de Embaralhamento (Shuffle Time Test):")
    rmse_original = run_temporal_cv(X_final, y, stable_features)
    if np.isnan(rmse_original):
        print("      ⚠️ Não foi possível rodar CV original. Pulando Shuffle Test.")
        return

    y_shuffled = y.copy()
    y_shuffled.iloc[:] = y.sample(frac=1, random_state=42).values
    rmse_shuffled = run_temporal_cv(X_final, y_shuffled, stable_features)

    print(f"      RMSE médio (CV original): {rmse_original:.2f} bps")
    print(f"      RMSE médio (CV com alvo embaralhado): {rmse_shuffled:.2f} bps")
    print("      Interpretação: O erro com o alvo embaralhado deve ser drasticamente maior.")
    if np.isnan(rmse_shuffled) or rmse_shuffled < rmse_original * 1.5:
        print("      ⚠️ ALERTA: O erro não aumentou significativamente. Pode haver leakage.")
    else:
        print("      ✅ SUCESSO: O erro aumentou como esperado, sem sinais de leakage.")

def test_benchmarks(y, model_preds):
    print("\n--- 4. TESTES DE BENCHMARKING ---")
    common_idx = y.index.intersection(model_preds.index)
    if common_idx.empty:
        print("  ⚠️ Sem dados em comum para benchmarking. Pulando testes.")
        return

    y_common = y.loc[common_idx]
    model_preds_common = model_preds.loc[common_idx]

    naive_preds = y_common.shift(1).dropna()
    y_naive_aligned = y_common.loc[naive_preds.index]
    if not naive_preds.empty:
        mae_naive = mean_absolute_error(y_naive_aligned, naive_preds)
        print(f"\n[4.1] Benchmark Ingênuo (Naive lag-1):")
        print(f"      MAE do Modelo IARP: {mean_absolute_error(y_common, model_preds_common):.2f} bps")
        print(f"      MAE do Benchmark Ingênuo: {mae_naive:.2f} bps")
        if mean_absolute_error(y_common, model_preds_common) < mae_naive - 1e-2:
            print("      ✅ SUCESSO: O modelo supera o benchmark ingênuo.")
        else:
            print("      ⚠️ ALERTA: O modelo não apresenta ganho sobre o benchmark ingênuo.")

    # Adicionado: Benchmark de Suavização Exponencial (SES)
    holdout_months = 12
    if len(y) > holdout_months:
        test_end_date = y.dropna().index.max()
        if pd.isna(test_end_date):
            print(f"\n[4.2] Dados insuficientes para o benchmark SES.")
            return

        test_start_date = test_end_date - pd.DateOffset(months=holdout_months-1)
        y_train_ses = y[y.index < test_start_date].dropna()
        y_test_ses = y[(y.index >= test_start_date) & (y.index <= test_end_date)].dropna()

        if len(y_train_ses) >= 2 and len(y_test_ses) > 0:
            try:
                ses_model = SimpleExpSmoothing(y_train_ses, initialization_method="estimated").fit()
                ses_forecast = ses_model.forecast(len(y_test_ses))
                ses_forecast.index = y_test_ses.index
                mae_ses = mean_absolute_error(y_test_ses, ses_forecast)

                model_holdout_preds = model_preds_common.loc[y_test_ses.index].dropna()
                y_model_holdout_aligned = y_test_ses.loc[model_holdout_preds.index]

                if not model_holdout_preds.empty:
                    model_holdout_mae = mean_absolute_error(y_model_holdout_aligned, model_holdout_preds)
                    print(f"\n[4.2] Benchmark Univariado (Suavização Exponencial):")
                    print(f"      Período de Holdout: {y_test_ses.index.min().date()} a {y_test_ses.index.max().date()}")
                    print(f"      MAE do Modelo IARP (Holdout): {model_holdout_mae:.2f} bps")
                    print(f"      MAE do Benchmark SES (Holdout): {mae_ses:.2f} bps")
                    if model_holdout_mae < mae_ses - 1e-2:
                        print("      ✅ SUCESSO: O modelo supera o benchmark de série temporal no holdout.")
                    else:
                        print("      ⚠️ ALERTA: O modelo não supera um benchmark univariado simples no holdout.")
            except Exception as e:
                print(f"\n[4.2] ⚠️ ERRO ao rodar Benchmark SES: {e}")

def test_robustness(X_final, y, stable_features):
    print("\n--- 5. TESTES DE ROBUSTEZ E SENSIBILIDADE ---")
    print("\n[5.1] Teste de Estresse (Choque Cambial de +20%):")

    # Corrigido: Garante que estamos usando apenas datas que existem tanto em X quanto em y
    common_idx = X_final.index.intersection(y.index)
    data = pd.concat([X_final[stable_features], y], axis=1).loc[common_idx]

    if len(data['y'].dropna()) < 37:
        print("      ⚠️ Não há dados suficientes (mínimo 37 meses com alvos válidos) para o teste de estresse.")
        return

    # Define o período de treino usando os dados combinados e alinhados
    train_end_date = data['y'].dropna().index.max()
    train_data = data[data.index <= train_end_date]
    X_train, y_train = train_data[stable_features], train_data['y']

    X_train_cleaned = X_train.fillna(method='ffill').fillna(method='bfill').dropna(how='any')
    y_train_cleaned = y_train.loc[X_train_cleaned.index]

    if len(X_train_cleaned) < 12:
        print("      ⚠️ Não há dados de treino limpos suficientes para o teste de estresse.")
        return

    pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=5.0))])
    pipeline.fit(X_train_cleaned, y_train_cleaned)

    pred_date = train_end_date + pd.DateOffset(months=1)
    if pred_date not in X_final.index:
        print(f"      ⚠️ Não há dados de feature para a data de previsão {pred_date.date()}. Pulando.")
        return

    X_pred_filled = X_final.loc[:pred_date, stable_features].fillna(method='ffill').iloc[-1].to_frame().T
    if X_pred_filled.isna().any().any():
        print(f"      ⚠️ Não foi possível preencher features para {pred_date.date()}. Pulando.")
        return

    pred_base = pipeline.predict(X_pred_filled.values.reshape(1, -1))[0]
    print(f"      Previsão base para {pred_date.date()}: {pred_base:.2f} bps")

    X_shocked_df = X_pred_filled.copy()
    cambio_lag0_col = 'cambio_lag0'
    if cambio_lag0_col in X_shocked_df.columns:
        X_shocked_df[cambio_lag0_col] *= 1.20
        pred_shocked = pipeline.predict(X_shocked_df.values.reshape(1, -1))[0]
        print(f"      Previsão PÓS-CHOQUE cambial: {pred_shocked:.2f} bps")
        print(f"      Impacto do choque: {pred_shocked - pred_base:+.2f} bps")
        if pred_shocked > pred_base:
            print("      ✅ SUCESSO: Choque cambial aumentou a previsão de risco.")
        else:
            print("      ⚠️ ALERTA: Choque cambial não aumentou a previsão de risco.")
    else:
        print(f"      ⚠️ Coluna '{cambio_lag0_col}' não encontrada. Pulando teste.")