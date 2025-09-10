# ===================================================================================
# MÓDULO DO PIPELINE DE MODELAGEM - IARP
#
# Descrição:
# Contém as funções principais para o treinamento do modelo, incluindo a seleção
# de features estáveis via bootstrap e a validação cruzada temporal.
# ===================================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Importa configurações do projeto
from src.config import N_BOOTSTRAP_RUNS, STABILITY_THRESHOLD, MIN_STABLE_FEATURES, N_CV_SPLITS

def select_stable_features(X, y, return_diagnostics=False):
    """
    Seleciona features estáveis usando um método de bootstrap com Ridge.
    """
    print("\n[FASE 2 de 4] Selecionando features estáveis...")
    common_idx = X.index.intersection(y.index)
    X_common, y_common = X.loc[common_idx].copy(), y.loc[common_idx].copy()

    variances = X_common.var()
    X_common = X_common.loc[:, variances > 1e-6]

    rng = np.random.RandomState(42)
    n = len(X_common)
    sample_size = max(1, int(n * 0.8))
    if sample_size > n: sample_size = n

    coeffs = []
    pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=5.0))])

    for _ in range(N_BOOTSTRAP_RUNS):
        idx = rng.choice(n, size=sample_size, replace=True)
        X_sample, y_sample = X_common.iloc[idx], y_common.iloc[idx]
        X_sample_filled = X_sample.fillna(method='ffill').fillna(method='bfill')
        valid_mask = ~X_sample_filled.isna().any(axis=1) & ~y_sample.isna()

        if valid_mask.sum() < 10: continue

        try:
            model = pipeline.fit(X_sample_filled[valid_mask], y_sample[valid_mask])
            coeffs.append(model.named_steps['ridge'].coef_)
        except ValueError:
            pass

    if not coeffs:
        print("  ❌ Nenhum modelo foi ajustado no bootstrap. Usando variância como fallback.")
        ranked = variances.sort_values(ascending=False).index.tolist()
        return ranked[:MIN_STABLE_FEATURES]

    coeffs_df = pd.DataFrame(coeffs, columns=X_common.columns)
    eps = 1e-8
    mean_abs_coefs = coeffs_df.abs().mean()
    cv_series = coeffs_df.std() / (mean_abs_coefs + eps)

    stable_features = cv_series[cv_series < STABILITY_THRESHOLD].index.tolist()
    if len(stable_features) < MIN_STABLE_FEATURES:
        stable_features = cv_series.sort_values().head(MIN_STABLE_FEATURES).index.tolist()

    print(f"    ✅ {len(stable_features)} features estáveis selecionadas.")
    if return_diagnostics:
        return stable_features, cv_series.loc[stable_features].sort_values()
    return stable_features


def run_temporal_cv(X, y, features):
    """
    Executa a Validação Cruzada Temporal e retorna o RMSE médio.
    """
    print(f"\n    Executando Validação Cruzada Temporal com {N_CV_SPLITS} dobras...")
    common_idx = X.index.intersection(y.index)
    X_cv, y_cv = X.loc[common_idx, features], y.loc[common_idx]

    if X_cv.empty or y_cv.empty:
        print("    ⚠️ Dados para CV estão vazios. Pulando.")
        return np.nan

    if len(X_cv) < N_CV_SPLITS + 1:
        print(f"    ⚠️ Dados insuficientes ({len(X_cv)} amostras) para {N_CV_SPLITS} dobras. Pulando.")
        return np.nan

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=5.0))])
    scores = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X_cv), start=1):
        X_train_raw, X_test_raw = X_cv.iloc[train_index], X_cv.iloc[test_index]
        y_train, y_test = y_cv.iloc[train_index], y_cv.iloc[test_index]

        X_train_filled = X_train_raw.fillna(method='ffill').fillna(method='bfill')
        X_train_cleaned = X_train_filled.dropna(how='any')
        y_train_cleaned = y_train.loc[X_train_cleaned.index]

        if len(X_train_cleaned) < 12:
            scores.append(np.nan)
            continue

        try:
            pipeline.fit(X_train_cleaned, y_train_cleaned)
            X_test_filled = X_test_raw.fillna(method='ffill').fillna(method='bfill')
            valid_test_mask = ~X_test_filled.isna().any(axis=1)
            X_test_cleaned = X_test_filled[valid_test_mask]
            y_test_cleaned = y_test[valid_test_mask]

            if X_test_cleaned.empty:
                scores.append(np.nan)
                continue

            preds = pipeline.predict(X_test_cleaned)
            rmse = np.sqrt(mean_squared_error(y_test_cleaned, preds))
            scores.append(rmse)
        except ValueError:
            scores.append(np.nan)

    mean_rmse = np.nanmean(scores)
    print(f"    RMSE médio nas {N_CV_SPLITS} dobras: {mean_rmse:.2f}")
    return mean_rmse
