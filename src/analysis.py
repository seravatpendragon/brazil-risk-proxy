# ===================================================================================
# MÓDULO DE ANÁLISE DE DRIVERS - IARP
#
# Descrição:
# Contém funções para analisar e interpretar as previsões do modelo,
# especificamente decompondo a previsão mais recente em suas features
# contribuintes e plotando o resultado final.
# ===================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def analyze_prediction_drivers(X_full, hybrid_target, stable_features, predictions_dynamic):
    """
    Analisa os drivers da previsão mais recente, mostrando a contribuição de cada feature.
    """
    print("\n" + "="*80)
    print("🔬 ANÁLISE DE DRIVERS DA ÚLTIMA PREVISÃO")
    print("="*80)

    DATE_TO_ANALYZE = predictions_dynamic.index.max()
    print(f"🔍 Analisando a previsão para a data: {DATE_TO_ANALYZE.strftime('%Y-%m-%d')}\n")

    df = pd.concat([X_full, hybrid_target.rename('y')], axis=1).dropna(subset=['y'])
    end_tr = df.index[df.index < DATE_TO_ANALYZE].max()
    start_tr = end_tr - pd.DateOffset(months=36)
    train_slice = df.loc[(df.index > start_tr) & (df.index <= end_tr)]

    X_tr_raw, y_tr_raw = train_slice[stable_features], train_slice['y']
    X_tr_filled = X_tr_raw.fillna(method='ffill').fillna(method='bfill')
    valid_rows = ~X_tr_filled.isna().any(axis=1)
    X_tr, y_tr = X_tr_filled[valid_rows], y_tr_raw[valid_rows]

    pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=5.0))]).fit(X_tr, y_tr)
    scaler, model = pipeline.named_steps['scaler'], pipeline.named_steps['ridge']

    X_full_ffilled = X_full.fillna(method='ffill')
    feature_values = X_full_ffilled.loc[DATE_TO_ANALYZE, stable_features].astype(float)

    X_scaled = (feature_values.values - scaler.mean_) / scaler.scale_
    contributions = model.coef_ * X_scaled

    analysis_df = pd.DataFrame({
        'Valor Real': feature_values.values,
        'Coeficiente': model.coef_,
        'Contribuição (bps)': contributions
    }, index=stable_features).sort_values(by='Contribuição (bps)', ascending=False)

    print(f"Previsão Base (Intercepto): {model.intercept_:.2f} bps")
    print("-" * 60)
    print(f"📈 TOP FATORES QUE MAIS AUMENTAM O RISCO:")
    print(analysis_df.head(7).to_string())
    print("\n" + "-" * 60 + "\n")
    print(f"📉 TOP FATORES QUE MAIS REDUZEM O RISCO:")
    print(analysis_df.tail(7).sort_values(by='Contribuição (bps)').to_string())

    reconstructed_pred = model.intercept_ + contributions.sum()
    print("\n" + "="*80)
    print(f"Soma (Base + Contribuições): {reconstructed_pred:.2f} bps")
    print(f"Previsão Original do Modelo: {predictions_dynamic.loc[DATE_TO_ANALYZE]:.2f} bps")
    if np.isclose(reconstructed_pred, predictions_dynamic.loc[DATE_TO_ANALYZE], atol=0.1):
        print("✅ Validação bem-sucedida. A análise reflete a predição.")

def plot_final_indicator_comparison(final_predictions, hybrid_target, embi_series, cds_series):
    """
    Plota o IARP final contra seus componentes (EMBI e CDS) e o alvo híbrido.
    """
    print("\n" + "="*80)
    print("📊 GERANDO GRÁFICO DE COMPARAÇÃO FINAL")
    print("="*80)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(hybrid_target.index, hybrid_target, label='Alvo Híbrido (Real)', color='black', linewidth=2.5, alpha=0.7, zorder=3)
    ax.plot(final_predictions.index, final_predictions, label='Indicador IARP v7.0 (Modelo)', color='royalblue', linestyle='--', linewidth=2.5, zorder=4)
    ax.plot(embi_series.index, embi_series, label='EMBI+ (Componente)', color='skyblue', linestyle=':', linewidth=2, alpha=0.8, zorder=1)
    ax.plot(cds_series.index, cds_series, label='CDS 5 Anos (Componente)', color='coral', linestyle=':', linewidth=2, alpha=0.8, zorder=2)

    transition_date = pd.to_datetime('2024-07-31')
    ax.axvline(transition_date, color='grey', linestyle='-.', lw=1.5, label=f'Transição EMBI/CDS ({transition_date.date()})')

    ax.set_title('IARP v7.0 vs. Componentes de Risco (EMBI+ e CDS)', fontsize=18, pad=20)
    ax.set_ylabel('Risco (bps)', fontsize=12)
    ax.set_xlabel('Data', fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

