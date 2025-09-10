# ===================================================================================
# SCRIPT PRINCIPAL DE EXECUÇÃO - IARP v7.0
#
# Autor: Projeto Argus & Gemini (Refatorado)
# Data: 10 de setembro de 2025
#
# Descrição:
# Orquestra a execução completa do pipeline do IARP:
# 1. Carrega e prepara os dados.
# 2. Seleciona as features mais estáveis.
# 3. Gera previsões out-of-sample para avaliação.
# 4. Executa uma suíte completa de testes de validação.
# 5. Analisa os drivers da última previsão.
# 6. Plota o gráfico de comparação final.
# ===================================================================================

import warnings
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Módulos do projeto
from src.config import OUTPUT_DIR
from src.data_loader import load_and_prep_data
from src.model_pipeline import select_stable_features
from src.analysis import analyze_prediction_drivers, plot_final_indicator_comparison # Adicionado
from testes.test_validation import (
    test_data_sanity,
    test_multicollinearity,
    test_leakage_shuffle,
    test_benchmarks,
    test_robustness
)

# --- 0. CONFIGURAÇÕES GLOBAIS ---
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: f'{x:.2f}')
pd.set_option('display.width', 120)


def generate_final_predictions(X_final, y, stable_features):
    """
    Gera previsões out-of-sample usando uma janela de treino rolante.
    Essas previsões são usadas para os testes de benchmarking e robustez.
    """
    print("\n[FASE 3 de 5] Gerando previsões do modelo final para testes...")
    df_for_preds = pd.concat([X_final[stable_features], y], axis=1).dropna(subset=['y'])

    if df_for_preds.empty:
        print("    ❌ Nenhuma data com alvo e features válidos encontrada.")
        return pd.Series(dtype=float)

    model_preds = pd.Series(index=df_for_preds.index, dtype=float)
    window_months = 36
    min_train_samples = 12

    for t in df_for_preds.index:
        train_slice = df_for_preds.loc[df_for_preds.index < t]
        if len(train_slice) > window_months:
            train_slice = train_slice.iloc[-window_months:]

        X_tr, y_tr = train_slice[stable_features], train_slice['y']
        X_tr_cleaned = X_tr.fillna(method='ffill').fillna(method='bfill').dropna(how='any')
        y_tr_cleaned = y_tr.loc[X_tr_cleaned.index]

        if len(X_tr_cleaned) < min_train_samples:
            continue

        pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=5.0))])
        try:
            pipeline.fit(X_tr_cleaned, y_tr_cleaned)
            xt_pred_filled = X_final.loc[:t, stable_features].fillna(method='ffill').iloc[-1].to_frame().T

            if xt_pred_filled.isna().any().any():
                continue

            y_hat = pipeline.predict(xt_pred_filled.values.reshape(1, -1))[0]
            model_preds.loc[t] = y_hat
        except Exception:
            pass

    model_preds.dropna(inplace=True)
    print(f"    ✅ {len(model_preds)} previsões geradas para testes.")
    return model_preds


def main():
    """Função principal que executa todo o pipeline de validação."""
    print("="*80)
    print("=== SCRIPT DE VALIDAÇÃO FINAL E TESTE DE ESTRESSE - IARP v7.0 ===")
    print("="*80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # FASE 1 & 2: Carregar dados e selecionar features
    X_final, hybrid_target, embi_series, cds_series = load_and_prep_data()
    if X_final is None or hybrid_target is None:
        print("\n❌ Falha ao carregar dados. Abortando validação.")
        return

    stable_features = select_stable_features(X_final, hybrid_target)
    if not stable_features:
        print("\n❌ Nenhuma feature estável encontrada. Abortando validação.")
        return

    # FASE 3: Gerar previsões do modelo final
    model_preds = generate_final_predictions(X_final, hybrid_target, stable_features)

    # FASE 4: Executar suíte de testes
    print("\n[FASE 4 de 6] Executando suíte de testes de validação...")
    test_data_sanity(X_final)
    test_multicollinearity(X_final, stable_features)
    test_leakage_shuffle(X_final, hybrid_target, stable_features)

    if not model_preds.empty:
        test_benchmarks(hybrid_target, model_preds)
        test_robustness(X_final, hybrid_target, stable_features)

        # FASE 5: Análise de Drivers
        analyze_prediction_drivers(X_final, hybrid_target, stable_features, model_preds)

        output_path = OUTPUT_DIR / "iarp_indicator.csv"
        model_preds.to_csv(output_path, header=['iarp_prediction'])
        print(f"\n✅ Indicador salvo com sucesso em: {output_path}")

        # FASE 6: Visualização Final
        plot_final_indicator_comparison(model_preds, hybrid_target, embi_series, cds_series)
    else:
        print("\n⚠️ Previsões do modelo final não puderam ser geradas. Pulando testes de benchmark, robustez e análise.")


    print("\n" + "="*80)
    print("🏁 VALIDAÇÃO COMPLETA 🏁")
    print("="*80)


if __name__ == "__main__":
    main()

