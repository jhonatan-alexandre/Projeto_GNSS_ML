"""Treina o Núcleo 1: XGBoost, SMOTE e validação temporal com gap."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)

from config import (  # noqa: E402
    CLASSIFICATION_TUNING,
    CLASSIFICATION_FEATURES,
    MODELS_DIRECTORY,
    PROCESSED_DATA,
    RANDOM_STATE,
    RESULTS_DIRECTORY,
    S4_PROBABILITY_THRESHOLD,
    STATION,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TSCV_GAP,
    TSCV_SPLITS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SCALE_POS_WEIGHT,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
)


PASTA_ESTACAO = os.path.join(PROCESSED_DATA, STATION)
DATASET = os.path.join(PASTA_ESTACAO, f"{STATION}_dataset_classificacao.csv")
PASTA_MODELO = os.path.join(MODELS_DIRECTORY, STATION, "nucleo1")
PASTA_RESULTADO = os.path.join(RESULTS_DIRECTORY, STATION, "nucleo1")
MODELO = os.path.join(PASTA_MODELO, "xgboost_s4.joblib")
PREDICOES = os.path.join(PASTA_RESULTADO, "predicoes_xgboost.csv")


def _dependencias() -> dict[str, Any]:
    try:
        from imblearn.over_sampling import SMOTE
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            average_precision_score,
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import TimeSeriesSplit
        from xgboost import XGBClassifier
    except ModuleNotFoundError as erro:
        raise ModuleNotFoundError(
            "Instale as dependências com 'python -m pip install -r requirements.txt'."
        ) from erro
    return locals()


def _novo_modelo(XGBClassifier, parametros=None):
    opcoes = dict(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        scale_pos_weight=XGB_SCALE_POS_WEIGHT,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=int(os.environ.get("GNSS_XGB_N_JOBS", "2")),
    )
    opcoes.update({k: v for k, v in (parametros or {}).items() if k != "balanceamento"})
    return XGBClassifier(**opcoes)


def _ajustar_amostragem(X: np.ndarray, y: np.ndarray, deps):
    imputer = deps["SimpleImputer"](strategy="median")
    X_imp = imputer.fit_transform(X)
    contagens = np.bincount(y.astype("int8"), minlength=2)
    if contagens.min() < 2:
        raise ValueError("Treino sem amostras suficientes das duas classes para SMOTE.")
    smote = deps["SMOTE"](
        random_state=RANDOM_STATE, k_neighbors=min(5, int(contagens.min()) - 1)
    )
    X_balanceado, y_balanceado = smote.fit_resample(X_imp, y)
    return imputer, X_balanceado, y_balanceado


def _metricas(y: np.ndarray, prob: np.ndarray, deps,
              limiar: float = S4_PROBABILITY_THRESHOLD) -> dict[str, Any]:
    if not len(y) or len(y) != len(prob) or not np.isfinite(prob).all():
        raise ValueError("Métricas exigem alvos e probabilidades finitos e não vazios.")
    pred = (prob >= limiar).astype("int8")
    tn, fp, fn, tp = deps["confusion_matrix"](y, pred, labels=[0, 1]).ravel()
    metricas = {
        "accuracy": float(deps["accuracy_score"](y, pred)),
        "precision": float(deps["precision_score"](y, pred, zero_division=0)),
        "recall": float(deps["recall_score"](y, pred, zero_division=0)),
        "f1": float(deps["f1_score"](y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_positivos": int(tp + fn),
        "n_amostras": int(len(y)),
        "limiar_probabilidade": float(limiar),
        "f2": float(5 * tp / (5 * tp + 4 * fn + fp)) if tp + fn + fp else 0.0,
    }
    metricas["roc_auc"] = (
        float(deps["roc_auc_score"](y, prob)) if np.unique(y).size == 2 else None
    )
    if "average_precision_score" in deps:
        metricas["average_precision"] = (
            float(deps["average_precision_score"](y, prob)) if (y == 1).any() else None
        )
    return metricas


def selecionar_limiar(y: np.ndarray, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Maximiza F2 em probabilidades de validação; desempate por maior limiar."""
    from sklearn.metrics import precision_recall_curve
    if len(y) != len(prob) or np.unique(y).size != 2 or not np.isfinite(prob).all():
        raise ValueError("Seleção exige duas classes e probabilidades finitas.")
    precision, recall, thresholds = precision_recall_curve(y, prob)
    precision, recall = precision[:-1], recall[:-1]
    f2 = np.divide(5 * precision * recall, 4 * precision + recall,
                   out=np.zeros_like(recall), where=(4 * precision + recall) > 0)
    curve = pd.DataFrame({"limiar": thresholds, "precision": precision,
                          "recall": recall, "f2": f2})
    best = curve.sort_values(["f2", "limiar"], ascending=[False, False]).iloc[0]
    return float(best["limiar"]), curve


def _ajustar_modelo(base: pd.DataFrame, deps, parametros=None):
    parametros = dict(parametros or {})
    X = base[list(CLASSIFICATION_FEATURES)].to_numpy()
    y = base["classe_future"].to_numpy()
    if parametros.get("balanceamento", "smote") == "pesos":
        contagens = np.bincount(y.astype("int8"), minlength=2)
        if not contagens.min():
            raise ValueError("Treinamento exige as duas classes.")
        imputer = deps["SimpleImputer"](strategy="median")
        X_bal, y_bal = imputer.fit_transform(X), y
        parametros["scale_pos_weight"] = float(contagens[0] / contagens[1])
    else:
        if parametros.get("balanceamento") == "smote":
            parametros["scale_pos_weight"] = 1.0
        imputer, X_bal, y_bal = _ajustar_amostragem(X, y, deps)
    modelo = _novo_modelo(deps["XGBClassifier"], parametros)
    modelo.fit(X_bal, y_bal)
    return imputer, modelo, {
        "classe_0_antes": int((y == 0).sum()), "classe_1_antes": int((y == 1).sum()),
        "classe_0_ajuste": int((y_bal == 0).sum()), "classe_1_ajuste": int((y_bal == 1).sum()),
        "balanceamento": parametros.get("balanceamento", "smote"),
        "scale_pos_weight": parametros.get("scale_pos_weight", XGB_SCALE_POS_WEIGHT),
    }


def executar_validacao(
    df: pd.DataFrame, deps, parametros=None,
    limiar: float = S4_PROBABILITY_THRESHOLD, max_folds: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Executa TSCV e devolve probabilidades fora da amostra para o treino."""
    treino = df.loc[df["target_datetime"] <= pd.Timestamp(TRAIN_END)].copy()
    tempos = np.sort(treino["target_datetime"].unique())
    if len(tempos) <= TSCV_SPLITS:
        raise ValueError("Poucas épocas para Multi-TSCV-GKF.")
    divisor = deps["TimeSeriesSplit"](n_splits=TSCV_SPLITS)
    resultados: list[dict[str, Any]] = []
    probabilidades_oof = pd.Series(np.nan, index=df.index, dtype="float64")
    gap = pd.Timedelta(TSCV_GAP)
    for fold, (_, idx_val) in enumerate(divisor.split(tempos), start=1):
        if max_folds is not None and fold > max_folds:
            break
        inicio_val = pd.Timestamp(tempos[idx_val[0]])
        fim_val = pd.Timestamp(tempos[idx_val[-1]])
        mascara_treino = treino["target_datetime"] < inicio_val - gap
        mascara_val = treino["target_datetime"].between(inicio_val, fim_val)
        base_treino = treino.loc[mascara_treino]
        base_val = treino.loc[mascara_val]
        if base_treino.empty or base_val.empty:
            continue
        imputer, modelo, contagens = _ajustar_modelo(base_treino, deps, parametros)
        prob = modelo.predict_proba(
            imputer.transform(
                base_val[list(CLASSIFICATION_FEATURES)].to_numpy()
            )
        )[:, 1]
        probabilidades_oof.loc[base_val.index] = prob
        resultados.append(
            {
                "fold": fold,
                "inicio_validacao": inicio_val,
                "fim_validacao": fim_val,
                "n_treino": len(base_treino),
                "n_validacao": len(base_val),
                **contagens,
                **_metricas(base_val["classe_future"].to_numpy(), prob, deps, limiar),
            }
        )
    if not resultados:
        raise ValueError("Nenhum fold temporal válido foi produzido.")
    return pd.DataFrame(resultados), probabilidades_oof


def selecionar_hiperparametros(df: pd.DataFrame, deps):
    """O último fold é reservado; o teste final não participa de nenhum ajuste."""
    if TSCV_SPLITS < 3:
        raise ValueError("Seleção temporal exige ao menos três folds.")
    candidatos = [
        {"n_estimators": 150, "max_depth": 6, "learning_rate": .05, "balanceamento": "smote"},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": .05, "min_child_weight": 5, "reg_lambda": 5., "balanceamento": "smote"},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": .03, "min_child_weight": 5, "reg_lambda": 5., "balanceamento": "pesos"},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": .05, "min_child_weight": 5, "reg_lambda": 5., "balanceamento": "pesos"},
    ]
    registros, curvas = [], []
    for indice, parametros in enumerate(candidatos):
        print(f"Seleção XGBoost {indice+1}/{len(candidatos)}: {parametros}", flush=True)
        _, oof = executar_validacao(df, deps, parametros=parametros, max_folds=TSCV_SPLITS - 1)
        validos = oof.notna()
        limiar, curva = selecionar_limiar(df.loc[validos, "classe_future"].to_numpy(), oof.loc[validos].to_numpy())
        linha = curva.loc[curva["limiar"].eq(limiar)].iloc[0].to_dict()
        registros.append({"candidato": indice, **parametros, **linha,
                           "n_selecao": int(validos.sum()), "folds_selecao": TSCV_SPLITS - 1})
        curva["candidato"] = indice
        curvas.append(curva)
    tabela = pd.DataFrame(registros)
    vencedor = tabela.sort_values(["f2", "precision", "candidato"], ascending=[False, False, True]).iloc[0]
    return candidatos[int(vencedor["candidato"])], float(vencedor["limiar"]), tabela, pd.concat(curvas, ignore_index=True)


def main() -> None:
    if not os.path.isfile(DATASET):
        raise FileNotFoundError(f"Execute primeiro o script 07: {DATASET}")
    os.makedirs(PASTA_MODELO, exist_ok=True)
    os.makedirs(PASTA_RESULTADO, exist_ok=True)
    deps = _dependencias()
    dados = pd.read_csv(DATASET, low_memory=True)
    dados["target_datetime"] = pd.to_datetime(dados["target_datetime"], errors="coerce")

    parametros, limiar = {}, S4_PROBABILITY_THRESHOLD
    if CLASSIFICATION_TUNING:
        parametros, limiar, candidatos, curvas = selecionar_hiperparametros(dados, deps)
        candidatos.to_csv(os.path.join(PASTA_RESULTADO, "busca_hiperparametros.csv"), index=False)
        curvas.to_csv(os.path.join(PASTA_RESULTADO, "curvas_limiar_validacao.csv"), index=False)
    validacao, probabilidades_oof = executar_validacao(dados, deps, parametros, limiar)
    validacao["uso"] = np.where(
        CLASSIFICATION_TUNING & validacao["fold"].lt(TSCV_SPLITS),
        "selecao_hiperparametros_descritivo", "validacao_independente" if CLASSIFICATION_TUNING else "validacao_fixa",
    )
    validacao.to_csv(os.path.join(PASTA_RESULTADO, "validacao_temporal.csv"), index=False)

    treino = dados.loc[dados["target_datetime"] <= pd.Timestamp(TRAIN_END)]
    imputer, modelo, contagens = _ajustar_modelo(treino, deps, parametros)
    with open(os.path.join(PASTA_RESULTADO, "hiperparametros_xgboost.json"), "w", encoding="utf-8") as arquivo:
        json.dump({"selecao": "F2 nos primeiros folds temporais; último fold reservado" if CLASSIFICATION_TUNING else "fixo",
                   "parametros": modelo.get_params(), "limiar_probabilidade": limiar,
                   "teste_usado_na_selecao": False, "revisao_manual_aplicada": False,
                   "amostragem_treino": contagens}, arquivo, ensure_ascii=False, indent=2)

    posteriores_ao_treino = dados["target_datetime"] > pd.Timestamp(TRAIN_END)
    probabilidades = probabilidades_oof.copy()
    probabilidades.loc[posteriores_ao_treino] = modelo.predict_proba(
        imputer.transform(
            dados.loc[posteriores_ao_treino, list(CLASSIFICATION_FEATURES)].to_numpy()
        )
    )[:, 1]
    predicoes = dados[["datetime", "target_datetime", "svid", "S4_future", "classe_future"]].copy()
    predicoes["probabilidade_severa"] = probabilidades
    predicoes["origem_probabilidade"] = "indisponivel_sem_passado"
    predicoes.loc[probabilidades_oof.notna(), "origem_probabilidade"] = "out_of_fold"
    predicoes.loc[posteriores_ao_treino, "origem_probabilidade"] = "modelo_final"
    predicoes["limiar_probabilidade"] = limiar
    predicoes["sinalizar_filtro"] = probabilidades.ge(limiar)
    predicoes.to_csv(PREDICOES, index=False, date_format="%Y-%m-%d %H:%M:%S")

    teste = predicoes["target_datetime"].between(
        pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    )
    if predicoes.loc[teste, "probabilidade_severa"].isna().any():
        raise ValueError("O modelo final não produziu todas as probabilidades de teste.")
    metricas_teste = _metricas(
        predicoes.loc[teste, "classe_future"].to_numpy(),
        predicoes.loc[teste, "probabilidade_severa"].to_numpy(),
        deps,
        limiar,
    )
    with open(os.path.join(PASTA_RESULTADO, "metricas_teste.json"), "w", encoding="utf-8") as arquivo:
        json.dump(metricas_teste, arquivo, indent=2)
    resumo_predicoes = {
        "limiar_probabilidade": limiar,
        "treino_observacoes": int((~posteriores_ao_treino).sum()),
        "treino_probabilidades_out_of_fold": int(
            probabilidades_oof.loc[~posteriores_ao_treino].notna().sum()
        ),
        "treino_sem_passado_suficiente": int(
            probabilidades_oof.loc[~posteriores_ao_treino].isna().sum()
        ),
        "gap_e_teste_probabilidades_modelo_final": int(posteriores_ao_treino.sum()),
    }
    with open(
        os.path.join(PASTA_RESULTADO, "resumo_predicoes.json"),
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(resumo_predicoes, arquivo, ensure_ascii=False, indent=2)
    joblib.dump(
        {
            "model": modelo,
            "imputer": imputer,
            "features": list(CLASSIFICATION_FEATURES),
            "probability_threshold": limiar,
        },
        MODELO,
    )
    print("Modelo salvo:", MODELO)
    print("Métricas de teste:", metricas_teste)


if __name__ == "__main__":
    main()
