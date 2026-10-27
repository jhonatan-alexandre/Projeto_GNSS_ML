"""Constrói o dataset para prever cintilação severa 30 minutos à frente."""

from __future__ import annotations

import json
import os
import sys

import pandas as pd


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)

from config import (  # noqa: E402
    CLASSIFICATION_FEATURES,
    FORECAST_HORIZON_STEPS,
    FORECAST_MINUTES,
    PROCESSED_DATA,
    S4_THRESHOLD,
    STATION,
    TIME_RESOLUTION,
)


PASTA_ESTACAO = os.path.join(PROCESSED_DATA, STATION)
ARQUIVO_ENTRADA = os.path.join(PASTA_ESTACAO, f"{STATION}_features.csv")
ARQUIVO_SAIDA = os.path.join(PASTA_ESTACAO, f"{STATION}_dataset_classificacao.csv")
ARQUIVO_RESUMO = os.path.join(PASTA_ESTACAO, "resumo_dataset_classificacao.json")


def criar_alvo(df: pd.DataFrame) -> pd.DataFrame:
    obrigatorias = {"datetime", "svid", "S4", *CLASSIFICATION_FEATURES}
    faltantes = sorted(obrigatorias.difference(df.columns))
    if faltantes:
        raise ValueError("Colunas ausentes: " + ", ".join(faltantes))

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values(["svid", "datetime"], kind="stable")
    grupos = df.groupby("svid", sort=False)
    df["S4_future"] = grupos["S4"].shift(-FORECAST_HORIZON_STEPS)
    df["target_datetime"] = grupos["datetime"].shift(-FORECAST_HORIZON_STEPS)

    esperado = pd.Timedelta(minutes=FORECAST_MINUTES)
    consecutivo = df["target_datetime"].sub(df["datetime"]).eq(esperado)
    df.loc[~consecutivo, ["S4_future", "target_datetime"]] = [pd.NA, pd.NaT]
    df = df.dropna(subset=["datetime", "target_datetime", "S4_future"]).copy()
    df["classe_future"] = (df["S4_future"] >= S4_THRESHOLD).astype("int8")
    colunas = [
        "datetime",
        "target_datetime",
        "svid",
        *CLASSIFICATION_FEATURES,
        "S4_future",
        "classe_future",
    ]
    return df[colunas].sort_values(["target_datetime", "svid"]).reset_index(drop=True)


def main() -> None:
    if not os.path.isfile(ARQUIVO_ENTRADA):
        raise FileNotFoundError(f"Execute primeiro o script 06: {ARQUIVO_ENTRADA}")
    dados = pd.read_csv(ARQUIVO_ENTRADA, low_memory=True)
    dataset = criar_alvo(dados)
    dataset.to_csv(ARQUIVO_SAIDA, index=False, date_format="%Y-%m-%d %H:%M:%S")
    resumo = {
        "estacao": STATION,
        "resolucao": TIME_RESOLUTION,
        "horizonte_minutos": FORECAST_MINUTES,
        "limiar_s4": S4_THRESHOLD,
        "observacoes": len(dataset),
        "classe_0": int((dataset["classe_future"] == 0).sum()),
        "classe_1": int((dataset["classe_future"] == 1).sum()),
        "features": list(CLASSIFICATION_FEATURES),
    }
    with open(ARQUIVO_RESUMO, "w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)
    print("Dataset salvo:", ARQUIVO_SAIDA)
    print("Dimensão:", dataset.shape)


if __name__ == "__main__":
    main()
