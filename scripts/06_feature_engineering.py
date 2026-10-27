"""Reamostra o ISMR e cria os atributos definidos no Estudo Dirigido."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)

from config import (  # noqa: E402
    CSV_CHUNK_SIZE,
    DAILY_LAG_STEPS,
    PROCESSED_DATA,
    ROLLING_WINDOW_STEPS,
    SPACE_WEATHER_COLUMNS,
    STATION,
    STUDY_START,
    TEST_END,
    TIME_RESOLUTION,
)


PASTA_ESTACAO = os.path.join(PROCESSED_DATA, STATION)
ARQUIVO_ENTRADA = os.path.join(PASTA_ESTACAO, f"{STATION}_clean.csv")
ARQUIVO_CLIMA = os.path.join(PROCESSED_DATA, "clima_espacial_30min.csv")
ARQUIVO_SAIDA = os.path.join(PASTA_ESTACAO, f"{STATION}_features.csv")
COLUNAS_ISMR = ("datetime", "svid", "elev", "azim", "S4", "VTEC")


def _agregar_bloco(bloco: pd.DataFrame) -> pd.DataFrame:
    for coluna in COLUNAS_ISMR[1:]:
        bloco[coluna] = pd.to_numeric(bloco[coluna], errors="coerce")
    bloco["datetime"] = pd.to_datetime(bloco["datetime"], errors="coerce")
    bloco = bloco.loc[
        bloco["datetime"].between(
            pd.Timestamp(STUDY_START), pd.Timestamp(TEST_END), inclusive="both"
        )
    ].copy()
    if bloco.empty:
        return pd.DataFrame()

    bloco["datetime"] = bloco["datetime"].dt.floor(TIME_RESOLUTION)
    azim_rad = np.deg2rad(bloco["azim"])
    bloco["azim_sin"] = np.sin(azim_rad)
    bloco["azim_cos"] = np.cos(azim_rad)
    return bloco.groupby(["datetime", "svid"], as_index=False).agg(
        elev_sum=("elev", "sum"),
        elev_count=("elev", "count"),
        azim_sin_sum=("azim_sin", "sum"),
        azim_sin_count=("azim_sin", "count"),
        azim_cos_sum=("azim_cos", "sum"),
        azim_cos_count=("azim_cos", "count"),
        S4_sum=("S4", "sum"),
        S4_count=("S4", "count"),
        VTEC_sum=("VTEC", "sum"),
        VTEC_count=("VTEC", "count"),
    )


def reamostrar_ismr(
    caminho: str = ARQUIVO_ENTRADA,
    tamanho_bloco: int = CSV_CHUNK_SIZE,
) -> pd.DataFrame:
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Execute primeiro o script 04: {caminho}")

    partes: list[pd.DataFrame] = []
    with pd.read_csv(
        caminho,
        usecols=list(COLUNAS_ISMR),
        chunksize=tamanho_bloco,
        low_memory=True,
    ) as leitor:
        for bloco in leitor:
            agregado = _agregar_bloco(bloco)
            if not agregado.empty:
                partes.append(agregado)
    if not partes:
        raise ValueError("Não há dados ISMR no período do estudo.")

    somas = pd.concat(partes, ignore_index=True).groupby(
        ["datetime", "svid"], as_index=False
    ).sum(numeric_only=True)
    for nome in ("elev", "azim_sin", "azim_cos", "S4", "VTEC"):
        somas[nome] = somas[f"{nome}_sum"].div(
            somas[f"{nome}_count"].replace(0, np.nan)
        )
    return somas[["datetime", "svid", "elev", "azim_sin", "azim_cos", "S4", "VTEC"]]


def completar_grade_temporal(df: pd.DataFrame) -> pd.DataFrame:
    inicio = pd.Timestamp(STUDY_START).floor(TIME_RESOLUTION)
    fim = pd.Timestamp(TEST_END).floor(TIME_RESOLUTION)
    grade = pd.date_range(inicio, fim, freq=TIME_RESOLUTION)
    grupos: list[pd.DataFrame] = []
    for svid, grupo in df.groupby("svid", sort=True):
        completo = grupo.set_index("datetime").reindex(grade)
        completo.index.name = "datetime"
        completo["svid"] = svid
        grupos.append(completo.reset_index())
    return pd.concat(grupos, ignore_index=True)


def criar_features(df: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    df = completar_grade_temporal(df).sort_values(["svid", "datetime"])
    segundos = (
        df["datetime"].dt.hour * 3600
        + df["datetime"].dt.minute * 60
        + df["datetime"].dt.second
    )
    fase_hora = 2 * np.pi * segundos / 86400
    fase_mes = 2 * np.pi * (df["datetime"].dt.month - 1) / 12
    df["hour_sin"] = np.sin(fase_hora)
    df["hour_cos"] = np.cos(fase_hora)
    df["month_sin"] = np.sin(fase_mes)
    df["month_cos"] = np.cos(fase_mes)

    grupo_s4 = df.groupby("svid", sort=False)["S4"]
    df["S4_mean_3h"] = grupo_s4.transform(
        lambda serie: serie.rolling(ROLLING_WINDOW_STEPS, min_periods=3).mean()
    )
    df["S4_std_3h"] = grupo_s4.transform(
        lambda serie: serie.rolling(ROLLING_WINDOW_STEPS, min_periods=3).std()
    )
    df["S4_lag_24h"] = grupo_s4.shift(DAILY_LAG_STEPS)

    clima = clima.copy()
    clima["datetime"] = pd.to_datetime(clima["datetime"], errors="coerce")
    faltantes = [c for c in SPACE_WEATHER_COLUMNS if c not in clima.columns]
    if faltantes:
        raise ValueError("Clima espacial sem: " + ", ".join(faltantes))
    return pd.merge_asof(
        df.sort_values("datetime"),
        clima[["datetime", *SPACE_WEATHER_COLUMNS]].sort_values("datetime"),
        on="datetime",
        direction="backward",
        tolerance=pd.Timedelta("24h"),
    ).sort_values(["datetime", "svid"], kind="stable").reset_index(drop=True)


def main() -> None:
    if not os.path.isfile(ARQUIVO_CLIMA):
        raise FileNotFoundError(f"Execute primeiro o script 05: {ARQUIVO_CLIMA}")
    print("Reamostrando ISMR para", TIME_RESOLUTION)
    ismr = reamostrar_ismr()
    clima = pd.read_csv(ARQUIVO_CLIMA, low_memory=True)
    dados = criar_features(ismr, clima)
    dados.to_csv(ARQUIVO_SAIDA, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print("Arquivo salvo:", ARQUIVO_SAIDA)
    print("Dimensão:", dados.shape)


if __name__ == "__main__":
    main()
