"""Aplica o filtro probabilístico do Núcleo 1 à série de VTEC."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)

from config import (  # noqa: E402
    PROCESSED_DATA,
    RESULTS_DIRECTORY,
    SPACE_WEATHER_COLUMNS,
    STATION,
    TEST_START,
    TEST_END,
    TIME_RESOLUTION,
    LOSS_MASKING,
    ALLOW_PARTIAL_TEST_OBSERVATIONS,
    MINIMUM_TEST_OBSERVATIONS,
)


PASTA_ESTACAO = os.path.join(PROCESSED_DATA, STATION)
FEATURES = os.path.join(PASTA_ESTACAO, f"{STATION}_features.csv")
PREDICOES = os.path.join(RESULTS_DIRECTORY, STATION, "nucleo1", "predicoes_xgboost.csv")
SAIDA = os.path.join(PASTA_ESTACAO, f"{STATION}_dataset_regressao.csv")
RESUMO = os.path.join(PASTA_ESTACAO, "resumo_filtragem_vtec.json")
from interpolacao_vtec import interpolar_por_periodo


def validar_vtec_observado_teste(
    features: pd.DataFrame, inicio: str, fim: str, resolucao: str = TIME_RESOLUTION
) -> int:
    """Exige medições no teste antes de interpolar as séries de treinamento."""
    grade = pd.date_range(
        pd.Timestamp(inicio).ceil(resolucao), pd.Timestamp(fim).floor(resolucao),
        freq=resolucao,
    )
    if grade.empty:
        raise ValueError("O intervalo de teste não contém épocas na resolução configurada.")
    valores = pd.to_numeric(features["VTEC"], errors="coerce")
    validos = pd.DataFrame({
        "datetime": pd.to_datetime(features["datetime"], errors="coerce"),
        "observado": np.isfinite(valores),
    }).groupby("datetime")["observado"].any().reindex(grade, fill_value=False)
    quantidade = int(validos.sum())
    ausentes = validos.index[~validos]
    if quantidade < MINIMUM_TEST_OBSERVATIONS:
        raise ValueError(
            f"Teste tem apenas {quantidade} épocas com VTEC observado; mínimo: "
            f"{MINIMUM_TEST_OBSERVATIONS}."
        )
    if len(ausentes) and not ALLOW_PARTIAL_TEST_OBSERVATIONS:
        raise ValueError(
            f"Teste sem VTEC observado em {len(ausentes)} de {len(grade)} épocas "
            f"(primeira: {ausentes.min()}; última: {ausentes.max()}). "
            "Escolha um teste com cobertura observada; a interpolação não substitui "
            "as medições de referência."
        )
    return quantidade


def construir_series(
    features: pd.DataFrame, predicoes: pd.DataFrame,
    inicio_teste: str | None = TEST_START,
) -> pd.DataFrame:
    features = features.copy()
    predicoes = predicoes.copy()
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
    predicoes["target_datetime"] = pd.to_datetime(
        predicoes["target_datetime"], errors="coerce"
    )
    sinais = predicoes[["target_datetime", "svid", "sinalizar_filtro"]].rename(
        columns={"target_datetime": "datetime"}
    )
    dados = features.merge(sinais, on=["datetime", "svid"], how="left")
    dados["sinalizar_filtro"] = (
        dados["sinalizar_filtro"].astype("boolean").fillna(False).astype(bool)
    )
    dados["svid_observado"] = dados["svid"].where(dados["VTEC"].notna())
    dados["foi_filtrado"] = dados["sinalizar_filtro"] & dados["VTEC"].notna()
    dados["VTEC_filtered"] = dados["VTEC"].mask(dados["sinalizar_filtro"])

    agregacoes = {
        "VTEC": "median",
        "VTEC_filtered": "median",
        "svid_observado": "nunique",
        "foi_filtrado": "sum",
        **{coluna: "first" for coluna in SPACE_WEATHER_COLUMNS},
    }
    serie = (
        dados.groupby("datetime", as_index=False)
        .agg(agregacoes)
        .rename(
            columns={
                "VTEC": "VTEC_original",
                "svid_observado": "n_svid_observados",
                "foi_filtrado": "n_svid_filtrados",
            }
        )
        .sort_values("datetime")
        .set_index("datetime")
        .asfreq(TIME_RESOLUTION)
    )
    inicio, fim = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    serie["observado_teste"] = (
        serie.index.to_series().between(inicio, fim)
        & serie["VTEC_original"].notna()
    )
    if LOSS_MASKING:
        # Preserva a série observada; a decisão do Núcleo 1 vira peso da loss.
        serie["VTEC_filtered"] = serie["VTEC_original"]
        serie["loss_weight"] = np.where(
            serie["VTEC_original"].notna() & serie["n_svid_filtrados"].eq(0), 1.0, 0.0
        )
    else:
        for coluna in ("VTEC_original", "VTEC_filtered"):
            serie[coluna] = interpolar_por_periodo(serie[coluna], inicio_teste)
        serie["loss_weight"] = serie["VTEC_original"].notna().astype(float)
    for coluna in SPACE_WEATHER_COLUMNS:
        serie[coluna] = serie[coluna].ffill().bfill()
    for coluna in ("n_svid_observados", "n_svid_filtrados"):
        serie[coluna] = serie[coluna].fillna(0).astype("int64")
    if not LOSS_MASKING and serie[["VTEC_original", "VTEC_filtered"]].isna().any().any():
        raise ValueError(
            "A série de VTEC permaneceu sem valores após a interpolação linear."
        )
    return serie.reset_index()


def main() -> None:
    for caminho in (FEATURES, PREDICOES):
        if not os.path.isfile(caminho):
            raise FileNotFoundError(f"Entrada ausente: {caminho}")
    features = pd.read_csv(FEATURES, low_memory=True)
    predicoes = pd.read_csv(PREDICOES, low_memory=True)
    epocas_teste = validar_vtec_observado_teste(features, TEST_START, TEST_END)
    print(f"Teste validado: {epocas_teste} épocas com VTEC observado antes da interpolação.")
    dados = construir_series(features, predicoes)
    dados.to_csv(SAIDA, index=False, date_format="%Y-%m-%d %H:%M:%S")
    observados = int(dados["n_svid_observados"].sum())
    filtrados = int(dados["n_svid_filtrados"].sum())
    resumo = {
        "estacao": STATION,
        "epocas": len(dados),
        "svid_epocas_observados": observados,
        "svid_epocas_filtrados": filtrados,
        "percentual_filtrado": 100 * filtrados / observados if observados else 0.0,
        "epocas_com_filtro": int(dados["n_svid_filtrados"].gt(0).sum()),
        "epocas_vtec_original_ausente": int(dados["VTEC_original"].isna().sum()),
        "epocas_vtec_filtrado_ausente": int(dados["VTEC_filtered"].isna().sum()),
        "interpolacao_anterior_ao_teste_usa_teste": False,
        "loss_masking": LOSS_MASKING,
    }
    with open(RESUMO, "w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)
    print("Dataset de regressão salvo:", SAIDA)
    print("Dimensão:", dados.shape)


if __name__ == "__main__":
    main()
