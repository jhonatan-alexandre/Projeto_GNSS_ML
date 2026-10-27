"""Interpolação do VTEC com separação entre histórico e teste."""
from __future__ import annotations

import pandas as pd


def interpolar_por_periodo(serie: pd.Series, inicio_teste: str | None) -> pd.Series:
    """Impede que medições do teste preencham lacunas anteriores ao teste."""
    if inicio_teste is None:
        return serie.interpolate(method="time", limit_direction="both")
    anterior = serie.index < pd.Timestamp(inicio_teste)
    return pd.concat([
        serie.loc[mascara].interpolate(method="time", limit_direction="both")
        for mascara in (anterior, ~anterior)
    ]).sort_index()
