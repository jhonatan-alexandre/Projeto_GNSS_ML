"""Extrai e padroniza as variáveis científicas dos arquivos ISMR.

O ISMR do Septentrio fornece S4 total e a contribuição de ruído térmico em
s4_correction. Quando configurado, o S4 principal é calculado por
sqrt(max(S4_total² - S4_correction², 0)).

O campo tec é tratado como TEC oblíquo (STEC). O VTEC é obtido com o
modelo de camada fina usando a elevação do satélite.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)

from config import (  # noqa: E402
    CSV_CHUNK_SIZE,
    EARTH_RADIUS_KM,
    IONOSPHERE_HEIGHT_KM,
    PROCESSED_DATA,
    STATION,
    STUDY_START,
    TEST_END,
    USE_CORRECTED_S4,
)


PASTA_ESTACAO = os.path.join(PROCESSED_DATA, STATION)
LISTA_ARQUIVOS = os.path.join(PASTA_ESTACAO, "lista_arquivos_ismr.txt")
ARQUIVO_SAIDA = os.path.join(PASTA_ESTACAO, f"{STATION}_raw.csv")


ALIASES_TEMPO = ("time_utc", "datetime_utc", "datetime", "timestamp", "utc_time")
GRUPOS_DATA = (
    ("year", "month", "day", "hour", "minute", "second"),
    ("ano", "mes", "dia", "hora", "minuto", "segundo"),
)
ALIASES = {
    "svid": ("svid", "prn", "sv", "sat"),
    "elev": ("elev", "elevation", "elevation_deg", "ele"),
    "azim": ("azim", "azimuth", "azimuth_deg", "az"),
    "s4_total": ("s4", "s4_l1", "sigma4", "s4_total"),
    "s4_correction": (
        "s4_correction",
        "s4_correction_l1",
        "s4_noise",
    ),
    "vtec": ("vtec", "vtec_tecu"),
}
TEC_PRIORITY = ("tec", "stec", "slant_tec", "tec_tecu")


def normalizar_nome_coluna(nome: object) -> str:
    """Normaliza um cabeçalho sem alterar os dados."""

    return str(nome).strip().lstrip("\ufeff").lower()


COLUNAS_RELEVANTES = {
    *ALIASES_TEMPO,
    "wn",
    "tow",
    *(nome for grupo in GRUPOS_DATA for nome in grupo),
    *(nome for alternativas in ALIASES.values() for nome in alternativas),
    *TEC_PRIORITY,
}


def carregar_lista_arquivos() -> list[str]:
    if not os.path.exists(LISTA_ARQUIVOS):
        raise FileNotFoundError(
            "Arquivo lista_arquivos_ismr.txt não encontrado. Execute primeiro o script 02."
        )

    with open(LISTA_ARQUIVOS, "r", encoding="utf-8") as arquivo:
        lista = list(dict.fromkeys(linha.strip() for linha in arquivo if linha.strip()))

    if not lista:
        raise ValueError("A lista de arquivos ISMR está vazia.")
    return lista


def mapa_colunas(df: pd.DataFrame) -> dict[str, str]:
    """Retorna nome normalizado -> nome original, ignorando colunas vazias."""

    resultado: dict[str, str] = {}
    for coluna in df.columns:
        normalizada = normalizar_nome_coluna(coluna)
        if normalizada and not normalizada.startswith("unnamed:"):
            resultado.setdefault(normalizada, coluna)
    return resultado


def encontrar_coluna(
    colunas: dict[str, str], alternativas: tuple[str, ...]
) -> str | None:
    for alternativa in alternativas:
        encontrada = colunas.get(normalizar_nome_coluna(alternativa))
        if encontrada is not None:
            return encontrada
    return None


def _datas_por_componentes(
    df: pd.DataFrame, colunas: dict[str, str], grupo: tuple[str, ...]
) -> pd.Series:
    originais = [colunas.get(nome) for nome in grupo]
    if not all(originais):
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    return pd.to_datetime(
        {
            "year": df[originais[0]],
            "month": df[originais[1]],
            "day": df[originais[2]],
            "hour": df[originais[3]],
            "minute": df[originais[4]],
            "second": df[originais[5]],
        },
        errors="coerce",
    )


def _datas_por_wn_tow(df: pd.DataFrame, colunas: dict[str, str]) -> pd.Series:
    """Converte WN/TOW preservando a escala de tempo GPS do ISMR."""

    coluna_wn = colunas.get("wn")
    coluna_tow = colunas.get("tow")
    resultado = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if coluna_wn is None or coluna_tow is None:
        return resultado

    wn = pd.to_numeric(df[coluna_wn], errors="coerce")
    tow = pd.to_numeric(df[coluna_tow], errors="coerce")
    validas = (
        wn.notna()
        & tow.notna()
        & np.isfinite(wn)
        & np.isfinite(tow)
        & wn.ge(0)
        & wn.eq(np.floor(wn))
        & tow.ge(0)
        & tow.lt(604800)
    )
    resultado.loc[validas] = (
        pd.Timestamp("1980-01-06")
        + pd.to_timedelta(wn.loc[validas] * 7, unit="D")
        + pd.to_timedelta(tow.loc[validas], unit="s")
    )
    return resultado


def criar_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Cria datetime, preenchendo falhas com as fontes temporais seguintes."""

    colunas = mapa_colunas(df)
    datas = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    for alias in ALIASES_TEMPO:
        coluna = colunas.get(alias)
        if coluna is not None:
            candidato = pd.to_datetime(df[coluna], errors="coerce")
            datas = datas.combine_first(candidato)

    for grupo in GRUPOS_DATA:
        datas = datas.combine_first(_datas_por_componentes(df, colunas, grupo))

    datas_gps = _datas_por_wn_tow(df, colunas)
    sobreposicao = datas.notna() & datas_gps.notna()
    if sobreposicao.any():
        divergencia = (datas.loc[sobreposicao] - datas_gps.loc[sobreposicao]).abs()
        if divergencia.gt(pd.Timedelta(seconds=1)).any():
            raise ValueError("As datas textuais e WN/TOW são incompatíveis.")
    datas = datas.combine_first(datas_gps)

    if not datas.notna().any():
        raise ValueError("Não foram encontradas informações temporais válidas no ISMR.")

    df = df.copy()
    df["datetime"] = datas
    return df


def fator_mapeamento_vtec(elevacao_graus: pd.Series) -> pd.Series:
    """Calcula o fator do modelo ionosférico de camada fina."""

    elevacao = pd.to_numeric(elevacao_graus, errors="coerce")
    resultado = pd.Series(np.nan, index=elevacao.index, dtype="float64")
    validas = elevacao.notna() & np.isfinite(elevacao) & elevacao.between(0, 90)
    elevacao_rad = np.deg2rad(elevacao.loc[validas])
    razao = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + IONOSPHERE_HEIGHT_KM)
    termo = 1.0 - np.square(razao * np.cos(elevacao_rad))
    resultado.loc[validas] = np.sqrt(np.clip(termo, 0.0, 1.0))
    return resultado


def selecionar_tec(
    df: pd.DataFrame, colunas: dict[str, str]
) -> tuple[pd.Series, pd.Series]:
    """Seleciona o primeiro STEC contemporâneo disponível em cada linha."""

    stec = pd.Series(np.nan, index=df.index, dtype="float64")
    origem = pd.Series(pd.NA, index=df.index, dtype="string")
    for alias in TEC_PRIORITY:
        coluna = colunas.get(alias)
        if coluna is None:
            continue
        candidato = pd.to_numeric(df[coluna], errors="coerce")
        mascara = stec.isna() & candidato.notna()
        stec.loc[mascara] = candidato.loc[mascara]
        origem.loc[mascara] = alias
    return stec, origem


def selecionar_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas, corrige S4 e calcula VTEC."""

    if df.empty:
        raise ValueError("O arquivo ISMR não contém observações.")

    colunas_validas = ~df.columns.astype(str).str.match(r"^Unnamed:", case=False)
    df = df.loc[:, colunas_validas].copy()
    df = criar_datetime(df)
    colunas = mapa_colunas(df)

    identificadas = {
        nome: encontrar_coluna(colunas, alternativas)
        for nome, alternativas in ALIASES.items()
    }
    faltantes = [
        nome
        for nome in ("svid", "elev", "s4_total")
        if identificadas[nome] is None
    ]
    if faltantes:
        raise ValueError(
            "Variáveis obrigatórias não encontradas: " + ", ".join(faltantes)
        )

    saida = pd.DataFrame(index=df.index)
    saida["datetime"] = df["datetime"]
    saida["svid"] = pd.to_numeric(df[identificadas["svid"]], errors="coerce")
    saida["elev"] = pd.to_numeric(df[identificadas["elev"]], errors="coerce")

    for nome in ("azim",):
        coluna = identificadas[nome]
        saida[nome] = (
            pd.to_numeric(df[coluna], errors="coerce")
            if coluna is not None
            else np.nan
        )

    s4_total = pd.to_numeric(df[identificadas["s4_total"]], errors="coerce")
    coluna_correcao = identificadas["s4_correction"]
    s4_correcao = (
        pd.to_numeric(df[coluna_correcao], errors="coerce")
        if coluna_correcao is not None
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    if USE_CORRECTED_S4 and coluna_correcao is not None:
        quadrado_corrigido = np.maximum(
            np.square(s4_total) - np.square(s4_correcao), 0.0
        )
        s4_corrigido = pd.Series(np.sqrt(quadrado_corrigido), index=df.index)
        saida["S4"] = s4_corrigido.where(s4_correcao.notna(), s4_total)
    else:
        saida["S4"] = s4_total

    stec, _ = selecionar_tec(df, colunas)

    coluna_vtec = identificadas["vtec"]
    vtec_direto = (
        pd.to_numeric(df[coluna_vtec], errors="coerce")
        if coluna_vtec is not None
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    vtec_calculado = stec * fator_mapeamento_vtec(saida["elev"])
    saida["VTEC"] = vtec_direto.fillna(vtec_calculado)

    # O Estudo Dirigido usa somente S4, VTEC e geometria. STEC permanece
    # apenas como grandeza intermediária para calcular VTEC quando necessário.
    saida = saida[["datetime", "svid", "elev", "azim", "S4", "VTEC"]]
    inicio = pd.Timestamp(STUDY_START)
    fim = pd.Timestamp(TEST_END)
    return saida.loc[saida["datetime"].between(inicio, fim, inclusive="both")]


def iterar_ismr(
    caminho: str, tamanho_bloco: int = CSV_CHUNK_SIZE
):
    print("\nLendo:", os.path.basename(caminho))
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo ISMR não encontrado: {caminho}")

    with pd.read_csv(
        caminho,
        comment="#",
        chunksize=tamanho_bloco,
        low_memory=True,
        usecols=lambda coluna: normalizar_nome_coluna(coluna) in COLUNAS_RELEVANTES,
    ) as leitor:
        for bloco in leitor:
            selecionado = selecionar_variaveis(bloco)
            if not selecionado.empty:
                yield selecionado


def ler_ismr(caminho: str) -> pd.DataFrame:
    """Compatibilidade para arquivos pequenos; a execução principal é incremental."""

    blocos = list(iterar_ismr(caminho))
    if not blocos:
        return pd.DataFrame(columns=["datetime", "svid", "elev", "azim", "S4", "VTEC"])
    return pd.concat(blocos, ignore_index=True)


def processar(arquivos: list[str] | None = None) -> pd.DataFrame:
    arquivos = carregar_lista_arquivos() if arquivos is None else arquivos
    print("\nQuantidade de arquivos:", len(arquivos))

    quadros: list[pd.DataFrame] = []
    for arquivo in arquivos:
        quadros.extend(iterar_ismr(arquivo))

    if not quadros:
        raise RuntimeError("Nenhuma observação do período do estudo foi processada.")

    dados = pd.concat(quadros, ignore_index=True)
    dados = dados.sort_values(["datetime", "svid"], kind="stable").reset_index(drop=True)

    print(f"Blocos processados: {len(quadros)}")
    return dados


def processar_para_csv(
    arquivos: list[str] | None = None,
    caminho_saida: str = ARQUIVO_SAIDA,
) -> int:
    """Extrai em blocos e substitui a saída apenas após sucesso completo."""

    arquivos = carregar_lista_arquivos() if arquivos is None else arquivos
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    descritor, temporario = tempfile.mkstemp(
        prefix=".raw_", suffix=".tmp", dir=os.path.dirname(caminho_saida)
    )
    os.close(descritor)
    escreveu = False
    total = 0
    try:
        for caminho in arquivos:
            for bloco in iterar_ismr(caminho):
                bloco = bloco.sort_values(["datetime", "svid"], kind="stable")
                bloco.to_csv(
                    temporario,
                    mode="a" if escreveu else "w",
                    header=not escreveu,
                    index=False,
                    date_format="%Y-%m-%d %H:%M:%S",
                )
                escreveu = True
                total += len(bloco)
        if not escreveu:
            raise RuntimeError("Nenhum dado do recorte do estudo foi encontrado.")
        dados = pd.read_csv(temporario, parse_dates=["datetime"])
        dados = dados.sort_values(["datetime", "svid"], kind="stable")
        dados.to_csv(
            temporario,
            mode="w",
            header=True,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        os.replace(temporario, caminho_saida)
    except Exception:
        if os.path.exists(temporario):
            os.remove(temporario)
        raise
    return total


def main() -> None:
    print("=" * 70)
    print("EXTRAÇÃO ISMR COM PRESERVAÇÃO TEMPORAL")
    print("ESTAÇÃO:", STATION)
    print("=" * 70)

    total = processar_para_csv()
    print("\nObservações extraídas:", total)
    print("\nArquivo salvo:", ARQUIVO_SAIDA)


if __name__ == "__main__":
    main()
