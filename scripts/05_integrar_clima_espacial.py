"""Padroniza Kp, Dst, F10.7, Ap e SSN exportados do NASA OmniWeb."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)

from config import (  # noqa: E402
    PROCESSED_DATA,
    SPACE_WEATHER_COLUMNS,
    SPACE_WEATHER_FILES,
    STUDY_START,
    TEST_END,
    TIME_RESOLUTION,
)


ARQUIVO_SAIDA = os.path.join(PROCESSED_DATA, "clima_espacial_30min.csv")
ALIASES = {
    "datetime": ("datetime", "time", "time_utc", "timestamp", "date"),
    "Kp": ("kp", "kp_index"),
    "Dst": ("dst", "dst_index", "dst_nt"),
    "F107": ("f107", "f10.7", "f10_7", "flux_f107", "f107_obs"),
    "Ap": ("ap", "ap_index"),
    "SSN": ("ssn", "sunspot_number", "sunspot", "r"),
}
CODIGOS_AUSENTES = {
    "Kp": (99.0, 999.0, 9999.0),
    "Dst": (9999.0, 99999.0),
    "F107": (999.9, 9999.0),
    "Ap": (999.0, 9999.0),
    "SSN": (999.0, 9999.0),
}

# Formato sem cabeçalho já existente neste projeto:
# ano, dia do ano, hora, Kp (décimos), SSN, Dst, Ap e F10.7.
COLUNAS_LEGADO = ("year", "doy", "hour", "Kp", "SSN", "Dst", "Ap", "F107")


def _normalizar(nome: object) -> str:
    return str(nome).strip().lstrip("\ufeff").lower().replace(" ", "_")


def _encontrar(colunas: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    return next((colunas.get(_normalizar(alias)) for alias in aliases if colunas.get(_normalizar(alias))), None)


def _criar_datetime(df: pd.DataFrame, colunas: dict[str, str]) -> pd.Series:
    direta = _encontrar(colunas, ALIASES["datetime"])
    if direta is not None:
        return pd.to_datetime(df[direta], errors="coerce", utc=True).dt.tz_localize(None)

    year = colunas.get("year") or colunas.get("yyyy")
    doy = colunas.get("doy") or colunas.get("day_of_year")
    hour = colunas.get("hour") or colunas.get("hr")
    if year is None or doy is None:
        raise ValueError("OmniWeb precisa de datetime ou das colunas year/doy.")
    anos = pd.to_numeric(df[year], errors="coerce")
    dias = pd.to_numeric(df[doy], errors="coerce")
    horas = pd.to_numeric(df[hour], errors="coerce") if hour else 0
    return (
        pd.to_datetime(anos.astype("Int64").astype("string"), format="%Y", errors="coerce")
        + pd.to_timedelta(dias - 1, unit="D")
        + pd.to_timedelta(horas, unit="h")
    )


def padronizar_clima_espacial(df: pd.DataFrame) -> pd.DataFrame:
    colunas = {_normalizar(coluna): coluna for coluna in df.columns}
    saida = pd.DataFrame({"datetime": _criar_datetime(df, colunas)})
    ausentes: list[str] = []
    for nome in SPACE_WEATHER_COLUMNS:
        original = _encontrar(colunas, ALIASES[nome])
        if original is None:
            ausentes.append(nome)
            continue
        valores = pd.to_numeric(df[original], errors="coerce")
        ausente = ~np.isfinite(valores) | valores.isin(CODIGOS_AUSENTES[nome])
        valores = valores.mask(ausente)
        # Algumas exportações OMNI fornecem Kp em décimos (0–90).
        if nome == "Kp" and valores.max(skipna=True) > 9:
            valores = valores / 10.0
        saida[nome] = valores
    if ausentes:
        raise ValueError("Variáveis OmniWeb ausentes: " + ", ".join(ausentes))

    saida = saida.dropna(subset=["datetime"]).sort_values("datetime")
    saida = saida.loc[
        saida["datetime"].between(
            pd.Timestamp(STUDY_START), pd.Timestamp(TEST_END), inclusive="both"
        )
    ]
    if saida.empty:
        raise ValueError("O arquivo OmniWeb não cobre o período do estudo.")

    reamostrado = (
        saida.set_index("datetime")
        .resample(TIME_RESOLUTION)
        .mean()
    )
    grade = pd.date_range(
        pd.Timestamp(STUDY_START).floor(TIME_RESOLUTION),
        pd.Timestamp(TEST_END).floor(TIME_RESOLUTION),
        freq=TIME_RESOLUTION,
        name="datetime",
    )
    # A fonte é horária e a saída é de 30 min: um único preenchimento é o
    # bastante para a meia hora intermediária. Limitar o preenchimento impede
    # que semanas ou meses ausentes recebam silenciosamente o último valor.
    reamostrado = reamostrado.reindex(grade).ffill(limit=1)
    linhas_ausentes = reamostrado[list(SPACE_WEATHER_COLUMNS)].isna().any(axis=1)
    if linhas_ausentes.any():
        datas_ausentes = reamostrado.index[linhas_ausentes]
        raise ValueError(
            "O clima espacial não cobre integralmente o período do estudo: "
            f"{len(datas_ausentes)} épocas ausentes entre "
            f"{datas_ausentes.min()} e {datas_ausentes.max()}."
        )
    return reamostrado.reset_index()


def _campo_numerico(valor: str) -> bool:
    try:
        float(valor)
    except ValueError:
        return False
    return True


def _ler_formato_legado(caminho: str | os.PathLike[str]) -> pd.DataFrame:
    registros: list[list[str]] = []
    with open(caminho, "r", encoding="utf-8-sig") as arquivo:
        tokens = arquivo.read().split()

    # Algumas cópias legadas perderam quebras de linha. O ano, presente no
    # início de toda observação, permite reconstruir os registros sem inferir
    # ou modificar nenhum valor científico.
    atual: list[str] = []
    for token in tokens:
        inicio_registro = token.isdigit() and 1900 <= int(token) <= 2200
        if inicio_registro:
            if atual:
                registros.append(atual)
            atual = [token]
        elif atual:
            atual.append(token)
    if atual:
        registros.append(atual)
    if not registros:
        raise ValueError(f"O arquivo de clima espacial está vazio: {caminho}")

    linhas: list[list[str]] = []
    for numero_registro, campos in enumerate(registros, start=1):
        # Corrige somente espaços acidentais dentro do último valor F10.7,
        # preservando sem alteração os sete campos científicos anteriores.
        if len(campos) > len(COLUNAS_LEGADO):
            campos = [*campos[:7], "".join(campos[7:])]
        if len(campos) != len(COLUNAS_LEGADO):
            raise ValueError(
                f"Registro {numero_registro} do clima legado tem {len(campos)} "
                f"campos; eram esperados {len(COLUNAS_LEGADO)}."
            )
        linhas.append(campos)

    dados = pd.DataFrame(linhas, columns=COLUNAS_LEGADO)
    for coluna in COLUNAS_LEGADO:
        dados[coluna] = pd.to_numeric(
            dados[coluna]
            .str.replace(r"[^0-9,\.\-+eE]", "", regex=True)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )
    return dados


def carregar_clima_espacial(caminho: str | os.PathLike[str]) -> pd.DataFrame:
    """Lê tanto exportações CSV do OmniWeb quanto o TXT legado do projeto."""

    with open(caminho, "r", encoding="utf-8-sig") as arquivo:
        primeira_linha = arquivo.readline().strip()
    if not primeira_linha:
        raise ValueError(f"O arquivo de clima espacial está vazio: {caminho}")

    campos = primeira_linha.replace(",", ".").split()
    legado_sem_cabecalho = (
        len(campos) >= len(COLUNAS_LEGADO)
        and all(_campo_numerico(campo) for campo in campos)
        and 1900 <= int(float(campos[0])) <= 2200
    )
    if legado_sem_cabecalho:
        return _ler_formato_legado(caminho)
    return pd.read_csv(caminho, sep=None, engine="python", comment="#")


def main() -> None:
    if not SPACE_WEATHER_FILES:
        grade = pd.date_range(
            pd.Timestamp(STUDY_START).floor(TIME_RESOLUTION),
            pd.Timestamp(TEST_END).floor(TIME_RESOLUTION),
            freq=TIME_RESOLUTION,
            name="datetime",
        )
        pd.DataFrame({"datetime": grade}).to_csv(
            ARQUIVO_SAIDA, index=False, date_format="%Y-%m-%d %H:%M:%S"
        )
        print("Clima espacial desativado; grade temporal salva:", ARQUIVO_SAIDA)
        return
    ausentes = [caminho for caminho in SPACE_WEATHER_FILES if not os.path.isfile(caminho)]
    if ausentes:
        raise FileNotFoundError(
            "Arquivos de clima espacial não encontrados: "
            + ", ".join(map(str, ausentes))
            + ". "
            "Forneça Kp, Dst, F10.7, Ap e SSN em CSV."
        )
    quadros = []
    for caminho in SPACE_WEATHER_FILES:
        print("Lendo clima espacial:", caminho)
        quadros.append(carregar_clima_espacial(caminho))
    bruto = pd.concat(quadros, ignore_index=True)
    dados = padronizar_clima_espacial(bruto)
    dados.to_csv(ARQUIVO_SAIDA, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print("Arquivo salvo:", ARQUIVO_SAIDA)
    print("Dimensão:", dados.shape)


if __name__ == "__main__":
    main()
