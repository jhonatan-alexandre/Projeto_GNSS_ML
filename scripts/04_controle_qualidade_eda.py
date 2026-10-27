"""Controle de qualidade e análise exploratória da base ISMR extraída.

O arquivo bruto pode conter dezenas de milhões de registros. Por isso, a
execução principal faz a leitura, a limpeza e a EDA em blocos; o resumo exato
lê no máximo uma coluna completa por vez.
"""

from __future__ import annotations

from collections import Counter
import os
import tempfile
from typing import Any, Iterator

_MPLCONFIGDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "matplotlib"
)
os.makedirs(_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import (
    CSV_CHUNK_SIZE,
    MAX_ELEVATION,
    MAX_S4,
    MAX_VTEC,
    MIN_ELEVATION,
    MIN_S4,
    MIN_VTEC,
    PROCESSED_DATA,
    S4_THRESHOLD,
    STATION,
    TIME_RESOLUTION,
)


PASTA_ESTACAO = os.path.join(PROCESSED_DATA, STATION)
ARQUIVO_RAW = os.path.join(PASTA_ESTACAO, f"{STATION}_raw.csv")
ARQUIVO_CLEAN = os.path.join(PASTA_ESTACAO, f"{STATION}_clean.csv")
PASTA_EDA = os.path.join(PASTA_ESTACAO, "EDA")

COLUNAS_ESTUDO = ("datetime", "svid", "elev", "azim", "S4", "VTEC")
COLUNAS_OBRIGATORIAS = COLUNAS_ESTUDO
COLUNAS_NUMERICAS = (
    "svid",
    "elev",
    "azim",
    "S4",
    "VTEC",
)
ORDEM_ESTATISTICAS = (
    "count",
    "unique",
    "top",
    "freq",
    "mean",
    "min",
    "25%",
    "50%",
    "75%",
    "max",
    "std",
)
LIMITE_SUPERIOR_SVID = 1 << 63


def _validar_csv(caminho: str) -> list[str]:
    if not os.path.isfile(caminho):
        raise FileNotFoundError(
            f"Arquivo bruto não encontrado: {caminho}. Execute primeiro o script 03."
        )

    try:
        colunas = list(pd.read_csv(caminho, nrows=0).columns)
    except pd.errors.EmptyDataError as erro:
        raise ValueError("O arquivo bruto está vazio.") from erro

    faltantes = [coluna for coluna in COLUNAS_OBRIGATORIAS if coluna not in colunas]
    if faltantes:
        raise ValueError(
            "Colunas obrigatórias ausentes no arquivo bruto: " + ", ".join(faltantes)
        )
    return colunas


def _padronizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza um bloco sem ocultar valores inválidos."""

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for coluna in COLUNAS_NUMERICAS:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
            # Os campos usados pelos critérios permanecem com infinito até a
            # limpeza, para que o relatório os contabilize como inválidos.
            if coluna not in {"svid", "elev", "S4", "VTEC"}:
                df[coluna] = df[coluna].replace([np.inf, -np.inf], np.nan)

    return df.loc[:, COLUNAS_ESTUDO]


def iterar_dados(
    caminho: str = ARQUIVO_RAW,
    tamanho_bloco: int = CSV_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Lê e padroniza o CSV em blocos de tamanho limitado."""

    _validar_csv(caminho)
    if tamanho_bloco <= 0:
        raise ValueError("O tamanho do bloco deve ser maior que zero.")

    with pd.read_csv(caminho, chunksize=tamanho_bloco, low_memory=True) as leitor:
        for bloco in leitor:
            yield _padronizar_tipos(bloco)


def carregar_dados(caminho: str = ARQUIVO_RAW) -> pd.DataFrame:
    """Carrega um CSV inteiro; mantido para bases pequenas e testes interativos.

    A execução principal usa :func:`processar_em_blocos`, que não concatena o
    arquivo completo em memória.
    """

    print("Carregando arquivo bruto...")
    quadros = list(iterar_dados(caminho))
    if not quadros:
        raise ValueError("O arquivo bruto está vazio.")
    return pd.concat(quadros, ignore_index=True)


def relatorio_inicial(df: pd.DataFrame) -> None:
    print("\n===== RELATÓRIO INICIAL =====")
    print("Número de observações:", len(df))
    print("Período:", df["datetime"].min(), "a", df["datetime"].max())
    print("Satélites:", df["svid"].nunique(dropna=True))
    print("\nValores ausentes:")
    print(df.isna().sum())


def _svid_valido(serie: pd.Series) -> pd.Series:
    """Valida o identificador antes da conversão potencialmente destrutiva."""

    return (
        serie.notna()
        & np.isfinite(serie)
        & serie.ge(1)
        & serie.lt(LIMITE_SUPERIOR_SVID)
        & serie.eq(np.floor(serie))
    )


def _limpar_bloco(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    criterios = {
        "datetime_invalido": df["datetime"].isna(),
        "svid_invalido": ~_svid_valido(df["svid"]),
        "elevacao_invalida": (
            df["elev"].isna()
            | ~np.isfinite(df["elev"])
            | ~df["elev"].between(MIN_ELEVATION, MAX_ELEVATION, inclusive="both")
        ),
        "s4_invalido": (
            df["S4"].isna()
            | ~np.isfinite(df["S4"])
            | ~df["S4"].between(MIN_S4, MAX_S4, inclusive="both")
        ),
    }

    rejeitar = pd.Series(False, index=df.index)
    for mascara in criterios.values():
        rejeitar |= mascara

    limpo = df.loc[~rejeitar].copy()
    limpo["svid"] = limpo["svid"].astype("int64")

    # TEC absoluto pode estar ausente ou conter viés de calibração. O registro
    # de S4 continua válido; somente o VTEC não físico é marcado como ausente.
    vtec_invalido = (
        limpo["VTEC"].notna()
        & (
            ~np.isfinite(limpo["VTEC"])
            | ~limpo["VTEC"].between(MIN_VTEC, MAX_VTEC, inclusive="both")
        )
    )
    limpo.loc[vtec_invalido, "VTEC"] = np.nan

    duplicadas = limpo.duplicated(subset=["datetime", "svid"], keep="first")
    limpo = limpo.loc[~duplicadas]
    limpo = limpo.sort_values(["datetime", "svid"], kind="stable").reset_index(
        drop=True
    )

    contagens = {
        "linhas_entrada": int(len(df)),
        **{nome: int(mascara.sum()) for nome, mascara in criterios.items()},
        "vtec_marcado_ausente": int(vtec_invalido.sum()),
        "duplicatas_removidas": int(duplicadas.sum()),
        "linhas_saida": int(len(limpo)),
    }
    return limpo, contagens


def _montar_relatorio(contagens: dict[str, int]) -> pd.DataFrame:
    ordem = (
        "linhas_entrada",
        "datetime_invalido",
        "svid_invalido",
        "elevacao_invalida",
        "s4_invalido",
        "vtec_marcado_ausente",
        "duplicatas_removidas",
        "linhas_saida",
    )
    return pd.DataFrame(
        [
            {"criterio": criterio, "quantidade": int(contagens.get(criterio, 0))}
            for criterio in ordem
        ]
    )


def limpeza(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica os filtros físicos a um DataFrame já carregado."""

    print("\nAplicando filtros...")
    limpo, contagens = _limpar_bloco(df)
    if limpo.empty:
        raise ValueError("Nenhuma observação permaneceu após o controle de qualidade.")

    print("Observações removidas:", len(df) - len(limpo))
    print("Observações finais:", len(limpo))
    print("Observações finais com VTEC:", int(limpo["VTEC"].notna().sum()))
    return limpo, _montar_relatorio(contagens)


def _atualizar_resumo_inicial(resumo: dict[str, Any], df: pd.DataFrame) -> None:
    resumo["linhas"] += len(df)

    inicio = df["datetime"].min()
    fim = df["datetime"].max()
    if pd.notna(inicio) and (resumo["inicio"] is None or inicio < resumo["inicio"]):
        resumo["inicio"] = inicio
    if pd.notna(fim) and (resumo["fim"] is None or fim > resumo["fim"]):
        resumo["fim"] = fim

    svids_validos = df.loc[_svid_valido(df["svid"]), "svid"]
    resumo["satelites"].update(svids_validos.unique().tolist())
    ausentes_bloco = df.isna().sum().astype("int64")
    if resumo["ausentes"] is None:
        resumo["ausentes"] = ausentes_bloco
    else:
        resumo["ausentes"] = resumo["ausentes"].add(
            ausentes_bloco, fill_value=0
        ).astype("int64")


def _imprimir_resumo_inicial(resumo: dict[str, Any]) -> None:
    print("\n===== RELATÓRIO INICIAL =====")
    print("Número de observações:", resumo["linhas"])
    print("Período:", resumo["inicio"], "a", resumo["fim"])
    print("Satélites:", len(resumo["satelites"]))
    print("\nValores ausentes:")
    print(resumo["ausentes"])


def _chave_ordenacao(linha: pd.Series) -> tuple[pd.Timestamp, int]:
    return pd.Timestamp(linha["datetime"]), int(linha["svid"])


def _substituir_saida(caminho_temporario: str, caminho_saida: str) -> None:
    """Substitui a saída, preservando a anterior se o Windows negar overwrite."""

    try:
        os.replace(caminho_temporario, caminho_saida)
        return
    except PermissionError:
        if not os.path.exists(caminho_saida):
            raise

    backup = f"{caminho_saida}.backup_{os.getpid()}"
    os.replace(caminho_saida, backup)
    try:
        os.replace(caminho_temporario, caminho_saida)
    except Exception:
        os.replace(backup, caminho_saida)
        raise
    else:
        os.remove(backup)


def processar_em_blocos(
    caminho_entrada: str = ARQUIVO_RAW,
    caminho_saida: str = ARQUIVO_CLEAN,
    tamanho_bloco: int = CSV_CHUNK_SIZE,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Limpa um CSV grande e grava a saída sem reuni-lo inteiro na RAM.

    O script 03 produz o arquivo bruto ordenado por ``datetime`` e ``svid``.
    Essa propriedade é validada entre os blocos para permitir a remoção exata
    de duplicatas que caiam na fronteira entre duas leituras.
    """

    _validar_csv(caminho_entrada)
    os.makedirs(os.path.dirname(os.path.abspath(caminho_saida)), exist_ok=True)
    descritor, caminho_temporario = tempfile.mkstemp(
        prefix=".clean_",
        suffix=".tmp",
        dir=os.path.dirname(os.path.abspath(caminho_saida)),
    )
    os.close(descritor)

    resumo: dict[str, Any] = {
        "linhas": 0,
        "inicio": None,
        "fim": None,
        "satelites": set(),
        "ausentes": None,
    }
    acumuladas: Counter[str] = Counter()
    ultima_chave: tuple[pd.Timestamp, int] | None = None
    cabecalho_escrito = False

    try:
        blocos = iterar_dados(caminho_entrada, tamanho_bloco)
        try:
            for numero_bloco, bloco in enumerate(blocos, start=1):
                _atualizar_resumo_inicial(resumo, bloco)
                limpo, contagens = _limpar_bloco(bloco)

                if not limpo.empty:
                    primeira_chave = _chave_ordenacao(limpo.iloc[0])
                    if ultima_chave is not None and primeira_chave < ultima_chave:
                        raise ValueError(
                            "O CSV bruto não está ordenado por datetime e svid. "
                            "Execute novamente o script 03 antes do controle de qualidade."
                        )
                    if ultima_chave is not None and primeira_chave == ultima_chave:
                        limpo = limpo.iloc[1:]
                        contagens["duplicatas_removidas"] += 1
                        contagens["linhas_saida"] -= 1

                if not limpo.empty:
                    ultima_chave = _chave_ordenacao(limpo.iloc[-1])
                    limpo.to_csv(
                        caminho_temporario,
                        mode="a" if cabecalho_escrito else "w",
                        header=not cabecalho_escrito,
                        index=False,
                        date_format="%Y-%m-%d %H:%M:%S",
                    )
                    cabecalho_escrito = True

                acumuladas.update(contagens)
                if numero_bloco == 1 or numero_bloco % 4 == 0:
                    print(f"Linhas processadas: {resumo['linhas']:,}")
        finally:
            blocos.close()

        if resumo["linhas"] == 0:
            raise ValueError("O arquivo bruto está vazio.")
        if not cabecalho_escrito:
            raise ValueError(
                "Nenhuma observação permaneceu após o controle de qualidade."
            )

        _substituir_saida(caminho_temporario, caminho_saida)
    except Exception:
        if os.path.exists(caminho_temporario):
            os.remove(caminho_temporario)
        raise

    return resumo, _montar_relatorio(dict(acumuladas))


def salvar_estatisticas(df: pd.DataFrame) -> None:
    estatisticas = df.describe(include="all")
    estatisticas.to_csv(os.path.join(PASTA_EDA, "resumo_estatistico.csv"))


def _descrever_datetime_csv(
    caminho: str,
    coluna: str = "datetime",
    tamanho_bloco: int = CSV_CHUNK_SIZE,
) -> pd.Series:
    """Descreve uma coluna temporal ordenada com memória limitada."""

    quantidade = 0
    soma_segundos = 0
    soma_nanos = 0
    minimo: pd.Timestamp | None = None
    maximo: pd.Timestamp | None = None
    ultima_data: pd.Timestamp | None = None

    parametros = {
        "usecols": [coluna],
        "parse_dates": [coluna],
        "date_format": "%Y-%m-%d %H:%M:%S",
        "chunksize": tamanho_bloco,
        "low_memory": True,
    }
    with pd.read_csv(caminho, **parametros) as leitor:
        for bloco in leitor:
            serie = bloco[coluna].dropna()
            if serie.empty:
                continue
            if not serie.is_monotonic_increasing:
                raise ValueError(
                    f"A coluna {coluna} do CSV limpo não está ordenada."
                )
            inicio_bloco = pd.Timestamp(serie.iloc[0])
            fim_bloco = pd.Timestamp(serie.iloc[-1])
            if ultima_data is not None and inicio_bloco < ultima_data:
                raise ValueError(
                    f"A coluna {coluna} do CSV limpo não está ordenada."
                )
            ultima_data = fim_bloco

            nanos = serie.astype("int64").to_numpy()
            segundos = np.floor_divide(nanos, 1_000_000_000)
            restos = nanos - segundos * 1_000_000_000
            quantidade += len(nanos)
            soma_segundos += int(segundos.sum(dtype="int64"))
            soma_nanos += int(restos.sum(dtype="int64"))
            minimo = inicio_bloco if minimo is None else min(minimo, inicio_bloco)
            maximo = fim_bloco if maximo is None else max(maximo, fim_bloco)

    if quantidade == 0:
        return pd.Series({"count": 0}, dtype="object")

    posicoes: dict[str, tuple[int, int, float]] = {}
    indices_necessarios: set[int] = set()
    for rotulo, quantil in (("25%", 0.25), ("50%", 0.5), ("75%", 0.75)):
        posicao = quantil * (quantidade - 1)
        inferior = int(np.floor(posicao))
        superior = int(np.ceil(posicao))
        posicoes[rotulo] = (inferior, superior, posicao - inferior)
        indices_necessarios.update((inferior, superior))

    valores: dict[int, pd.Timestamp] = {}
    deslocamento = 0
    with pd.read_csv(caminho, **parametros) as leitor:
        for bloco in leitor:
            serie = bloco[coluna].dropna().reset_index(drop=True)
            fim_bloco = deslocamento + len(serie)
            for indice in indices_necessarios:
                if deslocamento <= indice < fim_bloco:
                    valores[indice] = pd.Timestamp(serie.iloc[indice - deslocamento])
            deslocamento = fim_bloco
            if len(valores) == len(indices_necessarios):
                break

    quantis: dict[str, pd.Timestamp] = {}
    for rotulo, (inferior, superior, fracao) in posicoes.items():
        if inferior == superior:
            quantis[rotulo] = valores[inferior]
        else:
            quantis[rotulo] = pd.Series(
                [valores[inferior], valores[superior]]
            ).quantile(fracao)

    media_ns = (
        soma_segundos * 1_000_000_000 + soma_nanos
    ) // quantidade
    return pd.Series(
        {
            "count": quantidade,
            "mean": pd.Timestamp(media_ns),
            "min": minimo,
            **quantis,
            "max": maximo,
        },
        dtype="object",
    )


def salvar_estatisticas_csv(
    caminho: str = ARQUIVO_CLEAN,
    pasta_saida: str = PASTA_EDA,
    tamanho_bloco: int = CSV_CHUNK_SIZE,
) -> pd.DataFrame:
    """Calcula estatísticas exatas lendo uma coluna por vez.

    Essa estratégia conserva os mesmos quartis de ``DataFrame.describe`` sem
    manter todas as colunas do arquivo simultaneamente na memória.
    """

    colunas = list(pd.read_csv(caminho, nrows=0).columns)
    descricoes: dict[str, pd.Series] = {}
    for coluna in colunas:
        print(f"Calculando estatísticas: {coluna}")
        if coluna == "datetime":
            descricoes[coluna] = _descrever_datetime_csv(
                caminho, coluna, tamanho_bloco
            )
            continue
        elif coluna in COLUNAS_NUMERICAS:
            serie = pd.read_csv(
                caminho,
                usecols=[coluna],
                dtype={coluna: "float64"},
                low_memory=True,
            )[coluna]
        else:
            serie = pd.read_csv(
                caminho,
                usecols=[coluna],
                dtype={coluna: "category"},
                low_memory=True,
            )[coluna]
        descricoes[coluna] = serie.describe()

    os.makedirs(pasta_saida, exist_ok=True)
    estatisticas = pd.DataFrame(descricoes).reindex(ORDEM_ESTATISTICAS)
    estatisticas.to_csv(os.path.join(pasta_saida, "resumo_estatistico.csv"))
    return estatisticas


def _salvar_figura(nome: str, pasta_saida: str = PASTA_EDA) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, nome), dpi=300, bbox_inches="tight")
    plt.close()


def gerar_graficos(df: pd.DataFrame) -> None:
    """Gera os gráficos a partir de um DataFrame em memória."""

    print("\nGerando gráficos...")

    plt.figure(figsize=(8, 5))
    plt.hist(df["S4"].dropna(), bins=50)
    plt.axvline(
        S4_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"Limiar severo ({S4_THRESHOLD:g})",
    )
    plt.xlabel("S4 corrigido")
    plt.ylabel("Frequência")
    plt.title("Distribuição do índice S4 corrigido")
    plt.grid(alpha=0.3)
    plt.legend()
    _salvar_figura("distribuicao_S4.png")

    dados_vtec = df.dropna(subset=["VTEC"])
    if not dados_vtec.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(dados_vtec["VTEC"], bins=50)
        plt.xlabel("VTEC (TECU)")
        plt.ylabel("Frequência")
        plt.title("Distribuição do VTEC válido")
        plt.grid(alpha=0.3)
        _salvar_figura("distribuicao_VTEC.png")

        mediana_vtec = (
            dados_vtec.set_index("datetime")["VTEC"]
            .resample(TIME_RESOLUTION)
            .median()
        )
        plt.figure(figsize=(12, 5))
        plt.plot(mediana_vtec.index, mediana_vtec.values)
        plt.xlabel("Tempo")
        plt.ylabel("VTEC mediano (TECU)")
        plt.title(f"VTEC observacional agregado a cada {TIME_RESOLUTION}")
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        _salvar_figura("serie_temporal_VTEC.png")
    else:
        print("Aviso: não há valores VTEC válidos; gráficos de VTEC não foram gerados.")

    mediana_s4 = df.set_index("datetime")["S4"].resample(TIME_RESOLUTION).median()
    plt.figure(figsize=(12, 5))
    plt.plot(mediana_s4.index, mediana_s4.values)
    plt.axhline(S4_THRESHOLD, color="red", linestyle="--")
    plt.xlabel("Tempo")
    plt.ylabel("S4 corrigido mediano")
    plt.title(f"S4 observacional agregado a cada {TIME_RESOLUTION}")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    _salvar_figura("serie_temporal_S4.png")

    contagem = df["svid"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    contagem.plot(kind="bar")
    plt.xlabel("SVID")
    plt.ylabel("Quantidade")
    plt.title("Observações válidas por satélite")
    plt.grid(axis="y", alpha=0.3)
    _salvar_figura("observacoes_por_satelite.png")


def _limites_histograma(
    estatisticas: pd.DataFrame, coluna: str
) -> np.ndarray | None:
    if coluna not in estatisticas or pd.isna(estatisticas.at["min", coluna]):
        return None
    minimo = float(estatisticas.at["min", coluna])
    maximo = float(estatisticas.at["max", coluna])
    return np.histogram_bin_edges(np.array([minimo, maximo]), bins=50)


def _completar_indice_temporal(serie: pd.Series) -> pd.Series:
    serie = serie.dropna()
    if serie.empty:
        return serie
    indice = pd.date_range(serie.index.min(), serie.index.max(), freq=TIME_RESOLUTION)
    return serie.reindex(indice)


def _validar_resolucao_temporal_fixa() -> None:
    try:
        deslocamento = pd.tseries.frequencies.to_offset(TIME_RESOLUTION)
    except ValueError as erro:
        raise ValueError(f"TIME_RESOLUTION inválida: {TIME_RESOLUTION}.") from erro
    if not isinstance(deslocamento, pd.offsets.Tick):
        raise ValueError(
            "A EDA em blocos exige TIME_RESOLUTION fixa, por exemplo '30min' ou '1h'."
        )


def gerar_graficos_csv(
    caminho: str,
    estatisticas: pd.DataFrame,
    pasta_saida: str = PASTA_EDA,
    tamanho_bloco: int = CSV_CHUNK_SIZE,
) -> None:
    """Gera histogramas e séries temporais sem carregar o CSV completo."""

    print("\nGerando gráficos em blocos...")
    _validar_resolucao_temporal_fixa()
    bordas_s4 = _limites_histograma(estatisticas, "S4")
    bordas_vtec = _limites_histograma(estatisticas, "VTEC")
    hist_s4 = np.zeros(50, dtype="int64")
    hist_vtec = np.zeros(50, dtype="int64")
    contagem_svid: Counter[int] = Counter()
    medianas: list[pd.DataFrame] = []
    pendente = pd.DataFrame(columns=["datetime", "S4", "VTEC"])

    with pd.read_csv(
        caminho,
        usecols=["datetime", "svid", "S4", "VTEC"],
        dtype={"svid": "int64", "S4": "float64", "VTEC": "float64"},
        parse_dates=["datetime"],
        date_format="%Y-%m-%d %H:%M:%S",
        chunksize=tamanho_bloco,
        low_memory=True,
    ) as leitor:
        for bloco in leitor:
            if bordas_s4 is not None:
                hist_s4 += np.histogram(bloco["S4"].dropna(), bins=bordas_s4)[0]
            if bordas_vtec is not None:
                hist_vtec += np.histogram(bloco["VTEC"].dropna(), bins=bordas_vtec)[0]
            contagem_svid.update(
                {
                    int(svid): int(qtd)
                    for svid, qtd in bloco["svid"].value_counts().items()
                }
            )

            temporal = bloco[["datetime", "S4", "VTEC"]]
            if not pendente.empty:
                temporal = pd.concat([pendente, temporal], ignore_index=True)
            periodos = temporal["datetime"].dt.floor(TIME_RESOLUTION)
            ultimo_periodo = periodos.iloc[-1]
            finalizar = periodos.ne(ultimo_periodo)
            if finalizar.any():
                prontos = temporal.loc[finalizar].copy()
                prontos["_periodo"] = periodos.loc[finalizar].to_numpy()
                medianas.append(
                    prontos.groupby("_periodo", sort=False)[["S4", "VTEC"]].median()
                )
            pendente = temporal.loc[
                ~finalizar, ["datetime", "S4", "VTEC"]
            ].copy()

    if not pendente.empty:
        pendente["_periodo"] = pendente["datetime"].dt.floor(TIME_RESOLUTION)
        medianas.append(
            pendente.groupby("_periodo", sort=False)[["S4", "VTEC"]].median()
        )

    agregados = pd.concat(medianas).sort_index()

    if bordas_s4 is not None:
        plt.figure(figsize=(8, 5))
        plt.bar(
            bordas_s4[:-1], hist_s4, width=np.diff(bordas_s4), align="edge"
        )
        plt.axvline(
            S4_THRESHOLD,
            color="red",
            linestyle="--",
            label=f"Limiar severo ({S4_THRESHOLD:g})",
        )
        plt.xlabel("S4 corrigido")
        plt.ylabel("Frequência")
        plt.title("Distribuição do índice S4 corrigido")
        plt.grid(alpha=0.3)
        plt.legend()
        _salvar_figura("distribuicao_S4.png", pasta_saida)

    mediana_s4 = _completar_indice_temporal(agregados["S4"])
    plt.figure(figsize=(12, 5))
    plt.plot(mediana_s4.index, mediana_s4.values)
    plt.axhline(S4_THRESHOLD, color="red", linestyle="--")
    plt.xlabel("Tempo")
    plt.ylabel("S4 corrigido mediano")
    plt.title(f"S4 observacional agregado a cada {TIME_RESOLUTION}")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    _salvar_figura("serie_temporal_S4.png", pasta_saida)

    if bordas_vtec is not None:
        plt.figure(figsize=(8, 5))
        plt.bar(
            bordas_vtec[:-1], hist_vtec, width=np.diff(bordas_vtec), align="edge"
        )
        plt.xlabel("VTEC (TECU)")
        plt.ylabel("Frequência")
        plt.title("Distribuição do VTEC válido")
        plt.grid(alpha=0.3)
        _salvar_figura("distribuicao_VTEC.png", pasta_saida)

        mediana_vtec = _completar_indice_temporal(agregados["VTEC"])
        plt.figure(figsize=(12, 5))
        plt.plot(mediana_vtec.index, mediana_vtec.values)
        plt.xlabel("Tempo")
        plt.ylabel("VTEC mediano (TECU)")
        plt.title(f"VTEC observacional agregado a cada {TIME_RESOLUTION}")
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        _salvar_figura("serie_temporal_VTEC.png", pasta_saida)
    else:
        print("Aviso: não há valores VTEC válidos; gráficos de VTEC não foram gerados.")
        for nome in ("distribuicao_VTEC.png", "serie_temporal_VTEC.png"):
            caminho_obsoleto = os.path.join(pasta_saida, nome)
            if os.path.isfile(caminho_obsoleto):
                os.remove(caminho_obsoleto)

    contagem = pd.Series(dict(contagem_svid), dtype="int64").sort_index()
    plt.figure(figsize=(10, 5))
    contagem.plot(kind="bar")
    plt.xlabel("SVID")
    plt.ylabel("Quantidade")
    plt.title("Observações válidas por satélite")
    plt.grid(axis="y", alpha=0.3)
    _salvar_figura("observacoes_por_satelite.png", pasta_saida)


def main() -> None:
    print("=" * 70)
    print("CONTROLE DE QUALIDADE E ANÁLISE EXPLORATÓRIA")
    print("ESTAÇÃO:", STATION)
    print("=" * 70)

    os.makedirs(PASTA_ESTACAO, exist_ok=True)
    os.makedirs(PASTA_EDA, exist_ok=True)

    print(
        f"Carregando e filtrando o arquivo bruto em blocos de {CSV_CHUNK_SIZE:,} linhas..."
    )
    resumo, relatorio = processar_em_blocos()
    _imprimir_resumo_inicial(resumo)

    linhas_entrada = int(
        relatorio.loc[relatorio["criterio"] == "linhas_entrada", "quantidade"].iloc[0]
    )
    linhas_saida = int(
        relatorio.loc[relatorio["criterio"] == "linhas_saida", "quantidade"].iloc[0]
    )
    print("\nObservações removidas:", linhas_entrada - linhas_saida)
    print("Observações finais:", linhas_saida)

    relatorio.to_csv(
        os.path.join(PASTA_EDA, "relatorio_controle_qualidade.csv"), index=False
    )
    estatisticas = salvar_estatisticas_csv()
    gerar_graficos_csv(ARQUIVO_CLEAN, estatisticas)

    print("\nArquivo limpo salvo:", ARQUIVO_CLEAN)
    print("Relatórios e gráficos salvos:", PASTA_EDA)


if __name__ == "__main__":
    main()
