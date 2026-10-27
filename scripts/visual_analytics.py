"""Figuras científicas e revisão visual reproduzível das observações GNSS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpolacao_vtec import interpolar_por_periodo

from config import (
    PROCESSED_DATA, RESULTS_DIRECTORY, STATION, S4_PROBABILITY_THRESHOLD,
    S4_THRESHOLD, TEST_START, TEST_END,
)

COLORS = {"original": "#64748b", "filtered": "#087e8b", "alert": "#dc5b32",
          "manual": "#9333a8", "pred": "#2563b0", "ink": "#183149", "cs": "#b88919"}


def estilo() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#cbd5e1",
        "axes.grid": True, "grid.color": "#e2e8f0", "grid.alpha": 0.65,
        "axes.axisbelow": True, "legend.frameon": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def salvar(fig: plt.Figure, pasta: Path, nome: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(pasta / f"{nome}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def eixo_tempo(ax: plt.Axes, diario: bool = False) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    detalhe = ax.get_xlim()[1] - ax.get_xlim()[0] <= 4
    formato = "%H:%M" if diario else "%d/%m\n%H:%M" if detalhe else "%d/%m"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(formato))
    ax.set_xlabel("Hora UTC" if diario else "Data / hora UTC" if detalhe else "Data UTC")


def limiar_salvo(dados: pd.DataFrame) -> float:
    """Recupera o corte da estação; arquivos antigos usam a configuração legada."""
    if "limiar_probabilidade" not in dados:
        return float(S4_PROBABILITY_THRESHOLD)
    valores = pd.to_numeric(dados["limiar_probabilidade"], errors="raise").dropna().unique()
    if len(valores) != 1 or not np.isfinite(valores[0]) or not 0 <= valores[0] <= 1:
        raise ValueError("O arquivo deve conter um único limiar_probabilidade válido para a estação.")
    return float(valores[0])


def preparar_observacoes(features: pd.DataFrame, predicoes: pd.DataFrame) -> pd.DataFrame:
    """O alerta se junta ao VTEC observado em t+h, pela chave época/SVID."""
    obs = features[["datetime", "svid", "S4", "VTEC"]].copy()
    obs["datetime"] = pd.to_datetime(obs["datetime"], errors="raise")
    limiar = limiar_salvo(predicoes)
    cols = ["target_datetime", "svid", "probabilidade_severa", "origem_probabilidade"]
    if "sinalizar_filtro" in predicoes:
        cols.append("sinalizar_filtro")
    pred = predicoes[cols].copy()
    if "sinalizar_filtro" in pred:
        flags = pred["sinalizar_filtro"].astype("string").str.lower().str.strip().map(
            {"true": True, "false": False, "1": True, "0": False})
        if flags.isna().any():
            raise ValueError("sinalizar_filtro contém valores booleanos inválidos.")
        pred["sinalizar_filtro"] = flags.astype(bool)
    pred["target_datetime"] = pd.to_datetime(pred["target_datetime"], errors="raise")
    pred = pred.rename(columns={"target_datetime": "datetime"})
    obs = obs.merge(pred, on=["datetime", "svid"], how="left", validate="one_to_one")
    p = obs["probabilidade_severa"]
    if not p.dropna().between(0, 1).all():
        raise ValueError("Probabilidades devem pertencer a [0, 1].")
    obs["limiar_probabilidade"] = limiar
    # O CSV de inferência é a autoridade, incluindo regras de disponibilidade.
    obs["sinalizado_nucleo1"] = (obs["sinalizar_filtro"].eq(True)
                                    if "sinalizar_filtro" in obs else p.ge(limiar))
    if (obs["sinalizado_nucleo1"] & p.isna()).any():
        raise ValueError("Um alerta salvo não pode ter probabilidade ausente.")
    obs["origem_probabilidade"] = obs["origem_probabilidade"].fillna("sem_predicao")
    obs = obs.loc[obs[["S4", "VTEC"]].notna().any(axis=1)].sort_values(["datetime", "svid"])
    return obs.reset_index(drop=True)


def assinatura(obs: pd.DataFrame) -> str:
    """Impede reaplicar decisões a outro conjunto de observações/predições."""
    cols = ["datetime", "svid", "S4", "VTEC", "probabilidade_severa", "sinalizado_nucleo1"]
    return hashlib.sha256(pd.util.hash_pandas_object(obs[cols], index=False).values.tobytes()).hexdigest()


def anexar_observado(serie: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Mediana das medições existentes; lacunas continuam NaN na dispersão."""
    observado = obs.groupby("datetime")["VTEC"].median().rename("VTEC_observado")
    return serie.drop(columns=["VTEC_observado"], errors="ignore").merge(
        observado, on="datetime", how="left", validate="one_to_one"
    )


def resumir_cintilacao(obs: pd.DataFrame, serie: pd.DataFrame) -> pd.DataFrame:
    """Resume médias por SVID já reamostradas; não interpola S4 nem eventos CS."""
    dados = obs.assign(cs=obs["S4"].ge(S4_THRESHOLD))
    agregado = dados.groupby("datetime").agg(
        S4_mediana_observada=("S4", "median"), n_s4_observados=("S4", "count"),
        n_cs_observada=("cs", "sum"), n_alertas_nucleo1=("sinalizado_nucleo1", "sum"))
    agregado = agregado.reindex(pd.DatetimeIndex(serie["datetime"]))
    counts = ["n_s4_observados", "n_cs_observada", "n_alertas_nucleo1"]
    agregado[counts] = agregado[counts].fillna(0).astype(int)
    agregado["fracao_cs_observada"] = agregado["n_cs_observada"].div(
        agregado["n_s4_observados"].replace(0, np.nan))
    return agregado.reset_index()


def decisoes_revisao(revisao: dict) -> list[dict]:
    versao = revisao.get("schema_version")
    campo = "exclusoes" if versao == 1 else "decisoes"
    decisoes = revisao.get(campo)
    if versao not in (1, 2) or not isinstance(decisoes, list):
        raise ValueError("Revisão deve usar v1/exclusoes ou v2/decisoes.")
    return [dict(d, acao="excluir_observacao") if versao == 1 else d for d in decisoes]


def aplicar_revisao(obs: pd.DataFrame, serie: pd.DataFrame, revisao: dict,
                    station: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Revisão derivada: restaura falsos alertas ou exclui medições explicitamente."""
    if revisao.get("schema_version") not in (1, 2) or revisao.get("estacao") != station:
        raise ValueError("Revisão incompatível com a estação ou versão do formato.")
    if revisao.get("assinatura_dados") != assinatura(obs):
        raise ValueError("A revisão pertence a outro conjunto de dados; abra o painel atualizado.")
    decisoes = decisoes_revisao(revisao)
    obs = obs.copy()
    obs["removido_manual"] = False
    obs["falso_positivo_manual"] = False
    obs["acao_revisao"] = ""
    obs["motivo_revisao"] = ""
    obs["criterio_revisao"] = ""
    obs["limiar_revisao"] = np.nan
    keys = {(row.datetime, int(row.svid)): idx for idx, row in obs.iterrows()}
    seen = set()
    for decisao in decisoes:
        key = (pd.Timestamp(decisao["datetime"]), int(decisao["svid"]))
        if key in seen or key not in keys:
            raise ValueError(f"Chave duplicada ou ausente na revisão: {key}")
        seen.add(key)
        idx = keys[key]
        criterio = decisao.get("criterio")
        if criterio not in ("probabilidade_severa", "S4", "VTEC"):
            raise ValueError("Critério de revisão inválido.")
        limiar = float(decisao["limiar"])
        valor = obs.at[idx, criterio]
        if not np.isfinite(limiar) or pd.isna(valor) or not valor < limiar:
            raise ValueError(f"Observação não está abaixo do limiar: {key}")
        if not str(decisao.get("motivo", "")).strip():
            raise ValueError("Toda exclusão manual precisa de motivo.")
        if pd.isna(obs.at[idx, "VTEC"]):
            raise ValueError("A observação selecionada não possui VTEC.")
        acao = decisao.get("acao")
        if acao not in ("falso_positivo", "excluir_observacao"):
            raise ValueError("Ação inválida: use falso_positivo ou excluir_observacao.")
        if acao == "falso_positivo" and not obs.at[idx, "sinalizado_nucleo1"]:
            raise ValueError("Revisar um falso positivo exige sinalização original do Núcleo 1.")
        obs.at[idx, "falso_positivo_manual" if acao == "falso_positivo" else "removido_manual"] = True
        obs.at[idx, "acao_revisao"] = acao
        obs.at[idx, "motivo_revisao"] = str(decisao["motivo"]).strip()
        obs.at[idx, "criterio_revisao"] = criterio
        obs.at[idx, "limiar_revisao"] = limiar
    obs["sinalizado_revisado"] = obs["sinalizado_nucleo1"] & ~obs["falso_positivo_manual"]
    obs["VTEC_revisado"] = obs["VTEC"].mask(obs["sinalizado_revisado"] | obs["removido_manual"])
    agregado = obs.groupby("datetime")["VTEC_revisado"].median()
    saida = serie.copy().set_index("datetime")
    saida["VTEC_revisado"] = interpolar_por_periodo(agregado.reindex(saida.index), TEST_START)
    if saida["VTEC_revisado"].isna().any():
        raise ValueError("A seleção não deixa observações suficientes para reconstruir o VTEC.")
    contagem = obs.loc[obs["removido_manual"] & ~obs["sinalizado_nucleo1"]].groupby("datetime").size()
    saida["n_remocoes_manuais_adicionais"] = contagem.reindex(saida.index, fill_value=0).astype(int)
    restaurados = obs.loc[obs["falso_positivo_manual"]].groupby("datetime").size()
    saida["n_falsos_positivos_restaurados"] = restaurados.reindex(saida.index, fill_value=0).astype(int)
    return obs, saida.reset_index()


def _desenhar_historico(ax: plt.Axes, obs: pd.DataFrame, serie: pd.DataFrame) -> None:
    ax.scatter(obs["datetime"], obs["VTEC"], s=4, alpha=0.12, color=COLORS["original"],
               label="Observações por SVID (30 min)", rasterized=True)
    ax.plot(serie["datetime"], serie["VTEC_original"], color=COLORS["ink"], lw=1,
            label="Mediana original (com interpolação)")
    ax.plot(serie["datetime"], serie["VTEC_filtered"], color=COLORS["filtered"], lw=1.3,
            label="Mediana após filtro do Núcleo 1")
    alerts = obs.loc[obs["sinalizado_nucleo1"] & obs["VTEC"].notna()]
    ax.scatter(alerts["datetime"], alerts["VTEC"], s=36, facecolors="none",
               edgecolors=COLORS["alert"], linewidths=0.9, label="Época/SVID sinalizado pelo Núcleo 1")
    if "VTEC_revisado" in serie:
        ax.plot(serie["datetime"], serie["VTEC_revisado"], color=COLORS["manual"], lw=1.3, label="Após revisão manual")
        manual = obs.loc[obs["removido_manual"]]
        ax.scatter(manual["datetime"], manual["VTEC"], marker="x", s=38, color=COLORS["manual"], label="Remoção manual")
        restaurados = obs.loc[obs["falso_positivo_manual"]]
        ax.scatter(restaurados["datetime"], restaurados["VTEC"], marker="D", s=40,
                   facecolors="none", edgecolors=COLORS["manual"], label="Falso positivo · VTEC reincluído")
    ax.set_ylabel("VTEC (TECU)")


def figuras_historicas(obs: pd.DataFrame, serie: pd.DataFrame, pasta: Path, station: str) -> None:
    # Usa a última janela disponível, sem procurar um evento favorável no teste.
    fim = serie["datetime"].max()
    inicio = fim.normalize() - pd.Timedelta(days=2)
    zoom = serie.loc[serie["datetime"].between(inicio, fim)]
    pontos = obs.loc[obs["datetime"].between(inicio, fim)]
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), layout="constrained", gridspec_kw={"height_ratios": [1.1, 1.3, 0.6]})
    fig.suptitle(f"{station} | VTEC histórico e intervenção do Núcleo 1", fontsize=19, fontweight="bold")
    _desenhar_historico(axes[0], obs, serie)
    axes[0].set_title("a  Histórico completo · dispersão das observações e séries agregadas", loc="left")
    axes[0].axvspan(inicio, fim, alpha=0.12, color=COLORS["pred"])
    axes[0].legend(loc="upper left", ncols=2, fontsize=8)
    eixo_tempo(axes[0])
    _desenhar_historico(axes[1], pontos, zoom)
    axes[1].set_title(f"b  Detalhe de {inicio:%d/%m} a {fim:%d/%m} · círculos no instante-alvo t + 30 min", loc="left")
    eixo_tempo(axes[1])
    delta = zoom["VTEC_filtered"] - zoom["VTEC_original"]
    axes[2].fill_between(zoom["datetime"], delta, color=COLORS["filtered"], alpha=0.25)
    axes[2].plot(zoom["datetime"], delta, color=COLORS["filtered"], lw=1)
    if "VTEC_revisado" in zoom:
        axes[2].plot(zoom["datetime"], zoom["VTEC_revisado"] - zoom["VTEC_original"],
                     color=COLORS["manual"], lw=1.3, label="Revisado − original")
        axes[2].legend()
    axes[2].axhline(0, color=COLORS["ink"], lw=0.7)
    axes[2].set_title("c  Alteração na mediana · filtrado − original", loc="left")
    axes[2].set_ylabel("Δ VTEC (TECU)")
    eixo_tempo(axes[2])
    salvar(fig, pasta, "vtec_historico_deteccoes")

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, layout="constrained")
    fig.suptitle(f"{station} | Núcleo 1 · evidência observada e confiança da previsão", fontsize=18, fontweight="bold")
    axes[0].scatter(pontos["datetime"], pontos["S4"], s=11, color=COLORS["original"], alpha=0.35, label="S4 observado por SVID")
    alerts = pontos.loc[pontos["sinalizado_nucleo1"]]
    axes[0].scatter(alerts["datetime"], alerts["S4"], s=65, facecolors="none", edgecolors=COLORS["alert"], label="Sinalização do Núcleo 1")
    axes[0].axhline(S4_THRESHOLD, ls="--", color=COLORS["ink"], label=f"S4 = {S4_THRESHOLD:g}")
    axes[0].set_ylabel("S4 observado")
    axes[0].legend(ncols=3, fontsize=9)
    axes[1].scatter(pontos["datetime"], pontos["probabilidade_severa"], s=10, alpha=0.35, color=COLORS["pred"])
    axes[1].scatter(alerts["datetime"], alerts["probabilidade_severa"], s=50, facecolors="none", edgecolors=COLORS["alert"])
    limiar = limiar_salvo(obs)
    axes[1].axhline(limiar, ls="--", color=COLORS["alert"], label=f"Limiar salvo da estação = {limiar:g}")
    axes[1].set_ylim(-0.03, 1.06)
    axes[1].set_ylabel("P(S4 futuro ≥ limiar)")
    axes[1].legend(loc="upper right")
    _desenhar_historico(axes[2], pontos, zoom)
    for ax in axes:
        ax.axvspan(pd.Timestamp(TEST_START), min(fim, pd.Timestamp(TEST_END)), color="#f6c35b", alpha=0.1)
    eixo_tempo(axes[-1])
    axes[-1].set_title("VTEC correspondente · faixa amarela = teste final", loc="left")
    salvar(fig, pasta, "nucleo1_epocas_circuladas")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    fig.suptitle(f"{station} | Efeito da filtragem no histórico de VTEC", fontsize=18, fontweight="bold")
    x, y = serie["VTEC_original"], serie["VTEC_filtered"]
    affected = serie["n_svid_filtrados"].gt(0)
    axes[0].scatter(x, y, s=12, color=COLORS["filtered"], alpha=0.4, rasterized=True)
    axes[0].scatter(x[affected], y[affected], s=35, facecolors="none", edgecolors=COLORS["alert"], label="Época com SVID sinalizado")
    bounds = [min(x.min(), y.min()), max(x.max(), y.max())]
    if "VTEC_revisado" in serie:
        revised = serie["VTEC_revisado"]
        axes[0].scatter(x, revised, s=14, color=COLORS["manual"], alpha=0.5, label="Após revisão manual", rasterized=True)
        axes[1].scatter((x+revised)/2, revised-x, s=14, color=COLORS["manual"], alpha=0.5, label="Revisado − original", rasterized=True)
        axes[1].legend()
        bounds = [min(bounds[0], revised.min()), max(bounds[1], revised.max())]
    axes[0].plot(bounds, bounds, ls="--", color=COLORS["ink"], label="Identidade 1:1")
    axes[0].set(xlabel="VTEC original (TECU)", ylabel="VTEC filtrado (TECU)", aspect="equal")
    axes[0].legend(fontsize=9)
    axes[1].scatter((x+y)/2, y-x, s=12, alpha=0.4, color=COLORS["filtered"], rasterized=True)
    axes[1].scatter(((x+y)/2)[affected], (y-x)[affected], s=35, facecolors="none", edgecolors=COLORS["alert"])
    axes[1].axhline(0, color=COLORS["ink"], ls="--")
    axes[1].set(xlabel="Média entre original e filtrado (TECU)", ylabel="Filtrado − original (TECU)")
    axes[1].set_title(f"{len(serie):,} épocas · {affected.sum():,} com filtro\nDiferença absoluta média = {abs(y-x).mean():.3f} TECU", loc="left")
    salvar(fig, pasta, "vtec_dispersao_original_filtrado")


def figuras_s4_vtec(obs: pd.DataFrame, serie: pd.DataFrame, pasta: Path, station: str) -> None:
    """Painéis científicos alinhados: medidas, eventos observados e alertas salvos."""
    resumo = resumir_cintilacao(obs, serie)
    for nome, inicio, fim in (
        ("s4_vtec_cs_30min_historico", serie["datetime"].min(), serie["datetime"].max()),
        ("s4_vtec_cs_30min_teste", pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)),
    ):
        pontos = obs.loc[obs["datetime"].between(inicio, fim)]
        dados = resumo.loc[resumo["datetime"].between(inicio, fim)]
        agregado = serie.loc[serie["datetime"].between(inicio, fim)]
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [1.2, .75, 1.45, .65]})
        fig.suptitle(f"{station} | S4, cintilação observada e VTEC · 30 minutos", fontsize=19, fontweight="bold")
        cs = pontos.loc[pontos["S4"].ge(S4_THRESHOLD)]
        alertas = pontos.loc[pontos["sinalizado_nucleo1"]]
        axes[0].scatter(pontos["datetime"], pontos["S4"], s=9, color=COLORS["original"],
                        alpha=.3, label="S4 observado · média de cada SVID em 30 min", rasterized=True)
        axes[0].scatter(cs["datetime"], cs["S4"], s=17, color=COLORS["cs"],
                        alpha=.8, label=f"CS observada · S4 ≥ {S4_THRESHOLD:g}")
        axes[0].scatter(alertas["datetime"], alertas["S4"], s=45, facecolors="none",
                        edgecolors=COLORS["alert"], label="Alerta salvo do Núcleo 1")
        axes[0].axhline(S4_THRESHOLD, color=COLORS["cs"], ls="--", lw=1)
        axes[0].set(title="a  Medições reais em dispersão · círculos posicionados no instante-alvo", ylabel="S4")
        axes[0].legend(ncols=3, fontsize=8, loc="upper left")
        axes[1].scatter(pontos["datetime"], pontos["probabilidade_severa"], s=8,
                        color=COLORS["pred"], alpha=.4, label="Probabilidade prevista por SVID")
        axes[1].axhline(limiar_salvo(obs), color=COLORS["alert"], ls="--", lw=1,
                        label=f"Limiar salvo = {limiar_salvo(obs):g}")
        axes[1].set(ylabel="P(CS)", ylim=(-.03, 1.07), title="b  Inferência no horizonte de 30 min")
        axes[1].legend(fontsize=8, ncols=2, loc="upper left")
        _desenhar_historico(axes[2], pontos, agregado)
        axes[2].scatter(cs["datetime"], cs["VTEC"], s=12, color=COLORS["cs"], alpha=.65,
                        label="VTEC na mesma época/SVID de CS observada", rasterized=True)
        axes[2].set_title("c  VTEC real em pontos · medianas derivadas em linhas", loc="left")
        axes[2].legend(ncols=2, fontsize=8, loc="upper left")
        axes[3].bar(dados["datetime"], dados["n_cs_observada"], width=1/48, color=COLORS["cs"],
                    alpha=.65, label="SVIDs com CS observada")
        axes[3].scatter(dados["datetime"], dados["n_alertas_nucleo1"], color=COLORS["alert"],
                        s=12, marker="x", label="SVIDs sinalizados pelo Núcleo 1")
        axes[3].set(ylabel="Nº de SVIDs", title="d  Contagem por época · ausência de medidas não implica S4 = 0")
        axes[3].legend(ncols=2, fontsize=8, loc="upper left")
        axes[-1].set_xlim(inicio, fim)
        eixo_tempo(axes[-1], diario="teste" in nome)
        fig.supxlabel("Reamostragem já realizada na entrada: média em 30 min por SVID; CS é classificada após essa média.", fontsize=9)
        salvar(fig, pasta, nome)

    # Mesma codificação da figura de referência: cinza = normal,
    # viridis = SVID com CS observada, círculos = alertas (inclusive falsos).
    fim = serie["datetime"].max()
    inicio = max(serie["datetime"].min(), fim.normalize() - pd.Timedelta(days=2))
    pontos = obs.loc[obs["datetime"].between(inicio, fim)]
    severos = pontos.loc[pontos["S4"].ge(S4_THRESHOLD)]
    alertas = pontos.loc[pontos["sinalizado_nucleo1"]]
    fig, ax = plt.subplots(figsize=(16, 6), layout="constrained")
    fig.suptitle(f"{station} | Cintilação real × alertas previstos", fontsize=19, fontweight="bold")
    ax.scatter(pontos["datetime"], pontos["S4"], s=13, color="#aebac6", alpha=.25,
               label="S4 observado · média de 30 min por SVID", rasterized=True)
    if not severos.empty:
        colorido = ax.scatter(severos["datetime"], severos["S4"], c=severos["svid"], cmap="viridis",
                              vmin=obs["svid"].min(), vmax=obs["svid"].max(), s=25,
                              label=f"CS real · S4 ≥ {S4_THRESHOLD:g}", zorder=3)
        fig.colorbar(colorido, ax=ax, pad=.01, fraction=.025, label="SVID (identificador do satélite)")
    ax.scatter(alertas["datetime"], alertas["S4"], s=68, facecolors="none", linewidths=1.1,
               edgecolors=COLORS["alert"], label="Alerta salvo · instante-alvo t + 30 min", zorder=4)
    ax.axhline(S4_THRESHOLD, color=COLORS["ink"], ls="--", lw=1)
    ax.set(ylabel="Índice S4", title=f"{inicio:%d/%m/%Y} a {fim:%d/%m/%Y} · limiar probabilístico salvo = {limiar_salvo(obs):g}")
    ax.margins(y=.18)
    ax.legend(ncols=3, loc="upper left", fontsize=9)
    eixo_tempo(ax)
    salvar(fig, pasta, "s4_cintilacao_real_alertas_30min")


def carregar_modelos(pasta: Path) -> list[dict]:
    metricas = pd.read_csv(pasta / "metricas_regressao.csv")
    modelos = []
    for _, row in metricas.iterrows():
        pred = pd.read_csv(pasta / f"predicoes_{row['modelo']}.csv", parse_dates=["datetime"])
        if pred["datetime"].duplicated().any():
            raise ValueError("Predições de regressão com épocas duplicadas.")
        modelos.append({"id": str(row.get("config_id", row["modelo"])), "nome": row["modelo"],
                        "filtrado": str(row["filtrado"]).lower() == "true", "predicoes": pred,
                        "RMSE": float(row["RMSE"]), "R2": float(row["R2"])})
    return modelos


def figuras_modelos(modelos: list[dict], serie: pd.DataFrame, pasta: Path, station: str) -> None:
    test = serie.loc[serie["datetime"].between(TEST_START, TEST_END)]
    for filtered in (False, True):
        subset = [m for m in modelos if m["filtrado"] == filtered]
        if not subset:
            continue
        branch = "filtrado" if filtered else "original"
        nrows = (len(subset) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(17, 3.3*nrows), sharex=True, sharey=True, layout="constrained", squeeze=False)
        fig.suptitle(f"{station} | VTEC observado × previsto · histórico {branch}\nTeste de {pd.Timestamp(TEST_START):%d/%m/%Y} · oito configurações do relatório", fontsize=18, fontweight="bold")
        for ax, model in zip(axes.flat, subset):
            pred = model["predicoes"]
            ax.scatter(test["datetime"], test["VTEC_observado"], s=18, color=COLORS["ink"], alpha=0.8, label="Mediana observada (sem interpolação)")
            if filtered:
                ax.plot(pred["datetime"], pred["observado"], color=COLORS["filtered"], lw=1.1, label="Alvo filtrado")
            ax.plot(pred["datetime"], pred["predito"], color=COLORS["pred"], lw=1.6, label="Previsão")
            flagged = test.loc[test["n_svid_filtrados"].gt(0)]
            ax.scatter(flagged["datetime"], flagged["VTEC_observado"], s=65, facecolors="none", edgecolors=COLORS["alert"], label="Época com SVID sinalizado")
            ax.set_title(f"{model['id']} · RMSE {model['RMSE']:.2f} TECU · R² {model['R2']:.3f}", loc="left")
            ax.set_ylabel("VTEC (TECU)")
            eixo_tempo(ax, diario=True)
        for ax in list(axes.flat)[len(subset):]:
            ax.set_visible(False)
        axes[0, 0].legend(fontsize=7, loc="upper left")
        salvar(fig, pasta, f"regressao_series_todos_{branch}")

        fig, axes = plt.subplots(nrows, 2, figsize=(12, 3.8*nrows), layout="constrained", squeeze=False)
        fig.suptitle(f"{station} | Dispersão observado × previsto · {branch}\nMétricas calculadas contra o alvo de cada ramo", fontsize=17, fontweight="bold")
        for ax, model in zip(axes.flat, subset):
            p = model["predicoes"].merge(test[["datetime", "n_svid_filtrados"]], on="datetime", validate="one_to_one")
            ax.scatter(p["observado"], p["predito"], s=24, color=COLORS["pred"], alpha=0.7)
            alert = p.loc[p["n_svid_filtrados"].gt(0)]
            ax.scatter(alert["observado"], alert["predito"], s=80, facecolors="none", edgecolors=COLORS["alert"])
            limits = [min(p["observado"].min(), p["predito"].min()), max(p["observado"].max(), p["predito"].max())]
            ax.plot(limits, limits, ls="--", color=COLORS["ink"], lw=1)
            ax.set(xlabel=f"Observado · {branch} (TECU)", ylabel="Previsto (TECU)", aspect="equal")
            ax.set_title(f"{model['id']} · n={len(p)} · RMSE {model['RMSE']:.2f} · R² {model['R2']:.3f}", loc="left")
        for ax in list(axes.flat)[len(subset):]:
            ax.set_visible(False)
        salvar(fig, pasta, f"regressao_dispersao_todos_{branch}")


def figuras_vtec_real(obs: pd.DataFrame, serie: pd.DataFrame, modelos: list[dict],
                      pasta: Path, station: str) -> None:
    """Observações em pontos e previsões em linhas, sem fabricar medições."""
    reais = obs.loc[obs["VTEC"].notna()]
    teste = reais.loc[reais["datetime"].between(TEST_START, TEST_END)]
    mediana = serie.loc[serie["datetime"].between(TEST_START, TEST_END)]
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), layout="constrained")
    fig.suptitle(f"{station} | Série temporal do VTEC observado", fontsize=19, fontweight="bold")
    for ax, dados, agregados, title in (
        (axes[0], reais, serie, "Histórico disponível · lacunas sem pontos"),
        (axes[1], teste, mediana, f"Detalhe do teste · {pd.Timestamp(TEST_START):%d/%m/%Y}"),
    ):
        ax.scatter(dados["datetime"], dados["VTEC"], s=7, color=COLORS["original"], alpha=.23,
                   label="VTEC observado por SVID · média em 30 min", rasterized=True)
        ax.scatter(agregados["datetime"], agregados["VTEC_observado"], s=15, color=COLORS["ink"],
                   alpha=.8, label="Mediana dos SVIDs observados · sem interpolação")
        ax.set_title(title, loc="left")
        ax.set_ylabel("VTEC (TECU)")
        ax.legend(loc="upper left", fontsize=9)
        eixo_tempo(ax, diario=ax is axes[1])
    salvar(fig, pasta, "vtec_real_serie_temporal")

    if not modelos:
        return

    colors = ["#2563b0", "#d46a2b", "#229879", "#a7509b", "#ad8d10", "#42a5ba", "#674fb2", "#8c6553"]
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True, sharey=True, layout="constrained")
    fig.suptitle(f"{station} | VTEC real em dispersão × previsões em linhas\nTeste de {pd.Timestamp(TEST_START):%d/%m/%Y}", fontsize=19, fontweight="bold")
    for ax, filtered in zip(axes, (False, True)):
        ax.scatter(teste["datetime"], teste["VTEC"], s=10, color=COLORS["original"], alpha=.25,
                   label="Real por SVID (30 min)", rasterized=True)
        ax.scatter(mediana["datetime"], mediana["VTEC_observado"], s=27, color=COLORS["ink"],
                   label="Mediana real (sem interpolação)", zorder=4)
        for color, model in zip(colors, [m for m in modelos if m["filtrado"] == filtered]):
            p = model["predicoes"]
            ax.plot(p["datetime"], p["predito"], lw=1.6, color=color, label=f"{model['id']} · previsão")
        ax.set_title("Treinamento com histórico " + ("filtrado" if filtered else "original"), loc="left")
        ax.set_ylabel("VTEC (TECU)")
        ax.legend(ncols=5, loc="upper left", fontsize=8)
        eixo_tempo(ax, diario=True)
    salvar(fig, pasta, "vtec_real_previsoes_teste")


def _rows(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient="values", date_format="iso", double_precision=12))


def gerar_painel(obs: pd.DataFrame, serie: pd.DataFrame, modelos: list[dict], pasta: Path, station: str,
                 metricas_sha256: dict | None = None) -> None:
    cols = ["datetime", "svid", "S4", "VTEC", "probabilidade_severa", "sinalizado_nucleo1", "origem_probabilidade"]
    data = {"station": station, "signature": assinatura(obs), "probThreshold": limiar_salvo(obs),
            "s4Threshold": S4_THRESHOLD, "testStart": TEST_START, "testEnd": TEST_END,
            "metricsSha256": metricas_sha256 or {}, "reviewAvailable": bool(metricas_sha256),
            "sampling": "Média em 30 min por SVID, já realizada nos arquivos de entrada; CS após a média.",
            "observations": _rows(obs[cols]),
            "scintillation": _rows(resumir_cintilacao(obs, serie)),
            "series": _rows(serie[["datetime", "VTEC_original", "VTEC_filtered", "n_svid_filtrados", "VTEC_observado"]]),
            "models": [{k: v for k, v in m.items() if k != "predicoes"} | {
                "points": _rows(m["predicoes"][["datetime", "observado", "predito"]])} for m in modelos]}
    template = Path(__file__).with_name("visual_analytics.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (pasta / "painel_visual_analytics.html").write_text(template.replace("__GNSS_DATA__", payload), encoding="utf-8")


def sha256_arquivo(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def verificar_metricas_salvas(results: Path) -> dict[str, str]:
    """A revisão só existe depois da avaliação científica dos dois núcleos."""
    paths = [results / "nucleo1/metricas_teste.json", results / "nucleo2/metricas_regressao.csv"]
    if not all(p.is_file() for p in paths):
        raise ValueError("Conclua e salve as métricas dos dois núcleos antes da análise/revisão visual.")
    classificacao = json.loads(paths[0].read_text(encoding="utf-8"))
    regressao = pd.read_csv(paths[1])
    if not classificacao or regressao.empty or not {"modelo", "filtrado", "RMSE", "R2"}.issubset(regressao):
        raise ValueError("As métricas dos dois núcleos estão incompletas.")
    return {str(path.relative_to(results)).replace("\\", "/"): sha256_arquivo(path) for path in paths}


def gerar(station: str = STATION, revisao_path: Path | None = None) -> Path:
    processed = Path(PROCESSED_DATA) / station
    results = Path(RESULTS_DIRECTORY) / station
    pasta = results / "relatorio" / ("revisao_manual" if revisao_path else "visual_analytics")
    metricas_sha256 = verificar_metricas_salvas(results)
    fontes = [processed / f"{station}_features.csv", processed / f"{station}_dataset_regressao.csv",
              results / "nucleo1/predicoes_xgboost.csv", *[results / nome for nome in metricas_sha256],
              *sorted((results / "nucleo2").glob("predicoes_*.csv"))]
    hashes_fontes = {str(path.resolve()): sha256_arquivo(path) for path in fontes}
    features = pd.read_csv(processed / f"{station}_features.csv", usecols=["datetime", "svid", "S4", "VTEC"])
    pred = pd.read_csv(results / "nucleo1/predicoes_xgboost.csv")
    obs = preparar_observacoes(features, pred)
    serie = pd.read_csv(processed / f"{station}_dataset_regressao.csv", parse_dates=["datetime"])
    serie = anexar_observado(serie, obs)
    contagem = obs.loc[obs["sinalizado_nucleo1"] & obs["VTEC"].notna()].groupby("datetime").size()
    esperado = contagem.reindex(pd.DatetimeIndex(serie["datetime"]), fill_value=0).to_numpy()
    if not np.array_equal(esperado, serie["n_svid_filtrados"].to_numpy()):
        raise ValueError("O limiar/predições atuais não correspondem ao dataset de regressão salvo.")
    modelos = carregar_modelos(results / "nucleo2")
    revisao = None
    if revisao_path:
        revisao = json.loads(revisao_path.read_text(encoding="utf-8"))
        if "metricas_sha256" in revisao and revisao["metricas_sha256"] != metricas_sha256:
            raise ValueError("As métricas salvas mudaram desde a exportação da revisão; abra o painel atualizado.")
        obs, serie = aplicar_revisao(obs, serie, revisao, station)
    pasta.mkdir(parents=True, exist_ok=True)
    estilo()
    figuras_historicas(obs, serie, pasta, station)
    figuras_s4_vtec(obs, serie, pasta, station)
    if revisao is None:
        figuras_modelos(modelos, serie, pasta, station)
        figuras_vtec_real(obs, serie, modelos, pasta, station)
        gerar_painel(obs, serie, modelos, pasta, station, metricas_sha256)
    else:
        (pasta / "decisoes_revisao.json").write_text(json.dumps(revisao, ensure_ascii=False, indent=2), encoding="utf-8")
        serie.to_csv(pasta / f"{station}_dataset_regressao_revisado.csv", index=False)
        obs.loc[obs["removido_manual"]].to_csv(pasta / "observacoes_removidas.csv", index=False)
        obs.loc[obs["falso_positivo_manual"]].to_csv(pasta / "falsos_positivos_restaurados.csv", index=False)
        obs.loc[obs["acao_revisao"].ne("")].to_csv(pasta / "auditoria_revisao.csv", index=False)
    obs.loc[obs["sinalizado_nucleo1"]].to_csv(pasta / "epocas_sinalizadas_nucleo1.csv", index=False)
    serie[["datetime", "VTEC_observado", "n_svid_observados"]].to_csv(pasta / "vtec_observado_30min.csv", index=False)
    resumir_cintilacao(obs, serie).to_csv(pasta / "s4_cs_observada_30min.csv", index=False)
    if any(sha256_arquivo(Path(path)) != digest for path, digest in hashes_fontes.items()):
        raise RuntimeError("As fontes científicas mudaram durante a análise visual; confira a execução concorrente.")
    manifest = {"estacao": station, "periodo_inicio": str(serie["datetime"].min()),
                "periodo_fim": str(serie["datetime"].max()), "assinatura_dados": assinatura(obs),
                "limiar_probabilidade": limiar_salvo(obs), "limiar_s4": S4_THRESHOLD,
                "reamostragem": "Média de S4/VTEC em 30 min por SVID na entrada; CS classificada após a média.",
                "observacoes_vtec": int(obs["VTEC"].notna().sum()),
                "epocas_sem_vtec_observado": int(serie["VTEC_observado"].isna().sum()),
                "sinalizacoes_com_vtec": int((obs["sinalizado_nucleo1"] & obs["VTEC"].notna()).sum()),
                "epocas_com_filtro": int(serie["n_svid_filtrados"].gt(0).sum()),
                "decisoes_manuais": len(decisoes_revisao(revisao)) if revisao else 0,
                "remocoes_manuais": int(obs["removido_manual"].sum()) if revisao else 0,
                "falsos_positivos_restaurados": int(obs["falso_positivo_manual"].sum()) if revisao else 0,
                "metricas_sha256": metricas_sha256, "fontes_sha256": hashes_fontes,
                "metricas_preservadas": True, "revisao_posterior_as_metricas": True,
                "modelos_retreinados": False}
    (pasta / "manifesto_visual_analytics.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Visual analytics salvo:", pasta)
    return pasta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revisao", type=Path, help="JSON de decisões exportado pelo painel")
    args, _ = parser.parse_known_args()
    gerar(revisao_path=args.revisao)
