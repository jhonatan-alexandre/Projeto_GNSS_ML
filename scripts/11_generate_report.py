"""Gera tabelas, gráficos e síntese dos dois núcleos por estação."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)

from config import (  # noqa: E402
    PROCESSED_DATA,
    RESULTS_DIRECTORY,
    STATION,
    STUDY_START,
    TEST_END,
)
from visual_analytics import estilo, gerar as gerar_visual_analytics  # noqa: E402


PASTA = Path(RESULTS_DIRECTORY) / STATION
NUCLEO1 = PASTA / "nucleo1"
NUCLEO2 = PASTA / "nucleo2"
VALIDACAO = NUCLEO1 / "validacao_temporal.csv"
METRICAS_TESTE = NUCLEO1 / "metricas_teste.json"
REGRESSAO = NUCLEO2 / "metricas_regressao.csv"
RESUMO_FILTRAGEM = Path(PROCESSED_DATA) / STATION / "resumo_filtragem_vtec.json"
SAIDA = PASTA / "relatorio"


def _normalizar_bool(serie: pd.Series, nome: str) -> pd.Series:
    if serie.dtype == bool:
        return serie
    normalizada = (
        serie.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if normalizada.isna().any():
        raise ValueError(f"A coluna '{nome}' contém valores booleanos inválidos.")
    return normalizada.astype(bool)


def _salvar_figura(fig: plt.Figure, nome: str) -> None:
    fig.tight_layout()
    for extensao in ("png", "svg", "pdf"):
        fig.savefig(SAIDA / f"{nome}.{extensao}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _rotulo_modelo(linha: pd.Series) -> str:
    arquitetura = "Att" if linha["atencao"] else "LSTM"
    entrada = "Multi" if linha["multivariado"] else "Uni"
    organizacao = "Piece" if linha["organizacao"] == "piecewise" else "Cont"
    return f"{organizacao}-{arquitetura}-{entrada}"


def comparar_metricas_regressao(regressao: pd.DataFrame) -> pd.DataFrame:
    """Compara RMSE e R² dos conjuntos original e tratado do relatório."""
    chaves = ["organizacao", "atencao", "multivariado"]
    comparacoes = []
    for metrica in ("RMSE", "R2"):
        tabela = (
            regressao.pivot_table(
                index=chaves,
                columns="filtrado",
                values=metrica,
            )
            .reset_index()
            .rename(
                columns={
                    False: f"{metrica}_original",
                    True: f"{metrica}_filtrado",
                }
            )
        )
        esperadas = {f"{metrica}_original", f"{metrica}_filtrado"}
        if not esperadas.issubset(tabela.columns):
            raise ValueError(
                f"São necessários resultados originais e filtrados para {metrica}."
            )
        comparacoes.append(tabela)

    comparacao = comparacoes[0].merge(comparacoes[1], on=chaves, validate="one_to_one")
    comparacao["reducao_percentual"] = 100 * (
        comparacao["RMSE_original"] - comparacao["RMSE_filtrado"]
    ) / comparacao["RMSE_original"]
    # R² é adimensional e pode ser negativo; a diferença absoluta é mais
    # interpretável que uma variação percentual.
    comparacao["delta_R2"] = comparacao["R2_filtrado"] - comparacao["R2_original"]
    return comparacao


def _graficos_classificacao(
    validacao: pd.DataFrame, classificacao: dict[str, float]
) -> None:
    metricas = ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "average_precision"]
    disponiveis = [nome for nome in metricas if classificacao.get(nome) is not None]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    valores = [classificacao[nome] for nome in disponiveis]
    barras = ax.bar(disponiveis, valores, color="#2463a5")
    ax.bar_label(barras, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Métrica")
    ax.set_title(f"{STATION} — desempenho final do XGBoost")
    ax.grid(axis="y", alpha=0.25)
    _salvar_figura(fig, "classificacao_metricas_teste")

    metricas_fold = [nome for nome in ("precision", "recall", "f1") if nome in validacao]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for nome in metricas_fold:
        ax.plot(validacao["fold"], validacao[nome], marker="o", label=nome)
    if "uso" in validacao:
        descritivos = validacao.loc[validacao["uso"].eq("selecao_hiperparametros_descritivo"), "fold"]
        independentes = validacao.loc[validacao["uso"].eq("validacao_independente"), "fold"]
        if not descritivos.empty:
            ax.axvspan(descritivos.min()-.35, descritivos.max()+.35, color="#b88919", alpha=.12,
                       label="Folds usados na seleção · descritivos")
        if not independentes.empty:
            ax.axvspan(independentes.min()-.35, independentes.max()+.35, color="#087e8b", alpha=.12,
                       label="Validação temporal independente")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fold temporal")
    ax.set_ylabel("Métrica")
    ax.set_title(f"{STATION} — estabilidade na validação temporal")
    ax.grid(alpha=0.25)
    ax.legend()
    _salvar_figura(fig, "classificacao_validacao_temporal")

    matriz = np.array(
        [
            [classificacao.get("tn", 0), classificacao.get("fp", 0)],
            [classificacao.get("fn", 0), classificacao.get("tp", 0)],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    imagem = ax.imshow(matriz, cmap="Blues")
    for (linha, coluna), valor in np.ndenumerate(matriz):
        ax.text(coluna, linha, f"{int(valor):,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Pred. calma", "Pred. severa"])
    ax.set_yticks([0, 1], ["Real calma", "Real severa"])
    ax.set_title(f"{STATION} — matriz de confusão do teste")
    fig.colorbar(imagem, ax=ax, fraction=0.046, pad=0.04)
    _salvar_figura(fig, "classificacao_matriz_confusao")
    _graficos_selecao_classificacao()


def _graficos_selecao_classificacao() -> None:
    """Exibe decisões de treino/validação sem sugerir reamostragem do teste."""
    hiper_path = NUCLEO1 / "hiperparametros_xgboost.json"
    if not hiper_path.is_file():
        return
    hiper = json.loads(hiper_path.read_text(encoding="utf-8"))
    amostragem = hiper.get("amostragem_treino", {})
    if amostragem:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
        fig.suptitle(f"{STATION} | Distribuição de classes no treinamento", fontsize=17, fontweight="bold")
        pesos = amostragem.get("balanceamento") == "pesos"
        for ax, sufixo, titulo in zip(axes, ("antes", "ajuste"), (
            "Antes do balanceamento", "Ajuste com pesos · contagens preservadas" if pesos else "Após reamostragem do treino")):
            valores = [amostragem[f"classe_{i}_{sufixo}"] for i in (0, 1)]
            bars = ax.bar(["Classe 0 · sem CS", "Classe 1 · CS"], valores, color=["#64748b", "#087e8b"], width=.6)
            ax.bar_label(bars, labels=[f"{v:,}" for v in valores], padding=4)
            ax.set_title(titulo, loc="left")
            ax.margins(y=.2)
            ax.set_ylabel("Amostras de treinamento")
        tecnica = (f"Pesos de classe: scale_pos_weight = {amostragem.get('scale_pos_weight', 1):.3f}; sem amostras sintéticas."
                   if pesos else f"Técnica: {amostragem.get('balanceamento', 'não informada')}; aplicada somente no treinamento.")
        fig.supxlabel(tecnica + " Validação e teste preservam a distribuição observada.", fontsize=9)
        _salvar_figura(fig, "classificacao_classes_antes_depois")

    curvas_path, busca_path = NUCLEO1 / "curvas_limiar_validacao.csv", NUCLEO1 / "busca_hiperparametros.csv"
    if not curvas_path.is_file() or not busca_path.is_file():
        return
    busca = pd.read_csv(busca_path)
    vencedor = busca.sort_values(["f2", "precision", "candidato"], ascending=[False, False, True]).iloc[0]
    curvas = pd.read_csv(curvas_path)
    curva = curvas.loc[curvas["candidato"].eq(vencedor["candidato"])].sort_values("limiar")
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for metrica, cor in (("precision", "#2563b0"), ("recall", "#087e8b"), ("f2", "#9333a8")):
        ax.plot(curva["limiar"], curva[metrica], color=cor, lw=2, label=metrica)
    ax.axvline(hiper["limiar_probabilidade"], color="#dc5b32", ls="--", lw=1.5,
               label=f"Corte selecionado = {hiper['limiar_probabilidade']:.3f}")
    ax.axvline(.75, color="#64748b", ls=":", lw=1.5, label="Referência fixa = 0,75")
    ax.set(xlabel="Limiar de probabilidade", ylabel="Métrica na seleção temporal", ylim=(0, 1.05),
           title=f"{STATION} | Precisão, recall e F2 na escolha do limiar")
    ax.legend(ncols=2, fontsize=9)
    fig.supxlabel("Folds usados na seleção: curvas descritivas. O último fold e o teste ficam fora desta escolha.", fontsize=9)
    _salvar_figura(fig, "classificacao_selecao_limiar")


def _graficos_regressao(
    regressao: pd.DataFrame, comparacao: pd.DataFrame
) -> None:
    comparacao = comparacao.copy()
    comparacao["rotulo"] = comparacao.apply(_rotulo_modelo, axis=1)
    x = np.arange(len(comparacao))
    largura = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - largura / 2,
        comparacao["RMSE_original"],
        largura,
        label="Histórico original",
        color="#777777",
    )
    ax.bar(
        x + largura / 2,
        comparacao["RMSE_filtrado"],
        largura,
        label="Histórico filtrado",
        color="#1f9d8a",
    )
    ax.set_xticks(x, comparacao["rotulo"], rotation=35, ha="right")
    ax.set_ylabel("RMSE (TECU)")
    ax.set_title(f"{STATION} — comparação do erro de previsão do VTEC")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _salvar_figura(fig, "regressao_comparacao_rmse")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - largura / 2,
        comparacao["R2_original"],
        largura,
        label="Histórico original",
        color="#777777",
    )
    ax.bar(
        x + largura / 2,
        comparacao["R2_filtrado"],
        largura,
        label="Histórico filtrado",
        color="#1f9d8a",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, comparacao["rotulo"], rotation=35, ha="right")
    ax.set_ylabel("R²")
    ax.set_title(f"{STATION} — coeficiente de determinação da previsão do VTEC")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _salvar_figura(fig, "regressao_comparacao_r2")

    cores = np.where(comparacao["reducao_percentual"] >= 0, "#1f9d8a", "#c84b4b")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(comparacao["rotulo"], comparacao["reducao_percentual"], color=cores)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, comparacao["rotulo"], rotation=35, ha="right")
    ax.set_ylabel("Redução do RMSE (%)")
    ax.set_title(f"{STATION} — efeito do histórico filtrado")
    ax.grid(axis="y", alpha=0.25)
    _salvar_figura(fig, "regressao_reducao_rmse")

    observacoes = pd.read_csv(
        Path(PROCESSED_DATA) / STATION / f"{STATION}_features.csv",
        usecols=["datetime", "VTEC"], parse_dates=["datetime"],
    ).groupby("datetime")["VTEC"].median()
    fig, eixos = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for eixo, filtrado, titulo in zip(
        eixos,
        (False, True),
        ("Melhor modelo com histórico original", "Melhor modelo com histórico filtrado"),
    ):
        melhor = regressao.loc[regressao["filtrado"] == filtrado].nsmallest(1, "RMSE")
        if melhor.empty:
            continue
        nome = melhor.iloc[0]["modelo"]
        caminho = NUCLEO2 / f"predicoes_{nome}.csv"
        if not caminho.is_file():
            raise FileNotFoundError(f"Predições ausentes para o melhor modelo: {caminho}")
        predicoes = pd.read_csv(caminho, parse_dates=["datetime"])
        reais = observacoes.reindex(pd.DatetimeIndex(predicoes["datetime"]))
        eixo.scatter(predicoes["datetime"], reais, label="VTEC real (mediana observada)", s=24, color="#183149")
        if filtrado:
            eixo.plot(predicoes["datetime"], predicoes["observado"], label="Alvo filtrado", lw=1.2, color="#087e8b")
        eixo.plot(predicoes["datetime"], predicoes["predito"], label="Predito", lw=1.2)
        eixo.set_title(
            f"{titulo}: {nome} "
            f"(RMSE={melhor.iloc[0]['RMSE']:.3f}; R²={melhor.iloc[0]['R2']:.3f})"
        )
        eixo.set_ylabel("VTEC (TECU)")
        eixo.grid(alpha=0.25)
        eixo.legend()
    eixos[-1].set_xlabel("Data/hora")
    _salvar_figura(fig, "regressao_melhores_series")


def main() -> None:
    estilo()
    for caminho in (VALIDACAO, METRICAS_TESTE, REGRESSAO):
        if not caminho.is_file():
            raise FileNotFoundError(f"Resultado ausente: {caminho}")
    SAIDA.mkdir(parents=True, exist_ok=True)
    validacao = pd.read_csv(VALIDACAO)
    regressao = pd.read_csv(REGRESSAO)
    for coluna in ("filtrado", "atencao", "multivariado"):
        regressao[coluna] = _normalizar_bool(regressao[coluna], coluna)
    if "alvo_avaliacao" in regressao.columns:
        alvo_esperado = regressao["filtrado"].map(
            {False: "VTEC_original", True: "VTEC_filtered"}
        )
        if not regressao["alvo_avaliacao"].eq(alvo_esperado).all():
            raise ValueError(
                "O ramo original deve avaliar VTEC_original e o ramo filtrado, "
                "VTEC_filtered."
            )
    with open(METRICAS_TESTE, "r", encoding="utf-8") as arquivo:
        classificacao = json.load(arquivo)

    comparacao = comparar_metricas_regressao(regressao)
    comparacao.to_csv(SAIDA / "comparacao_regressao.csv", index=False)
    (SAIDA / "comparacao_regressao.tex").write_text(
        comparacao.to_latex(index=False, float_format="%.4f"), encoding="utf-8"
    )
    # Mantém os nomes históricos para compatibilidade com a consolidação.
    comparacao.to_csv(SAIDA / "comparacao_rmse.csv", index=False)
    (SAIDA / "comparacao_rmse.tex").write_text(
        comparacao.to_latex(index=False, float_format="%.4f"), encoding="utf-8"
    )

    resumo = {
        "estacao": STATION,
        "periodo": {"inicio": STUDY_START, "fim": TEST_END},
        "classificacao_teste": classificacao,
        "f1_validacao_medio": float(validacao["f1"].mean()),
        "uso_f1_validacao_medio": "descritivo; inclui folds de seleção quando presentes",
        "configuracoes_com_reducao_rmse": int(
            (comparacao["reducao_percentual"] > 0).sum()
        ),
        "total_configuracoes": len(comparacao),
        "melhor_reducao_rmse_percentual": float(comparacao["reducao_percentual"].max()),
        "configuracoes_com_aumento_R2": int((comparacao["delta_R2"] > 0).sum()),
        "melhor_R2_original": float(regressao.loc[~regressao["filtrado"], "R2"].max()),
        "melhor_R2_filtrado": float(regressao.loc[regressao["filtrado"], "R2"].max()),
        "maior_delta_R2": float(comparacao["delta_R2"].max()),
    }
    if "uso" in validacao:
        independente = validacao.loc[validacao["uso"].eq("validacao_independente")]
        if not independente.empty:
            resumo["f1_validacao_independente"] = float(independente["f1"].mean())
    if RESUMO_FILTRAGEM.is_file():
        with open(RESUMO_FILTRAGEM, "r", encoding="utf-8") as arquivo:
            resumo["filtragem"] = json.load(arquivo)
    with open(SAIDA / "resumo.json", "w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)

    _graficos_classificacao(validacao, classificacao)
    _graficos_regressao(regressao, comparacao)
    gerar_visual_analytics()
    (SAIDA / "resumo_metodologico.md").write_text(
        f"# Resultados — {STATION}\n\n"
        f"Período: {STUDY_START} a {TEST_END}.\n\n"
        f"F1 no teste: {classificacao['f1']:.4f}. "
        f"Recall: {classificacao['recall']:.4f}. "
        f"Precisão: {classificacao['precision']:.4f}.\n\n"
        f"A filtragem reduziu o RMSE em "
        f"{resumo['configuracoes_com_reducao_rmse']} de "
        f"{resumo['total_configuracoes']} configurações.\n\n"
        f"O R² aumentou em {resumo['configuracoes_com_aumento_R2']} de "
        f"{resumo['total_configuracoes']} configurações; os melhores valores "
        f"foram {resumo['melhor_R2_original']:.4f} no histórico original e "
        f"{resumo['melhor_R2_filtrado']:.4f} no filtrado.\n\n"
        "S4 e VTEC reais são médias observadas em 30 minutos por SVID, exibidas em dispersão. "
        "CS observada é definida após essa média; curvas agregadas podem incluir interpolação. "
        "As previsões dos modelos são linhas.\n\n"
        "A seleção de hiperparâmetros usa somente os folds declarados para seleção; suas métricas "
        "são descritivas. O último fold reservado fornece validação independente quando habilitada.\n\n"
        "A revisão manual é posterior às métricas dos dois núcleos. Retirar um falso alerta restaura "
        "o VTEC na série derivada; excluir uma medição é uma ação separada. Dados de treinamento, "
        "predições e métricas originais permanecem preservados e identificados por SHA-256.\n",
        encoding="utf-8",
    )
    print("Relatório, tabelas e gráficos salvos:", SAIDA)


if __name__ == "__main__":
    main()
