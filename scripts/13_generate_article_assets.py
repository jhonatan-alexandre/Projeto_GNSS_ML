"""Consolida tabelas e figuras multiestação para artigo ou relatório."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import sys

_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from config import (  # noqa: E402
    RESULTS_DIRECTORY,
    STATIONS,
    STUDY_START,
    TEST_END,
    TEST_START,
    TRAIN_END,
)


RESULTADOS = Path(RESULTS_DIRECTORY)
ENTRADA = RESULTADOS / "comparacao_estacoes.csv"
SAIDA = RESULTADOS / "artigo"


def _bool(serie: pd.Series) -> pd.Series:
    if serie.dtype == bool:
        return serie
    resultado = (
        serie.astype("string")
        .str.lower()
        .str.strip()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if resultado.isna().any():
        raise ValueError("Valor booleano inválido na tabela consolidada.")
    return resultado.astype(bool)


def _rotulo(linha: pd.Series) -> str:
    return "-".join(
        [
            "Piece" if linha["organizacao"] == "piecewise" else "Cont",
            "Att" if linha["atencao"] else "LSTM",
            "Multi" if linha["multivariado"] else "Uni",
        ]
    )


def _salvar(fig: plt.Figure, nome: str) -> None:
    fig.tight_layout()
    for extensao in ("png", "svg"):
        fig.savefig(SAIDA / f"{nome}.{extensao}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _versoes() -> dict[str, str]:
    resultado = {}
    for pacote in (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "imbalanced-learn",
        "tensorflow",
        "matplotlib",
    ):
        try:
            resultado[pacote] = version(pacote)
        except PackageNotFoundError:
            resultado[pacote] = "não instalado"
    return resultado


def main() -> None:
    if not ENTRADA.is_file():
        raise FileNotFoundError(f"Execute primeiro o pipeline multiestação: {ENTRADA}")
    SAIDA.mkdir(parents=True, exist_ok=True)
    dados = pd.read_csv(ENTRADA)
    colunas_regressao = {
        "RMSE_original",
        "RMSE_filtrado",
        "R2_original",
        "R2_filtrado",
        "delta_R2",
    }
    ausentes = colunas_regressao.difference(dados.columns)
    if ausentes:
        raise ValueError(
            "Métricas de regressão ausentes na consolidação: "
            + ", ".join(sorted(ausentes))
        )
    for coluna in ("atencao", "multivariado"):
        dados[coluna] = _bool(dados[coluna])
    dados["configuracao"] = dados.apply(_rotulo, axis=1)
    dados.to_csv(SAIDA / "tabela_completa_resultados.csv", index=False)
    (SAIDA / "tabela_completa_resultados.tex").write_text(
        dados.to_latex(index=False, float_format="%.4f"), encoding="utf-8"
    )

    por_estacao = (
        dados.groupby("estacao", as_index=False)
        .agg(
            RMSE_original_medio=("RMSE_original", "mean"),
            RMSE_filtrado_medio=("RMSE_filtrado", "mean"),
            R2_original_medio=("R2_original", "mean"),
            R2_filtrado_medio=("R2_filtrado", "mean"),
            delta_R2_medio=("delta_R2", "mean"),
            reducao_media_percentual=("reducao_percentual", "mean"),
            configuracoes_com_melhora=("reducao_percentual", lambda x: int((x > 0).sum())),
            configuracoes_com_aumento_R2=("delta_R2", lambda x: int((x > 0).sum())),
        )
        .sort_values("estacao")
    )
    por_estacao.to_csv(SAIDA / "tabela_resumo_estacoes.csv", index=False)
    (SAIDA / "tabela_resumo_estacoes.tex").write_text(
        por_estacao.to_latex(index=False, float_format="%.4f"), encoding="utf-8"
    )

    x = np.arange(len(por_estacao))
    largura = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        x - largura / 2,
        por_estacao["RMSE_original_medio"],
        largura,
        label="Original",
        color="#777777",
    )
    ax.bar(
        x + largura / 2,
        por_estacao["RMSE_filtrado_medio"],
        largura,
        label="Filtrado",
        color="#1f9d8a",
    )
    ax.set_xticks(x, por_estacao["estacao"])
    ax.set_ylabel("RMSE médio (TECU)")
    ax.set_title("Comparação multiestação da previsão de VTEC")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _salvar(fig, "rmse_medio_por_estacao")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        x - largura / 2,
        por_estacao["R2_original_medio"],
        largura,
        label="Original",
        color="#777777",
    )
    ax.bar(
        x + largura / 2,
        por_estacao["R2_filtrado_medio"],
        largura,
        label="Filtrado",
        color="#1f9d8a",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, por_estacao["estacao"])
    ax.set_ylabel("R² médio")
    ax.set_title("Coeficiente de determinação médio da previsão de VTEC")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _salvar(fig, "r2_medio_por_estacao")

    matriz = dados.pivot(
        index="configuracao", columns="estacao", values="reducao_percentual"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    imagem = ax.imshow(matriz.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(matriz.columns)), matriz.columns)
    ax.set_yticks(np.arange(len(matriz.index)), matriz.index)
    for (linha, coluna), valor in np.ndenumerate(matriz.to_numpy()):
        ax.text(coluna, linha, f"{valor:.1f}%", ha="center", va="center", fontsize=8)
    ax.set_title("Redução do RMSE pelo histórico filtrado")
    fig.colorbar(imagem, ax=ax, label="Redução (%)")
    _salvar(fig, "mapa_reducao_rmse_configuracoes")

    matriz_r2 = dados.pivot(
        index="configuracao", columns="estacao", values="delta_R2"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    limite = max(float(np.nanmax(np.abs(matriz_r2.to_numpy()))), 1e-12)
    imagem = ax.imshow(
        matriz_r2.to_numpy(),
        cmap="RdYlGn",
        aspect="auto",
        vmin=-limite,
        vmax=limite,
    )
    ax.set_xticks(np.arange(len(matriz_r2.columns)), matriz_r2.columns)
    ax.set_yticks(np.arange(len(matriz_r2.index)), matriz_r2.index)
    for (linha, coluna), valor in np.ndenumerate(matriz_r2.to_numpy()):
        ax.text(coluna, linha, f"{valor:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Variação do R² pelo histórico filtrado")
    fig.colorbar(imagem, ax=ax, label="ΔR² (filtrado − original)")
    _salvar(fig, "mapa_delta_r2_configuracoes")

    resumos_classificacao = []
    for estacao in STATIONS:
        caminho = RESULTADOS / estacao / "relatorio" / "resumo.json"
        if not caminho.is_file():
            raise FileNotFoundError(f"Resumo ausente para {estacao}: {caminho}")
        with open(caminho, "r", encoding="utf-8") as arquivo:
            resumo = json.load(arquivo)
        resumos_classificacao.append(
            {"estacao": estacao, **resumo["classificacao_teste"]}
        )
    classificacao = pd.DataFrame(resumos_classificacao)
    classificacao.to_csv(SAIDA / "tabela_classificacao_teste.csv", index=False)
    (SAIDA / "tabela_classificacao_teste.tex").write_text(
        classificacao.to_latex(index=False, float_format="%.4f"), encoding="utf-8"
    )

    metadados = {
        "estacoes": list(STATIONS),
        "periodo": {
            "inicio": STUDY_START,
            "fim_treino": TRAIN_END,
            "inicio_teste": TEST_START,
            "fim_teste": TEST_END,
        },
        "alvos_regressao": {
            "historico_original": "VTEC_original",
            "historico_filtrado": "VTEC_filtered",
        },
        "metricas_regressao": ["MAE", "MSE", "RMSE", "R2"],
        "configuracoes_por_estacao": int(dados.groupby("estacao").size().min()),
        "versoes": _versoes(),
    }
    (SAIDA / "metadados_experimento.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (SAIDA / "LEIA-ME.md").write_text(
        "# Pacote de resultados para publicação\n\n"
        f"Estações: {', '.join(STATIONS)}.\n\n"
        f"Treino até {TRAIN_END}; teste de {TEST_START} a {TEST_END}.\n\n"
        "As comparações usam `VTEC_original` no ramo original e "
        "`VTEC_filtered` no ramo submetido à filtragem, conforme o relatório, "
        "e apresentam RMSE e R². "
        "Arquivos PNG são destinados à "
        "visualização e SVG à edição vetorial. As tabelas CSV preservam os dados "
        "e as tabelas TEX podem ser incorporadas ao artigo.\n",
        encoding="utf-8",
    )
    print("Pacote para artigo/relatório salvo:", SAIDA)


if __name__ == "__main__":
    main()
