"""Gera figuras de metodo para apresentar o protocolo de testes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resultados_artigo_12_modelos_20260922" / "documentacao" / "figuras_metodo"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, xy, text, color, width=3.3, height=0.82, fontsize=10):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.2, edgecolor="#263238", facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, wrap=True)
    return patch


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color="#37474F",
    ))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flow_test():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 7); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Fluxo experimental e separação do teste", fontsize=16, weight="bold", pad=18)
    items = [
        ("Dados ISMR + índices\nsolares/geomagnéticos", "#DCEFF2", 8.65),
        ("Controle de qualidade\ne engenharia de atributos", "#DCEFF2", 7.25),
        ("Desenvolvimento\n(treino + validação temporal)", "#FFF0C2", 5.85),
        ("Seleção de hiperparâmetros\ne épocas", "#FFF0C2", 4.45),
        ("Reajuste final\ncom todo o desenvolvimento", "#CDECCF", 3.05),
        ("Teste congelado\nsem ajuste posterior", "#F6D0D0", 1.65),
    ]
    for text, color, y in items:
        box(ax, (1.85, y), text, color)
    for upper, lower in zip(items[:-1], items[1:]):
        arrow(ax, (3.5, lower[2] + 0.82), (3.5, upper[2]))
    ax.text(5.35, 4.8, "Nenhum dado do teste\nparticipa da escolha", ha="center", va="center",
            fontsize=9, color="#8E2424", bbox={"boxstyle": "round", "facecolor": "#FFF7F7", "edgecolor": "#D55E5E"})
    arrow(ax, (4.95, 4.95), (5.05, 2.1))
    save(fig, "01_fluxo_teste")


def hyperparameters():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Estratégia de escolha de hiperparâmetros", fontsize=16, weight="bold", pad=18)
    box(ax, (0.4, 4.7), "Candidatas\nM1–M8 / M9–M12", "#DCEFF2", width=2.5)
    box(ax, (3.5, 4.7), "Ajuste temporal\nsem o teste", "#FFF0C2", width=2.5)
    box(ax, (6.6, 4.7), "Early stopping\nna validação", "#FFF0C2", width=2.5)
    box(ax, (9.7, 4.7), "Seleção da melhor\ncandidata", "#FFF0C2", width=2.0)
    arrow(ax, (2.9, 5.1), (3.5, 5.1)); arrow(ax, (6.0, 5.1), (6.6, 5.1)); arrow(ax, (9.1, 5.1), (9.7, 5.1))
    box(ax, (2.0, 2.8), "Congelamento\nde escolhas", "#CDECCF", width=2.5)
    box(ax, (5.2, 2.8), "Novo ajuste final\ncom todo desenvolvimento", "#CDECCF", width=3.0)
    box(ax, (8.8, 2.8), "Avaliação no dia\nde teste", "#F6D0D0", width=2.5)
    arrow(ax, (10.7, 4.7), (3.2, 3.62)); arrow(ax, (4.5, 3.2), (5.2, 3.2)); arrow(ax, (8.2, 3.2), (8.8, 3.2))
    ax.text(6, 1.25, "Regra: o teste mede desempenho; não escolhe limiar, arquitetura, taxa ou época.",
            ha="center", va="center", fontsize=11, weight="bold", color="#263238")
    save(fig, "02_selecao_hiperparametros")


def threshold():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Escolha do limiar de cintilação", fontsize=16, weight="bold", pad=18)
    box(ax, (0.4, 4.7), "XGBoost produz\nP(S4 severo)", "#DCEFF2", width=2.5)
    box(ax, (3.5, 4.7), "Curva precisão–recall\nna validação temporal", "#FFF0C2", width=3.0)
    box(ax, (7.2, 4.7), "Maximizar F2\npriorizar recall", "#FFF0C2", width=2.5)
    box(ax, (10.2, 4.7), "Limiar congelado", "#CDECCF", width=1.5, fontsize=9)
    arrow(ax, (2.9, 5.1), (3.5, 5.1)); arrow(ax, (6.5, 5.1), (7.2, 5.1)); arrow(ax, (9.7, 5.1), (10.2, 5.1))
    box(ax, (2.0, 2.55), "Teste observado\nsem retuning", "#F6D0D0", width=2.7)
    box(ax, (5.3, 2.55), "Classe severa /\nalerta de filtragem", "#DCEFF2", width=2.7)
    box(ax, (8.6, 2.55), "VTEC original\nversus filtrado", "#CDECCF", width=2.7)
    arrow(ax, (10.95, 4.7), (3.35, 3.37)); arrow(ax, (4.7, 2.96), (5.3, 2.96)); arrow(ax, (8.0, 2.96), (8.6, 2.96))
    ax.text(6, 1.1, "Trade-off: limiar menor -> maior recall e mais falsos positivos; limiar maior -> maior precisão e risco de perder eventos.",
            ha="center", va="center", fontsize=10, color="#263238")
    save(fig, "03_escolha_limiar_cintilacao")


def temporal_partitions():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title("Particionamento temporal sem vazamento do teste", fontsize=16, weight="bold", pad=18)
    segments = [
        (0, 63, "Ajuste\ndesenvolvimento", "#CDECCF"),
        (63, 76, "Early stopping\nvalidação", "#FFF0C2"),
        (76, 88, "Seleção\nde candidata", "#FFE1A8"),
        (88, 90, "Gap", "#E0E0E0"),
        (90, 100, "Teste congelado\nsem retuning", "#F6D0D0"),
    ]
    for start, end, label, color in segments:
        box(ax, (start, 1.9), label, color, width=end-start, height=0.9, fontsize=10)
    ax.text(0, 1.35, "Tempo →", fontsize=10, weight="bold")
    ax.text(50, 0.75, "A ordem temporal é preservada; nenhuma observação do teste participa da escolha.",
            ha="center", fontsize=11, weight="bold", color="#263238")
    ax.text(6, 3.25, "Parâmetros e escalas ajustados somente no passado", fontsize=9)
    ax.annotate("", xy=(88, 2.35), xytext=(0, 2.35), arrowprops={"arrowstyle": "-[", "lw": 1.2})
    save(fig, "04_particoes_temporais")


def model_matrix():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Matriz de modelos e entradas", fontsize=16, weight="bold", pad=18)
    headers = ["Modelo", "Organização", "Atenção", "Entrada", "Exógenos", "Objetivo"]
    xs = [0.1, 1.0, 2.35, 3.15, 4.0, 5.0]
    widths = [0.75, 1.25, 0.75, 0.8, 0.9, 0.9]
    for x, w, h in zip(xs, widths, headers): box(ax, (x, 7.1), h, "#263238", width=w, height=.55, fontsize=8); ax.text(x+w/2, 7.37, h, color="white", ha="center", va="center", fontsize=8)
    rows = []
    for i in range(1, 13):
        if i <= 4: organization = "Conv."; attention = "Não" if i in (1,2) else "Sim"
        elif i <= 8: organization = "Piecewise"; attention = "Não" if i in (5,6) else "Sim"
        else: organization = "ConvLSTM" if i in (9,10) else "Piecewise ConvLSTM"; attention = "N/A"
        entry = "Uni" if i in (1,3,5,7,10,11) else "Multi"
        exog = "Não" if entry == "Uni" else "Kp...SSN"
        objective = "VTEC"
        rows.append((f"M{i}", organization, attention, entry, exog, objective))
    for row_index, row in enumerate(rows):
        y = 6.35 - row_index * .48
        color = "#DCEFF2" if row_index % 2 == 0 else "#F7F9FA"
        for x, w, value in zip(xs, widths, row): box(ax, (x, y), value, color, width=w, height=.4, fontsize=7)
    ax.text(3, 0.35, "Uni = somente VTEC; Multi = VTEC + Kp, Dst, F107, Ap e SSN.", ha="center", fontsize=10)
    save(fig, "05_matriz_modelos_M1_M12")


def decision_metrics():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.8); ax.axis("off")
    ax.set_title("Critérios de decisão em cada núcleo", fontsize=16, weight="bold", pad=18)
    box(ax, (0.4, 4.25), "Núcleo 1\nXGBoost", "#DCEFF2", width=2.4)
    box(ax, (3.3, 4.25), "Escolha do limiar\nF2 + validação temporal", "#FFF0C2", width=3.0)
    box(ax, (6.8, 4.25), "Saída\nprobabilidade/classe", "#CDECCF", width=2.1)
    box(ax, (9.4, 4.25), "Entrada do\nNúcleo 2", "#DCEFF2", width=2.0)
    arrow(ax, (2.8, 4.65), (3.3, 4.65)); arrow(ax, (6.3, 4.65), (6.8, 4.65)); arrow(ax, (8.9, 4.65), (9.4, 4.65))
    box(ax, (1.0, 1.9), "Núcleo 2\nLSTM / ConvLSTM", "#DCEFF2", width=2.5)
    box(ax, (4.2, 1.9), "Seleção por RMSE\nno desenvolvimento", "#FFF0C2", width=2.8)
    box(ax, (7.6, 1.9), "Métricas finais\nMAE · RMSE · R²", "#CDECCF", width=2.7)
    box(ax, (10.1, 1.9), "Relatório\ne figuras", "#E7D8F4", width=1.5, fontsize=9)
    arrow(ax, (10.4, 4.25), (2.2, 2.72)); arrow(ax, (3.5, 2.3), (4.2, 2.3)); arrow(ax, (7.0, 2.3), (7.6, 2.3)); arrow(ax, (10.3, 2.3), (10.1, 2.3))
    ax.text(6, .75, "Precisão/recall avaliam detecção; MAE/RMSE/R² avaliam previsão contínua de VTEC.", ha="center", fontsize=10, weight="bold")
    save(fig, "06_criterios_decisao_metricas")


if __name__ == "__main__":
    flow_test(); hyperparameters(); threshold(); temporal_partitions(); model_matrix(); decision_metrics()
    print("Figuras de metodo salvas em:", OUT)