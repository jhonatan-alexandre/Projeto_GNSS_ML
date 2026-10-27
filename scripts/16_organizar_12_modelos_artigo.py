"""Organiza a suite limpa M1-M12 por estacao, sem comparacoes de protocolo."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "estudo_20240328" / "resultados" / "convlstm_4"
DESTINATION = ROOT / "resultados_artigo_12_modelos_20260922"
STATIONS = ("FRTZ", "PALM", "POAL", "PRU2")
MODELS = tuple(f"M{index}" for index in range(1, 13))
VARIANTS = ("original", "filtrado")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(SOURCE / "metricas_todos_modelos.csv")
    expected = {(station, model, variant) for station in ("FRTZ", "PALM") for model in MODELS for variant in VARIANTS}
    available = set(zip(metrics.station, metrics.model, metrics.variant))
    if available != expected:
        missing = sorted(expected - available)
        raise ValueError(f"Suite M1-M12 incompleta na fonte: {missing}")

    inventory: list[dict[str, str | int | float]] = []
    sequence_rows: list[dict[str, str | int | float]] = []
    for row in metrics.itertuples(index=False):
        source = Path(str(row.source))
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = DESTINATION / row.station / row.model / row.variant / "predicoes.csv"
        link_or_copy(source, destination)
        record = {
            "station": row.station,
            "model": row.model,
            "variant": row.variant,
            "n": int(row.n),
            "RMSE": float(row.RMSE),
            "MAE": float(row.MAE),
            "R2": float(row.R2),
            "source": str(source.relative_to(ROOT)),
            "organized": str(destination.relative_to(ROOT)),
        }
        sequence_rows.append(record)
        inventory.append({**record, "kind": "prediction"})

    for station in ("FRTZ", "PALM"):
        for name in (f"{station}_modelos.png", f"{station}_modelos.pdf", f"{station}_modelos.svg"):
            source = SOURCE / "figuras" / name
            if source.is_file():
                destination = DESTINATION / station / "figuras" / name
                link_or_copy(source, destination)
                inventory.append({"station": station, "model": "M1-M12", "variant": "ambos", "kind": "figure", "source": str(source.relative_to(ROOT)), "organized": str(destination.relative_to(ROOT))})
        for model in ("M9", "M10", "M11", "M12"):
            for extension in ("png", "pdf", "svg"):
                source = SOURCE / "figuras" / f"{station}_{model}_dia.{extension}"
                if source.is_file():
                    destination = DESTINATION / station / "figuras" / source.name
                    link_or_copy(source, destination)
                    inventory.append({"station": station, "model": model, "variant": "ambos", "kind": "figure", "source": str(source.relative_to(ROOT)), "organized": str(destination.relative_to(ROOT))})

    for name in ("metricas_todos_modelos.csv", "previsoes_todos_modelos.csv", "filtragem_todos_modelos.csv", "hiperparametros_finais.csv", "particoes_reajuste_integral.csv", "auditoria.json", "LEGENDAS.md", "CATALOGO_FIGURAS_20240328.pdf"):
        source = SOURCE / name
        if source.is_file():
            destination = DESTINATION / "tabelas_e_auditoria" / name
            link_or_copy(source, destination)
            inventory.append({"station": "GLOBAL", "model": "M1-M12", "variant": "ambos", "kind": "table_or_audit", "source": str(source.relative_to(ROOT)), "organized": str(destination.relative_to(ROOT))})

    status = []
    for station in STATIONS:
        complete = station in ("FRTZ", "PALM")
        status.append({
            "station": station,
            "models_expected": 12,
            "models_available": 12 if complete else 0,
            "status": "completo" if complete else "pendente",
            "note": "M1-M12 no protocolo vigente de 28/03/2024" if complete else "Sem suite M1-M12 vigente; nao misturar resultados historicos de outro protocolo",
        })

    pd.DataFrame(sequence_rows).sort_values(["station", "model", "variant"]).to_csv(DESTINATION / "sequencia_M1_M12.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(status).to_csv(DESTINATION / "status_por_estacao.csv", index=False, encoding="utf-8-sig")
    with (DESTINATION / "INVENTARIO.csv").open("w", encoding="utf-8", newline="") as file:
        fields = ("station", "model", "variant", "kind", "source", "organized")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in fields} for item in inventory)

    (DESTINATION / "README.md").write_text(
        "# Suite limpa de 12 modelos\n\n"
        "Esta pasta nao contem comparacoes de execucoes Inicial/Xie. A sequencia e M1-M12, "
        "com os quatro modelos ConvLSTM M9-M12, organizada por estacao, modelo e variante.\n\n"
        "FRTZ e PALM possuem a suite completa no protocolo vigente de 28/03/2024. POAL e PRU2 "
        "permanecem pendentes; seus resultados historicos nao foram misturados.\n\n"
        "Use `sequencia_M1_M12.csv` para a tabela principal, `status_por_estacao.csv` para declarar cobertura "
        "e `tabelas_e_auditoria/` para metricas, previsoes e procedencia.\n",
        encoding="utf-8",
    )
    print(f"Suite M1-M12 organizada em: {DESTINATION}")
    print(f"Previsoes organizadas: {len(sequence_rows)} arquivos")


if __name__ == "__main__":
    main()