"""Consolida artefatos recentes por estacao para revisao e redacao."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "resultados_artigo_organizados_20260922"
SOURCE_DIRECTORIES = (
    ROOT / "resultados",
    ROOT / "experimentos",
    ROOT / "artigo_resultados_20260921",
)
STATIONS = ("FRTZ", "PALM", "POAL", "PRU2")
ALLOWED_EXTENSIONS = None
SKIP_PARTS = {".git", ".venv", ".uv-cache", "__pycache__"}


def _station(path: Path) -> str:
    upper_parts = {part.upper() for part in path.parts}
    matches = [station for station in STATIONS if station in upper_parts or station in path.name.upper()]
    return matches[0] if matches else "GLOBAL"


def _category(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".keras", ".joblib", ".h5", ".pkl"}:
        return "modelos_pesos"
    if suffix in {".log", ".lock"} or "logs" in {part.lower() for part in path.parts}:
        return "logs_execucao"
    if suffix in {".py", ".ps1", ".sh"}:
        return "codigo_execucao"
    if suffix in {".ismr", ".gz", ".zip", ".feather", ".parquet"}:
        return "dados_brutos_processados"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "graficos_imagens"
    if suffix in {".csv", ".tex", ".xlsx"}:
        return "tabelas_dados"
    if suffix in {".md", ".html"}:
        return "relatorios_documentacao"
    if "metric" in name or "resumo" in name or "auditoria" in name or "manifest" in name:
        return "metricas_procedencia"
    return "configuracoes_json"


def _safe_destination(station: str, category: str, source: Path) -> Path:
    relative = source.relative_to(ROOT).with_suffix("")
    source_tag = "__".join(relative.parts).replace(" ", "_")
    return DESTINATION / station / category / f"{source_tag}{source.suffix}"


def main() -> None:
    cutoff = datetime.now() - timedelta(days=7)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    seen: set[Path] = set()

    for source_root in SOURCE_DIRECTORIES:
        if not source_root.is_dir():
            continue
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            if any(part in SKIP_PARTS for part in source.parts):
                continue
            if source in seen or datetime.fromtimestamp(source.stat().st_mtime) < cutoff:
                continue
            seen.add(source)
            station = _station(source)
            category = _category(source)
            destination = _safe_destination(station, category, source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination.unlink(missing_ok=True)
                destination.hardlink_to(source)
            except OSError:
                shutil.copy2(source, destination)
            records.append({
                "station": station,
                "category": category,
                "source": str(source.relative_to(ROOT)),
                "destination": str(destination.relative_to(ROOT)),
                "modified": datetime.fromtimestamp(source.stat().st_mtime).isoformat(timespec="seconds"),
                "bytes": source.stat().st_size,
            })

    manifest = DESTINATION / "INVENTARIO.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("station", "category", "source", "destination", "modified", "bytes"))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda item: (str(item["station"]), str(item["category"]), str(item["destination"]))))

    counts: dict[str, int] = {}
    for record in records:
        key = f"{record['station']}/{record['category']}"
        counts[key] = counts.get(key, 0) + 1
    readme = DESTINATION / "README.md"
    lines = [
        "# Resultados organizados para o relatorio",
        "",
        f"Arquivos modificados nos ultimos 7 dias, coletados em {datetime.now().isoformat(timespec='seconds')}.",
        "Os originais nao foram movidos ou apagados. O inventario completo esta em `INVENTARIO.csv`.",
        "",
        "## Estrutura",
        "",
        "Cada estacao possui `graficos_imagens`, `tabelas_dados`, `metricas_procedencia` e `relatorios_documentacao`.",
        "Nomes de arquivo incluem a origem relativa para evitar colisao entre experimentos.",
        "",
        "## Quantidades",
        "",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"- `{key}`: {count} arquivos")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Arquivos organizados: {len(records)}")
    print(f"Destino: {DESTINATION}")


if __name__ == "__main__":
    main()