"""Reexecuta o fluxo completo por estação e compara com uma cópia dos resultados anteriores.

Não importa config.py: a configuração científica é resolvida somente nos filhos,
com ambiente GNSS limpo. --preflight-only nunca inicia extração ou treinamento.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import html
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = PROJECT_ROOT / "scripts"
STATIONS = ("FRTZ", "PALM", "PRU2", "POAL")
DEFAULT_OUTPUT = PROJECT_ROOT / "experimentos/xie_20260920"
METRICS = ("MAE", "MSE", "RMSE", "R2")
PRE_LSTM_STAGES = ("02_localizar_arquivos_ismr.py", "03_extrair_ismr.py", "04_controle_qualidade_eda.py",
                   "06_feature_engineering.py", "07_dataset_classificacao.py", "08_train_xgboost.py", "09_filtrar_vtec.py")
THREAD_ENVIRONMENT = {
    "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
    "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
    "OMP_NUM_THREADS": "2", "OMP_THREAD_LIMIT": "2",
    "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
    "LOKY_MAX_CPU_COUNT": "2",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def source_manifest() -> dict[str, str]:
    files = {str(path.relative_to(PROJECT_ROOT)): sha256(path)
             for path in sorted(SCRIPT_DIRECTORY.rglob("*"))
             if path.is_file() and path.suffix in (".py", ".html", ".js", ".css")}
    requirements = PROJECT_ROOT / "requirements.txt"
    if requirements.is_file():
        files[requirements.name] = sha256(requirements)
    return files


def source_inputs(config: dict, station: str) -> list[dict]:
    """Inventário de tamanho/mtime; ISMR não é relido integralmente para calcular hash."""
    inputs = config.get("inputs", {})
    data_root = (PROJECT_ROOT / inputs.get("data_root", "../Dados")).resolve()
    paths: set[Path] = set()
    period = config["period"]
    for directory in inputs.get("ismr_directories", ["Dados ismr 2024"]):
        base = (data_root / directory).resolve()
        if (base / station).is_dir():
            base = base / station
        for path in base.rglob("*"):
            if not path.is_file() or not path.name.upper().startswith(station + "_"):
                continue
            if not path.name.lower().endswith((".ismr", ".ismr.gz")):
                continue
            dates = re.findall(r"\d{4}-\d{2}-\d{2}", path.name)
            if len(dates) >= 2 and (dates[0] > period["test_end"][:10] or dates[1] < period["study_start"][:10]):
                continue
            paths.add(path.resolve())
    paths.update((PROJECT_ROOT / filename).resolve() for filename in inputs.get("space_weather_files", []))
    inventory = []
    for path in sorted(paths):
        item = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            stat = path.stat()
            item.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        inventory.append(item)
    return inventory


def child_invocation(station: str, config: Path, output: Path, preflight_only: bool = False,
                     min_free_space_gb: float = 5., recovery_manifest: Path | None = None):
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("GNSS_")}
    environment.update(THREAD_ENVIRONMENT)
    environment["GNSS_MIN_FREE_SPACE_GB"] = str(float(min_free_space_gb))
    command = [sys.executable, "-X", "utf8", "-u", str(SCRIPT_DIRECTORY / "12_multistation_pipeline.py"),
               "--config", str(config.resolve()), "--output-root", str(output.resolve()), "--stations", station]
    if preflight_only:
        command.append("--preflight-only")
    if recovery_manifest is not None:
        command.extend(["--start-at-lstm", "--recovery-manifest", str(recovery_manifest.resolve())])
    return command, environment


def expected_models(config: dict) -> list[dict]:
    models = []
    for filtered in (False, True):
        for item in config["regression"]["architectures"]:
            name = (f"{item['id'].lower()}_{item['organization']}_"
                    f"{'attention' if item['attention'] else 'lstm'}_"
                    f"{'multi' if item['multivariate'] else 'uni'}_"
                    f"{'filtrado' if filtered else 'original'}")
            models.append({"config_id": item["id"], "filtrado": filtered,
                           "organizacao": item["organization"], "modelo": name})
    return models


def truthy(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).strip().lower() in ("true", "1"):
        return True
    if str(value).strip().lower() in ("false", "0"):
        return False
    raise ValueError(f"Valor booleano inválido: {value!r}")


def read_metrics(folder: Path) -> pd.DataFrame:
    path = folder / "nucleo2/metricas_regressao.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["filtrado"] = frame["filtrado"].map(truthy)
    if frame.duplicated(["config_id", "filtrado"]).any():
        raise ValueError(f"Métrica repetida por configuração/ramo: {path}")
    return frame


def completeness_errors(output: Path, station: str, config: dict) -> list[str]:
    """Valida a matriz inteira, previsões temporais e todos os 48 slots piecewise."""
    results = output / "resultados" / station
    models = output / "modelos" / station
    errors = []
    try:
        if read_json(output / "resultados/pipeline_status.json").get("status") != "concluido":
            errors.append("pipeline não concluído")
        metrics = read_metrics(results)
        expected = expected_models(config)
        if len(expected) != 16 or len(metrics) != 16:
            errors.append("matriz M1–M8 × dois ramos precisa conter 16 métricas")
        actual_keys = set(zip(metrics["config_id"], metrics["filtrado"]))
        if actual_keys != {(item["config_id"], item["filtrado"]) for item in expected}:
            errors.append("configurações/ramos incompletos")
        if not np.isfinite(metrics[list(METRICS)].to_numpy(dtype=float)).all():
            errors.append("métricas não finitas")
        period = config["period"]
        resolution = config.get("preprocessing", {}).get("time_resolution", "30min")
        epochs = pd.date_range(pd.Timestamp(period["test_start"]).ceil(resolution),
                               pd.Timestamp(period["test_end"]).floor(resolution), freq=resolution)
        required = [models / "nucleo1/xgboost_s4.joblib", results / "nucleo1/metricas_teste.json",
                    results / "relatorio/comparacao_rmse.csv",
                    results / "relatorio/visual_analytics/painel_visual_analytics.html"]
        for item in expected:
            name = item["modelo"]
            row = metrics.loc[metrics["modelo"].eq(name)]
            if len(row) != 1 or int(row.iloc[0]["n_predicoes_teste"]) != len(epochs):
                errors.append(f"métrica/contagem incorreta: {name}")
            required.append(models / "nucleo2" / f"escalas_{name}.joblib")
            if item["organizacao"] == "piecewise":
                required.extend(models / "nucleo2" / name / f"slot_{slot:02d}.keras" for slot in range(48))
            else:
                required.append(models / "nucleo2" / f"{name}.keras")
            prediction = results / "nucleo2" / f"predicoes_{name}.csv"
            if not prediction.is_file():
                errors.append(f"previsão ausente: {name}")
                continue
            frame = pd.read_csv(prediction, parse_dates=["datetime"])
            if frame["datetime"].duplicated().any() or not pd.DatetimeIndex(frame["datetime"].sort_values()).equals(epochs):
                errors.append(f"épocas de teste incorretas: {name}")
            if not np.isfinite(frame[["observado", "predito"]].to_numpy(dtype=float)).all():
                errors.append(f"previsão não finita: {name}")
        errors.extend(f"artefato ausente/vazio: {path}" for path in required
                      if not path.is_file() or path.stat().st_size == 0)
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"artefato inválido: {error}")
    return errors


def can_resume(record: dict, fingerprints: dict, output: Path, station: str, config: dict) -> bool:
    return (record.get("status") == "concluido"
            and all(record.get(key) == value for key, value in fingerprints.items())
            and not completeness_errors(output, station, config))


def validate_pre_lstm_stages(status: dict, station: str) -> None:
    stages = {item["script"]: item for item in status.get("estacoes", {}).get(station, {}).get("etapas", [])}
    stages.update({item["script"]: item for item in status.get("etapas_globais", [])})
    for script in ("01_validar_insumos.py", "05_integrar_clima_espacial.py", *PRE_LSTM_STAGES):
        item = stages.get(script, {})
        if item.get("status") != "concluido" or item.get("codigo_saida") != 0:
            raise ValueError(f"Retomada exige etapa anterior concluída: {station}/{script}")


def validate_recovery_code(before: dict, after: dict) -> list[str]:
    old = {name.replace("\\", "/"): digest for name, digest in before.items()}
    new = {name.replace("\\", "/"): digest for name, digest in after.items()}
    changed = sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))
    allowed = {"scripts/10_train_lstm.py", "scripts/12_multistation_pipeline.py", "scripts/14_retreinar_xie.py"}
    forbidden = set(changed) - allowed
    if forbidden:
        raise ValueError("Recuperação rejeitada: código científico anterior mudou: " + ", ".join(sorted(forbidden)))
    return changed


def validate_recovery_manifest(manifest_path: Path, root: Path, station: str,
                               plan: dict | None = None, previous: dict | None = None) -> dict:
    """Exige adoção explícita e preservada; nunca aceita silenciosamente outro código."""
    root = root.resolve()
    adoption = read_json(manifest_path)
    evidence = adoption.get("runner_recovery", {})
    output = root / station
    if evidence.get("schema_version") != 1 or evidence.get("station") != station:
        raise ValueError("Manifesto de recuperação não corresponde à estação/schema")
    if Path(evidence.get("output_root", "")).resolve() != output.resolve():
        raise ValueError("Manifesto de recuperação aponta para outra raiz de estação")
    saved_files = {}
    for key, original_name in (("previous_manifest", "manifesto_execucao.json"),
                                ("previous_runner_status", "runner_status.json")):
        path = Path(evidence[key]).resolve()
        if path == root / original_name:
            raise ValueError("A recuperação exige cópia preservada do manifesto/estado anterior")
        if sha256(path) != evidence[key + "_sha256"]:
            raise ValueError(f"Evidência histórica alterada: {path}")
        saved_files[key] = read_json(path)
    historical = saved_files["previous_manifest"]
    old_record = saved_files["previous_runner_status"]["estacoes"][station]
    config_path = Path(historical["estacoes"][station]["config"]).resolve()
    config = read_json(config_path)
    code = source_manifest()
    fingerprints = {"config_digest": sha256(config_path), "code_digest": object_digest(code),
                    "inputs_digest": object_digest(source_inputs(config, station))}
    if plan is not None and (plan["fingerprints"] != fingerprints or plan["config"].resolve() != config_path):
        raise ValueError("Plano corrente difere da configuração/insumos adotados")
    if previous is not None:
        current_record = previous.get("estacoes", {}).get(station, {})
        for key in ("config_digest", "code_digest", "inputs_digest", "config", "output_root"):
            if current_record.get(key) != old_record.get(key):
                raise ValueError(f"Estado atual diverge da execução preservada: {key}")
    for key in ("config_digest", "inputs_digest"):
        if not (evidence.get(key) == fingerprints[key] == old_record.get(key) == historical["estacoes"][station].get(key)):
            raise ValueError(f"Proveniência alterada na recuperação: {key}")
    old_code = historical["arquivos_codigo_sha256"]
    if evidence.get("old_code_digest") != object_digest(old_code) or old_record.get("code_digest") != object_digest(old_code):
        raise ValueError("Hash de código anterior incompatível")
    if evidence.get("new_code_digest") != fingerprints["code_digest"]:
        raise ValueError("Hash de código novo incompatível com a adoção explícita")
    changed = validate_recovery_code(old_code, code)
    pipeline_path = output / "resultados/pipeline_status.json"
    if sha256(pipeline_path) != evidence.get("previous_pipeline_status_sha256"):
        raise ValueError("Estado do pipeline mudou desde a adoção")
    prior_pipeline = read_json(pipeline_path)
    validate_pre_lstm_stages(prior_pipeline, station)
    artifacts = {Path(name).resolve(): digest for name, digest in adoption.get("artefatos", {}).items()}
    required = {
        output / "dados_processados" / station / f"{station}_features.csv",
        output / "dados_processados" / station / f"{station}_dataset_regressao.csv",
        output / "dados_processados" / station / f"{station}_dataset_classificacao.csv",
        output / "resultados" / station / "nucleo1/predicoes_xgboost.csv",
        output / "resultados" / station / "nucleo1/metricas_teste.json",
        output / "resultados" / station / "nucleo1/hiperparametros_xgboost.json",
        output / "modelos" / station / "nucleo1/xgboost_s4.joblib",
    }
    required.update(path.resolve() for directory in (output / "modelos" / station / "nucleo2",
                                                      output / "resultados" / station / "nucleo2")
                    for path in directory.rglob("*") if path.is_file())
    missing = required - artifacts.keys()
    if missing:
        raise ValueError("Artefatos sem hash de adoção: " + ", ".join(str(path) for path in sorted(missing)))
    for path, digest in artifacts.items():
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Artefato adotado ausente, alterado ou fora da estação: {path}")
    return {"manifesto": str(manifest_path.resolve()), "manifesto_sha256": sha256(manifest_path),
            "arquivos_verificados": len(artifacts), "scripts_alterados": changed,
            "code_digest_anterior": evidence["old_code_digest"], "code_digest_atual": evidence["new_code_digest"]}


def observed_medians(features: Path) -> pd.DataFrame:
    data = pd.read_csv(features, usecols=["datetime", "VTEC"], parse_dates=["datetime"])
    data["VTEC"] = pd.to_numeric(data["VTEC"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return data.groupby("datetime", as_index=False)["VTEC"].median().rename(columns={"VTEC": "VTEC_observado"})


def capture_baseline(root: Path, stations: list[str]) -> dict:
    destination = root / "baseline"
    manifest_path = destination / "manifesto_baseline.json"
    if manifest_path.is_file():
        baseline = read_json(manifest_path)
        if not set(stations).issubset(baseline["estacoes"]):
            raise ValueError("A cópia anterior não contém estas estações; use outra --output-root.")
        for relative, digest in baseline["arquivos_sha256"].items():
            if not (destination / relative).is_file() or sha256(destination / relative) != digest:
                raise ValueError(f"Cópia anterior alterada: {relative}")
        return baseline
    if destination.exists():
        raise ValueError(f"Cópia anterior incompleta; preserve {destination} e use outra --output-root.")
    staging = root / f".baseline_{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    manifest = {"capturado_em": now(), "origem": str(PROJECT_ROOT), "estacoes": {}, "arquivos_sha256": {}}
    for station in stations:
        source = PROJECT_ROOT / "resultados" / station
        selected = [source / "nucleo1/metricas_teste.json", source / "nucleo1/validacao_temporal.csv",
                    source / "nucleo1/resumo_predicoes.json",
                    source / "nucleo2/metricas_regressao.csv", source / "relatorio/resumo.json"]
        selected.extend(sorted((source / "nucleo2").glob("predicoes_*.csv")))
        missing = []
        for path in selected:
            if not path.is_file():
                missing.append(str(path))
                continue
            target = staging / station / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        features = PROJECT_ROOT / "dados_processados" / station / f"{station}_features.csv"
        (staging / station).mkdir(exist_ok=True)
        if features.is_file():
            observed_medians(features).to_csv(staging / station / "vtec_observado_30min.csv", index=False)
        else:
            missing.append(str(features))
        manifest["estacoes"][station] = {"origem_resultados": str(source), "ausentes": missing}
    manifest["arquivos_sha256"] = {path.relative_to(staging).as_posix(): sha256(path)
                                   for path in sorted(staging.rglob("*")) if path.is_file()}
    atomic_json(staging / manifest_path.name, manifest)
    staging.rename(destination)
    return manifest


def common_observed_metrics(station: str, baseline: Path, current: Path, observed: pd.DataFrame):
    """Avalia todos os ramos/execuções na mesma mediana observada e mesmas épocas."""
    truth = observed.set_index("datetime")["VTEC_observado"].dropna().astype(float)
    truth = truth.loc[np.isfinite(truth)]
    predictions = []
    common = truth.index
    for execution, folder in (("antes", baseline), ("depois", current)):
        metrics = read_metrics(folder)
        for row in metrics.to_dict("records"):
            path = folder / "nucleo2" / f"predicoes_{row['modelo']}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Previsão necessária à comparação: {path}")
            pred = pd.read_csv(path, usecols=["datetime", "predito"], parse_dates=["datetime"])
            if pred["datetime"].duplicated().any():
                raise ValueError(f"Previsões duplicadas: {path}")
            series = pred.set_index("datetime")["predito"].astype(float)
            series = series.loc[np.isfinite(series)]
            common = common.intersection(series.index)
            predictions.append((execution, row, series))
    common = common.sort_values()
    coverage = {"estacao": station, "epocas_observadas": len(truth), "epocas_comuns": len(common),
                "series_comparadas": len(predictions),
                "referencia": "Mediana de VTEC dos SVIDs observados, antes da interpolação, da nova execução"}
    results = []
    if len(common) and predictions:
        actual = truth.reindex(common).to_numpy()
        total = float(np.square(actual - actual.mean()).sum())
        for execution, row, pred in predictions:
            error = pred.reindex(common).to_numpy() - actual
            mse = float(np.square(error).mean())
            results.append({"estacao": station, "execucao": execution, "config_id": row["config_id"],
                            "filtrado": row["filtrado"], "modelo": row["modelo"],
                            "n_epocas_comuns": len(common), "inicio_comum": str(common.min()),
                            "fim_comum": str(common.max()), "MAE": float(np.abs(error).mean()),
                            "MSE": mse, "RMSE": float(np.sqrt(mse)),
                            "R2": 1 - float(np.square(error).sum()) / total if total else np.nan})
    return pd.DataFrame(results), coverage


def comparison_table(frame: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    available = [column for column in columns if column in frame.columns]
    before = frame.loc[frame["execucao"].eq("antes"), keys + available]
    after = frame.loc[frame["execucao"].eq("depois"), keys + available]
    paired = before.merge(after, on=keys, suffixes=("_antes", "_depois"), how="outer", validate="one_to_one")
    for column in available:
        paired[f"delta_{column}"] = paired[f"{column}_depois"] - paired[f"{column}_antes"]
    return paired


def enrich_classification(metrics: dict) -> dict:
    result = dict(metrics)
    if all(key in result for key in ("tn", "fp", "fn", "tp")):
        result.setdefault("n_positivos", result["tp"] + result["fn"])
        result.setdefault("n_amostras", sum(result[key] for key in ("tn", "fp", "fn", "tp")))
        denominator = 5 * result["tp"] + 4 * result["fn"] + result["fp"]
        result.setdefault("f2", 5 * result["tp"] / denominator if denominator else 0.)
    return result


def consolidate(root: Path, plans: dict, status: dict) -> dict:
    destination = root / "comparacao"
    destination.mkdir(exist_ok=True)
    classification, regression, common, coverage, periods = [], [], [], [], []
    for station, plan in plans.items():
        periods.append({"estacao": station, **plan["manifest"]["period"],
                        "status": status["estacoes"][station]["status"]})
        baseline = root / "baseline" / station
        current = plan["output"] / "resultados" / station
        completed = status["estacoes"][station]["status"] == "concluido"
        for execution, folder in (("antes", baseline), ("depois", current)):
            if execution == "depois" and not completed:
                continue
            path = folder / "nucleo1/metricas_teste.json"
            if path.is_file():
                metrics = enrich_classification(read_json(path))
                summary = folder / "nucleo1/resumo_predicoes.json"
                if summary.is_file():
                    threshold = read_json(summary).get("limiar_probabilidade")
                    if threshold is not None:
                        metrics.setdefault("limiar_probabilidade", threshold)
                classification.append({"estacao": station, "execucao": execution, **metrics})
            frame = read_metrics(folder)
            if not frame.empty:
                frame.insert(0, "execucao", execution)
                frame.insert(0, "estacao", station)
                regression.append(frame)
        if completed:
            features = plan["output"] / "dados_processados" / station / f"{station}_features.csv"
            if not features.is_file():
                raise FileNotFoundError(f"Observações necessárias à comparação ausentes: {features}")
            observed = observed_medians(features)
            observed.to_csv(destination / f"{station}_vtec_observado_comum.csv", index=False)
            frame, item = common_observed_metrics(station, baseline, current, observed)
            if not frame.empty:
                common.append(frame)
            prior = baseline / "vtec_observado_30min.csv"
            if prior.is_file():
                old = pd.read_csv(prior, parse_dates=["datetime"])
                joined = old.merge(observed, on="datetime", suffixes=("_antes", "_depois"), validate="one_to_one").dropna()
                item["medianas_antes_depois_diferentes"] = int((~np.isclose(joined["VTEC_observado_antes"], joined["VTEC_observado_depois"], rtol=1e-8, atol=1e-8)).sum())
            coverage.append(item)
    frames = {
        "periodos_estacoes": pd.DataFrame(periods),
        "classificacao_antes_depois": pd.DataFrame(classification),
        "regressao_alvo_proprio": pd.concat(regression, ignore_index=True) if regression else pd.DataFrame(),
        "regressao_observado_comum": pd.concat(common, ignore_index=True) if common else pd.DataFrame(),
        "cobertura_observado_comum": pd.DataFrame(coverage),
    }
    frames["comparacao_classificacao"] = comparison_table(frames["classificacao_antes_depois"], ["estacao"],
        ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "average_precision",
         "n_positivos", "n_amostras", "limiar_probabilidade", "tn", "fp", "fn", "tp"])
    for source, target in (("regressao_alvo_proprio", "comparacao_regressao_alvo_proprio"),
                           ("regressao_observado_comum", "comparacao_regressao_observado_comum")):
        frames[target] = comparison_table(frames[source], ["estacao", "config_id", "filtrado"], list(METRICS))
    for name, frame in frames.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    atomic_json(destination / "resumo_comparacao.json", {"atualizado_em": now(), "cobertura": coverage,
                "estacoes_concluidas": [station for station in plans if status["estacoes"][station]["status"] == "concluido"]})
    write_html(root, plans, status, frames)
    return {name: len(frame) for name, frame in frames.items()}


def write_html(root: Path, plans: dict, status: dict, frames: dict) -> None:
    sections = []
    for station in plans:
        folder = f"{station}/resultados/{station}/relatorio/visual_analytics"
        state = status["estacoes"][station]["status"]
        link = f'<a href="{folder}/painel_visual_analytics.html">painel da estação</a>' if state == "concluido" else "resultados ainda indisponíveis"
        sections.append(f"<li><strong>{station}</strong>: {html.escape(state)} — {link}</li>")
    tables = []
    labels = {"periodos_estacoes": "Períodos e estado", "comparacao_classificacao": "Classificação: antes e depois",
              "comparacao_regressao_alvo_proprio": "Regressão: alvo próprio de cada ramo",
              "comparacao_regressao_observado_comum": "Regressão: referência observada comum",
              "cobertura_observado_comum": "Cobertura da referência observada"}
    for name, label in labels.items():
        frame = frames[name]
        table = frame.to_html(index=False, float_format=lambda value: f"{value:.5g}", na_rep="—", escape=True) if not frame.empty else "<p>Aguardando resultados completos.</p>"
        tables.append(f'<h2>{label}</h2><p><a href="comparacao/{name}.csv">Baixar CSV</a></p><div class="scroll">{table}</div>')
    document = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retreinamento Xie — comparação</title><style>
body{font:16px system-ui,sans-serif;max-width:1500px;margin:40px auto;padding:0 24px;color:#193341;background:#f5f8fa}
h1,h2{color:#103c50}a{color:#006c92}.scroll{overflow:auto;background:white;border:1px solid #d6e0e5}
table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #e2e9ee;white-space:nowrap}
th{background:#e9f1f5}li{margin:9px 0}.note{padding:18px;background:#fff5d8;line-height:1.6}
</style><h1>Retreinamento Xie: antes e depois</h1>'''
    document += f'<p>Atualizado: {html.escape(now())} · Estado: <strong>{html.escape(status["status"])}</strong></p>'
    document += '<p class="note">Cada estação preserva seu recorte: FRTZ/PALM em janeiro–março, PRU2 em outubro–dezembro e POAL em janeiro–fevereiro de 2024. As diferenças entre estações são descritivas: não constituem uma comparação pareada no mesmo período. Δ = depois − antes; RMSE/MAE menores e R² maior são preferíveis. O alvo filtrado difere do observado; a tabela de referência comum reavalia todas as previsões contra a mediana observada dos SVIDs, sem interpolação, nas mesmas épocas disponíveis por estação. Essa mediana é uma referência observacional e não uma verdade ionosférica independente.</p>'
    document += "<ul>" + "".join(sections) + "</ul>" + "".join(tables)
    document += '<p><a href="runner_status.json">Estado detalhado</a> · <a href="manifesto_execucao.json">Manifesto e hashes</a> · <a href="baseline/manifesto_baseline.json">Origem dos resultados anteriores</a></p></html>'
    temporary = root / ".relatorio_comparativo.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, root / "relatorio_comparativo.html")


@contextmanager
def run_lock(root: Path):
    """Bloqueio do sistema operacional; liberado mesmo após queda do executor."""
    path = root / ".runner.lock"
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Outro executor está usando {root}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def archive_previous(root: Path, station: str, output: Path) -> str | None:
    if not output.exists():
        return None
    # Ambos os caminhos resolvidos precisam permanecer dentro da raiz isolada.
    root = root.resolve()
    output = output.resolve()
    if output != root / station or not output.is_relative_to(root):
        raise ValueError(f"Diretório de estação fora da raiz isolada: {output}")
    archive = root / "tentativas_anteriores" / f"{station}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    archive.parent.mkdir(exist_ok=True)
    if not archive.resolve().is_relative_to(root):
        raise ValueError("Arquivo de tentativas fora da raiz isolada")
    output.rename(archive)
    return str(archive)


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE: consulta somente.
        if not handle:
            return ctypes.get_last_error() == 5  # Acesso negado: conservadoramente ativo.
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reject_running_children(previous: dict) -> None:
    for station, record in previous.get("estacoes", {}).items():
        if record.get("status") == "executando" and record.get("pid") and process_alive(int(record["pid"])):
            raise RuntimeError(f"O pipeline anterior de {station} ainda está ativo (PID {record['pid']}); não é seguro retomar a mesma raiz.")


def execute_plans(plans: dict, status: dict, root: Path, workers: int, preflight_only: bool,
                  min_free_space_gb: float = 5.) -> None:
    pending = [station for station in plans if status["estacoes"][station]["status"] != "concluido"]
    active: dict[str, tuple] = {}
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    def save():
        status["atualizado_em"] = now()
        atomic_json(root / "runner_status.json", status)

    try:
        while pending or active:
            while pending and len(active) < workers:
                station = pending.pop(0)
                plan = plans[station]
                record = status["estacoes"][station]
                log = logs / f"{station}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.log"
                command, environment = child_invocation(station, plan["config"], plan["output"], preflight_only,
                                                        min_free_space_gb, plan.get("recovery_manifest"))
                stream = log.open("w", encoding="utf-8", newline="\n")
                try:
                    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=stream,
                                               stderr=subprocess.STDOUT)
                except Exception as error:
                    stream.close()
                    record.update(status="erro", erro=f"{type(error).__name__}: {error}", log=str(log), fim=now())
                    save()
                    continue
                record.update(status="executando", pid=process.pid, inicio=now(), log=str(log), comando=command)
                active[station] = (process, stream, time.monotonic())
                print(f"[{station}] PID {process.pid}; log: {log}", flush=True)
                save()
            for station in list(active):
                process, stream, start = active[station]
                returncode = process.poll()
                if returncode is None:
                    continue
                stream.close()
                record = status["estacoes"][station]
                plan = plans[station]
                errors = [] if preflight_only or returncode else completeness_errors(plan["output"], station, plan["manifest"])
                record.update(codigo_saida=returncode, fim=now(), duracao_segundos=round(time.monotonic() - start, 2))
                if returncode or errors:
                    record.update(status="erro", erro="; ".join(errors) if errors else f"Pipeline saiu com código {returncode}")
                else:
                    record.update(status="preflight_concluido" if preflight_only else "concluido")
                del active[station]
                print(f"[{station}] {record['status']}", flush=True)
                save()
            if active:
                time.sleep(1)
    except BaseException:
        for station, (process, stream, _) in active.items():
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    process.terminate()
                process.wait()
            stream.close()
            status["estacoes"][station].update(status="interrompido", fim=now())
        save()
        raise


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    parser.add_argument("--stations", default=",".join(STATIONS))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1,
                        help="estações simultâneas; padrão 1 para a máquina com 8 GB de RAM")
    parser.add_argument("--min-free-space-gb", type=float, default=5.,
                        help="reserva exigida no preflight: padrão 5 GB; saída anterior total ~3,3 GB")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="retoma estações completas compatíveis e reexecuta as demais integralmente")
    parser.add_argument("--resume-training", action="store_true", help="preserva etapas até09 e retoma10 mediante adoção explícita verificada")
    parser.add_argument("--recovery-manifest", type=Path, help="manifesto com runner_recovery e hashes dos artefatos adotados")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = arguments(argv)
    if args.resume_training:
        if args.preflight_only or args.recovery_manifest is None:
            raise ValueError("--resume-training exige --recovery-manifest e não aceita --preflight-only")
        args.resume = True
    elif args.recovery_manifest is not None:
        raise ValueError("--recovery-manifest exige --resume-training")
    stations = [station.strip().upper() for station in args.stations.split(",") if station.strip()]
    if not stations or len(set(stations)) != len(stations) or not set(stations).issubset(STATIONS):
        raise ValueError("--stations deve selecionar FRTZ,PALM,PRU2,POAL sem repetições.")
    if not np.isfinite(args.min_free_space_gb) or args.min_free_space_gb <= 0:
        raise ValueError("--min-free-space-gb precisa ser positivo e finito.")
    root = args.output_root.resolve()
    protected = [PROJECT_ROOT, *(PROJECT_ROOT / folder for folder in ("resultados", "modelos", "dados_processados", "scripts", "configs"))]
    if any(root == path or (path != PROJECT_ROOT and root.is_relative_to(path)) for path in protected):
        raise ValueError("Use uma raiz isolada, por exemplo experimentos/xie_20260920.")
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        status_path = root / "runner_status.json"
        previous = read_json(status_path) if status_path.is_file() else {}
        if previous and not args.resume:
            raise ValueError("Esta raiz já possui uma execução. Use --resume ou outra --output-root.")
        reject_running_children(previous)
        if not previous and any((root / station).exists() for station in stations):
            raise ValueError("Há saídas sem estado do executor. Preserve esta raiz e escolha outra --output-root.")
        code = source_manifest()
        plans = {}
        for station in stations:
            path = (args.config_dir / f"xie_{station.lower()}.json").resolve()
            config = read_json(path)
            if config.get("stations") != [station]:
                raise ValueError(f"{path} deve declarar somente a estação {station}")
            if len(expected_models(config)) != 16:
                raise ValueError(f"{path} precisa preservar M1–M8, em ambos os ramos")
            inventory = source_inputs(config, station)
            plans[station] = {"config": path, "manifest": config, "output": root / station, "inputs": inventory,
                              "fingerprints": {"config_digest": sha256(path), "code_digest": object_digest(code),
                                               "inputs_digest": object_digest(inventory)}}
        recovery = None
        recovery_station = None
        if args.resume_training:
            recovery_station = read_json(args.recovery_manifest).get("runner_recovery", {}).get("station")
            if recovery_station not in plans:
                raise ValueError("A estação do manifesto precisa pertencer a --stations")
            recovery = validate_recovery_manifest(args.recovery_manifest, root, recovery_station,
                                                  plans[recovery_station], previous)
            plans[recovery_station]["recovery_manifest"] = args.recovery_manifest.resolve()
            for station, plan in plans.items():
                old = previous.get("estacoes", {}).get(station, {})
                if station != recovery_station and plan["output"].exists() and not can_resume(old, plan["fingerprints"], plan["output"], station, plan["manifest"]):
                    raise ValueError(f"{station} também contém execução incompleta; precisa de adoção própria, não será arquivada")
        baseline = capture_baseline(root, stations)
        status = {"inicio": now(), "status": "executando", "pid": os.getpid(), "python": sys.executable,
                  "workers": args.workers, "min_free_space_gb": args.min_free_space_gb,
                  "preflight_only": args.preflight_only, "estacoes": {}}
        for station, plan in plans.items():
            old = previous.get("estacoes", {}).get(station, {})
            record = {"status": "pendente", "config": str(plan["config"]), "output_root": str(plan["output"]),
                      **plan["fingerprints"]}
            if station == recovery_station:
                record.update(recuperacao=recovery, inicio_original=old.get("inicio"), log_anterior=old.get("log"))
            elif args.resume and not args.preflight_only and can_resume(old, plan["fingerprints"], plan["output"], station, plan["manifest"]):
                record = {**old, "retomado_sem_reexecutar": True}
            else:
                archived = archive_previous(root, station, plan["output"])
                if archived:
                    record["tentativa_anterior"] = archived
            status["estacoes"][station] = record
        packages = {}
        for package in ("numpy", "pandas", "scikit-learn", "xgboost", "tensorflow", "keras", "imbalanced-learn"):
            try:
                packages[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                packages[package] = None
        manifest = {"criado_em": now(), "python": sys.version, "executavel": sys.executable,
                    "workers": args.workers, "ambiente_filhos": THREAD_ENVIRONMENT, "pacotes": packages,
                    "min_free_space_gb": args.min_free_space_gb,
                    "dimensionamento": "Execução anterior: dados 2,164 GiB + modelos 0,955 GiB + resultados 0,202 GiB; reserva padrão 5 GiB; uma estação simultânea devido à RAM de 8 GB.",
                    "arquivos_codigo_sha256": code, "baseline_capturado_em": baseline["capturado_em"],
                    "estacoes": {station: {"config": str(plan["config"]), "configuracao": plan["manifest"],
                                          "insumos_tamanho_mtime": plan["inputs"], **plan["fingerprints"]}
                                  for station, plan in plans.items()}}
        if recovery is not None:
            manifest["recuperacao"] = recovery
            status["recuperacao"] = recovery
        atomic_json(root / "manifesto_execucao.json", manifest)
        atomic_json(status_path, status)
        try:
            execute_plans(plans, status, root, args.workers, args.preflight_only, args.min_free_space_gb)
            success = "preflight_concluido" if args.preflight_only else "concluido"
            status["status"] = success if all(item["status"] == success for item in status["estacoes"].values()) else "erro"
            if not args.preflight_only:
                if source_manifest() != code or any(sha256(plan["config"]) != plan["fingerprints"]["config_digest"] for plan in plans.values()):
                    for record in status["estacoes"].values():
                        if record["status"] == "concluido":
                            record.update(status="erro", erro="Código/configuração mudou durante a execução")
                    raise RuntimeError("Código/configuração mudou durante a execução; resultados precisam de nova execução isolada.")
                status["comparacao"] = consolidate(root, plans, status)
        except KeyboardInterrupt:
            status.update(status="interrompido", erro="Interrompido pelo usuário")
        except Exception as error:
            status.update(status="erro", erro=f"{type(error).__name__}: {error}")
            print(status["erro"], file=sys.stderr, flush=True)
        finally:
            status["fim"] = now()
            atomic_json(status_path, status)
        print(f"Estado final: {status['status']}; {status_path}", flush=True)
        return 0 if status["status"] in ("concluido", "preflight_concluido") else 1


if __name__ == "__main__":
    raise SystemExit(main())
