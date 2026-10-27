"""Executa o fluxo multiestação com preflight, logs e estado recuperável."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import pandas as pd


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from config import (  # noqa: E402
    EXPERIMENT_CONFIG_PATH,
    EXPERIMENT_MODE,
    EXPERIMENT_NAME,
    PROJECT_ROOT,
    RESULTS_DIRECTORY,
    STATIONS,
    experiment_digest,
    resolved_experiment,
    show_config,
)


SCRIPT_PREFLIGHT = "01_validar_insumos.py"
SCRIPT_COMUM = "05_integrar_clima_espacial.py"
SCRIPT_ARTIGO = "13_generate_article_assets.py"
SCRIPTS_ESTACAO = (
    "02_localizar_arquivos_ismr.py",
    "03_extrair_ismr.py",
    "04_controle_qualidade_eda.py",
    "06_feature_engineering.py",
    "07_dataset_classificacao.py",
    "08_train_xgboost.py",
    "09_filtrar_vtec.py",
    "10_train_lstm.py",
    "11_generate_report.py",
)
SCRIPTS = (SCRIPT_PREFLIGHT, SCRIPT_COMUM, *SCRIPTS_ESTACAO, SCRIPT_ARTIGO)
PASTA_LOGS = Path(RESULTS_DIRECTORY) / "logs"
ARQUIVO_STATUS = Path(RESULTS_DIRECTORY) / "pipeline_status.json"
ARQUIVO_CONFIG_RESOLVIDA = Path(RESULTS_DIRECTORY) / "configuracao_resolvida.json"


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _salvar_status(status: dict[str, Any]) -> None:
    ARQUIVO_STATUS.parent.mkdir(parents=True, exist_ok=True)
    descritor, temporario = tempfile.mkstemp(
        prefix=".pipeline_status_", suffix=".tmp", dir=ARQUIVO_STATUS.parent
    )
    try:
        with os.fdopen(descritor, "w", encoding="utf-8") as arquivo:
            json.dump(status, arquivo, ensure_ascii=False, indent=2)
        os.replace(temporario, ARQUIVO_STATUS)
    except Exception:
        if os.path.exists(temporario):
            os.remove(temporario)
        raise


def _salvar_configuracao_resolvida() -> None:
    ARQUIVO_CONFIG_RESOLVIDA.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".configuracao_resolvida_",
        suffix=".tmp",
        dir=ARQUIVO_CONFIG_RESOLVIDA.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(resolved_experiment(), file, ensure_ascii=False, indent=2)
        os.replace(temporary, ARQUIVO_CONFIG_RESOLVIDA)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa o duplo núcleo e a comparação M1--M8 descritos no "
            "Estudo Dirigido."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="manifesto JSON; padrão: configs/estudo_dirigido.json",
    )
    parser.add_argument("--data-root", type=Path, help="raiz dos dados de entrada")
    parser.add_argument(
        "--ismr-dir",
        type=Path,
        action="append",
        help="diretório ISMR; a opção pode ser repetida",
    )
    parser.add_argument(
        "--space-weather",
        type=Path,
        action="append",
        help="CSV/TXT OmniWeb; a opção pode ser repetida",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="raiz isolada para dados processados, modelos e resultados",
    )
    parser.add_argument(
        "--stations",
        help="estações separadas por vírgula, por exemplo FRTZ,PALM",
    )
    parser.add_argument("--station", help=argparse.SUPPRESS)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="valida cobertura e dependências sem processar ou treinar",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="exibe a configuração resolvida e encerra",
    )
    parser.add_argument("--start-at-lstm", action="store_true",
                        help="retoma10 após validar etapas anteriores e manifesto de recuperação")
    parser.add_argument("--recovery-manifest", type=Path,
                        help="adoção explícita com proveniência e hashes dos checkpoints existentes")
    return parser.parse_args()


def executar_script(nome: str, estacao: str, recovery_manifest: Path | None = None) -> dict[str, Any]:
    caminho = SCRIPT_DIRECTORY / nome
    if not caminho.is_file():
        raise FileNotFoundError(f"Etapa inexistente: {caminho}")
    ambiente = os.environ.copy()
    ambiente["GNSS_STATION"] = estacao
    ambiente["GNSS_EXPERIMENT_CONFIG"] = str(EXPERIMENT_CONFIG_PATH.resolve())
    ambiente["PYTHONUTF8"] = "1"
    if nome == "10_train_lstm.py" and recovery_manifest is not None:
        ambiente["GNSS_LSTM_RESUME_MANIFEST"] = str(recovery_manifest.resolve())
    PASTA_LOGS.mkdir(parents=True, exist_ok=True)
    rotulo = "GLOBAL" if nome in (SCRIPT_PREFLIGHT, SCRIPT_COMUM, SCRIPT_ARTIGO) else estacao
    log = PASTA_LOGS / f"{rotulo}_{Path(nome).stem}.log"
    print(f"\n[{rotulo}] Executando {nome}", flush=True)
    inicio = time.monotonic()
    with open(log, "w", encoding="utf-8", newline="\n") as arquivo_log:
        processo = subprocess.Popen(
            [sys.executable, "-u", str(caminho)],
            cwd=PROJECT_ROOT,
            env=ambiente,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert processo.stdout is not None
        for linha in processo.stdout:
            print(linha, end="", flush=True)
            arquivo_log.write(linha)
            arquivo_log.flush()
        codigo = processo.wait()
    duracao = round(time.monotonic() - inicio, 2)
    resultado = {
        "script": nome,
        "status": "concluido" if codigo == 0 else "erro",
        "codigo_saida": codigo,
        "duracao_segundos": duracao,
        "log": str(log),
        "fim": _agora(),
    }
    if codigo != 0:
        raise subprocess.CalledProcessError(codigo, [sys.executable, str(caminho)])
    return resultado


def consolidar() -> None:
    tabelas: list[pd.DataFrame] = []
    for estacao in STATIONS:
        caminho = Path(RESULTS_DIRECTORY) / estacao / "relatorio" / "comparacao_rmse.csv"
        if not caminho.is_file():
            raise FileNotFoundError(f"Relatório ausente para {estacao}: {caminho}")
        tabela = pd.read_csv(caminho)
        tabela.insert(0, "estacao", estacao)
        tabelas.append(tabela)
    destino = Path(RESULTS_DIRECTORY) / "comparacao_estacoes.csv"
    descritor, temporario = tempfile.mkstemp(
        prefix=".comparacao_estacoes_", suffix=".tmp", dir=destino.parent
    )
    os.close(descritor)
    try:
        pd.concat(tabelas, ignore_index=True).to_csv(temporario, index=False)
        os.replace(temporario, destino)
    except Exception:
        if os.path.exists(temporario):
            os.remove(temporario)
        raise


def _preparar_retomada(args: argparse.Namespace) -> dict[str, Any]:
    if args.preflight_only or args.recovery_manifest is None or len(STATIONS) != 1:
        raise ValueError("--start-at-lstm exige uma estação, --recovery-manifest e não aceita --preflight-only")
    spec = importlib.util.spec_from_file_location("retrain_recovery", SCRIPT_DIRECTORY / "14_retreinar_xie.py")
    recovery = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(recovery)
    station = STATIONS[0]
    root = Path(RESULTS_DIRECTORY).resolve().parent.parent
    evidence = recovery.validate_recovery_manifest(args.recovery_manifest, root, station)
    prior = json.loads(ARQUIVO_STATUS.read_text(encoding="utf-8"))
    if prior.get("experimento", {}).get("digest_sha256") != experiment_digest():
        raise ValueError("Configuração resolvida mudou; não é seguro reutilizar o pré-processamento")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    preserved_status = ARQUIVO_STATUS.with_name(f"pipeline_status_antes_retomada_{stamp}.json")
    shutil.copy2(ARQUIVO_STATUS, preserved_status)
    log = PASTA_LOGS / f"{station}_10_train_lstm.log"
    preserved_log = None
    if log.is_file():
        preserved_log = log.with_name(f"{station}_10_train_lstm_antes_retomada_{stamp}.log")
        shutil.copy2(log, preserved_log)
    prior.setdefault("historico_recuperacoes", []).append({
        **evidence, "retomada_em": _agora(), "inicio_original": prior.get("inicio"),
        "status_anterior": prior.get("status"), "estado_preservado": str(preserved_status),
        "log_lstm_preservado": str(preserved_log) if preserved_log else None,
    })
    prior.update(inicio=_agora(), status="executando", python=sys.executable)
    prior.pop("erro", None)
    prior.pop("fim", None)
    station_status = prior["estacoes"][station]
    station_status["status"] = "pendente"
    station_status.pop("erro", None)
    station_status["etapas"] = [item for item in station_status["etapas"] if item["script"] in recovery.PRE_LSTM_STAGES]
    prior["etapas_globais"] = [item for item in prior["etapas_globais"] if item["script"] in (SCRIPT_PREFLIGHT, SCRIPT_COMUM)]
    return prior


def main() -> None:
    args = _argumentos()
    if args.show_config:
        show_config()
        return
    if args.recovery_manifest is not None and not args.start_at_lstm:
        raise ValueError("--recovery-manifest exige --start-at-lstm")
    status: dict[str, Any] = _preparar_retomada(args) if args.start_at_lstm else {
        "inicio": _agora(),
        "status": "executando",
        "python": sys.executable,
        "experimento": {
            "nome": EXPERIMENT_NAME,
            "modo": EXPERIMENT_MODE,
            "configuracao": str(EXPERIMENT_CONFIG_PATH),
            "configuracao_resolvida": str(ARQUIVO_CONFIG_RESOLVIDA),
            "digest_sha256": experiment_digest(),
        },
        "estacoes": {estacao: {"status": "pendente", "etapas": []} for estacao in STATIONS},
        "etapas_globais": [],
    }
    _salvar_configuracao_resolvida()
    _salvar_status(status)
    try:
        scripts_globais_iniciais = (
            (SCRIPT_PREFLIGHT,) if args.preflight_only or args.start_at_lstm else (SCRIPT_PREFLIGHT, SCRIPT_COMUM)
        )
        for script in scripts_globais_iniciais:
            resultado = executar_script(script, STATIONS[0])
            status["etapas_globais"].append(resultado)
            _salvar_status(status)

        if args.preflight_only:
            status["status"] = "preflight_concluido"
            status["fim"] = _agora()
            _salvar_status(status)
            print("\nPré-validação concluída; treinamento não iniciado.")
            return

        falhas: list[str] = []
        for estacao in STATIONS:
            status["estacoes"][estacao]["status"] = "executando"
            _salvar_status(status)
            try:
                scripts_estacao = ("10_train_lstm.py", "11_generate_report.py") if args.start_at_lstm else SCRIPTS_ESTACAO
                for script in scripts_estacao:
                    if args.start_at_lstm and script == "10_train_lstm.py":
                        resultado = executar_script(script, estacao, recovery_manifest=args.recovery_manifest)
                    else:
                        resultado = executar_script(script, estacao)
                    status["estacoes"][estacao]["etapas"].append(resultado)
                    _salvar_status(status)
            except Exception as erro:
                status["estacoes"][estacao]["status"] = "erro"
                status["estacoes"][estacao]["erro"] = f"{type(erro).__name__}: {erro}"
                falhas.append(estacao)
                _salvar_status(status)
            else:
                status["estacoes"][estacao]["status"] = "concluido"
                _salvar_status(status)

        if falhas:
            raise RuntimeError(
                "Falha em uma ou mais estações: " + ", ".join(falhas)
            )
        consolidar()
        resultado = executar_script(SCRIPT_ARTIGO, STATIONS[0])
        status["etapas_globais"].append(resultado)
        status["status"] = "concluido"
        status["fim"] = _agora()
        _salvar_status(status)
    except Exception as erro:
        status["status"] = "erro"
        status["fim"] = _agora()
        status["erro"] = f"{type(erro).__name__}: {erro}"
        _salvar_status(status)
        raise
    print("\nPipeline multiestação concluído.")


if __name__ == "__main__":
    main()
