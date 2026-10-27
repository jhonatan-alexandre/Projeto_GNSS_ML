"""Valida cobertura, clima espacial, dependências e espaço antes do pipeline."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import sys
import tempfile
from typing import Any

import pandas as pd


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from config import (  # noqa: E402
    ALLOW_ISMR_GAPS,
    ISMR_DIRECTORIES,
    ISMR_FILE_EXTENSIONS,
    PROJECT_ROOT,
    RESULTS_DIRECTORY,
    SPACE_WEATHER_FILES,
    STATIONS,
    STUDY_START,
    TEST_END,
    TEST_START,
)


PADRAO_DATA = re.compile(r"\d{4}-\d{2}-\d{2}")
SAIDA = Path(RESULTS_DIRECTORY) / "preflight.json"
SAIDA_LACUNAS = Path(RESULTS_DIRECTORY) / "dados_faltantes.csv"
DEPENDENCIAS = (
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "xgboost",
    "imblearn",
    "joblib",
    "tensorflow",
    "jinja2",
)


def _intervalo_nome(caminho: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    datas = PADRAO_DATA.findall(caminho.name)
    if len(datas) < 2:
        return None
    return pd.Timestamp(datas[0]), pd.Timestamp(datas[1])


def _arquivos_estacao(estacao: str) -> list[Path]:
    extensoes = tuple(extensao.lower() for extensao in ISMR_FILE_EXTENSIONS)
    encontrados: set[Path] = set()
    for diretorio in ISMR_DIRECTORIES:
        raiz = Path(diretorio)
        if not raiz.is_dir():
            continue
        pasta_estacao = raiz / estacao
        bases = (pasta_estacao,) if pasta_estacao.is_dir() else (raiz,)
        for base in bases:
            for caminho in base.rglob("*"):
                if (
                    caminho.is_file()
                    and caminho.name.upper().startswith(f"{estacao}_")
                    and caminho.name.lower().endswith(extensoes)
                ):
                    encontrados.add(caminho.resolve())
    return sorted(encontrados)


def _fundir_intervalos(
    intervalos: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    fundidos: list[list[pd.Timestamp]] = []
    for inicio, fim in sorted(intervalos):
        if fim <= inicio:
            continue
        if not fundidos or inicio > fundidos[-1][1]:
            fundidos.append([inicio, fim])
        else:
            fundidos[-1][1] = max(fundidos[-1][1], fim)
    return [(inicio, fim) for inicio, fim in fundidos]


def _auditar_estacao(estacao: str) -> tuple[dict[str, Any], list[str]]:
    inicio_requerido = pd.Timestamp(STUDY_START).normalize()
    fim_exclusivo = pd.Timestamp(TEST_END).normalize() + pd.Timedelta(days=1)
    arquivos = _arquivos_estacao(estacao)
    intervalos = [
        intervalo
        for caminho in arquivos
        if (intervalo := _intervalo_nome(caminho)) is not None
        and intervalo[1] > inicio_requerido
        and intervalo[0] < fim_exclusivo
    ]
    fundidos = _fundir_intervalos(intervalos)
    cursor = inicio_requerido
    lacunas: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for inicio, fim in fundidos:
        if fim <= cursor:
            continue
        if inicio > cursor:
            lacunas.append((cursor, min(inicio, fim_exclusivo)))
        cursor = max(cursor, fim)
        if cursor >= fim_exclusivo:
            break
    if cursor < fim_exclusivo:
        lacunas.append((cursor, fim_exclusivo))
    lacunas = [(inicio, fim) for inicio, fim in lacunas if fim > inicio]

    erros = [
        f"{estacao}: ausência de ISMR entre {inicio.date()} e {fim.date()}"
        for inicio, fim in lacunas
    ]
    return (
        {
            "arquivos_encontrados": len(arquivos),
            "arquivos_no_periodo": len(intervalos),
            "bytes_no_periodo": int(
                sum(
                    caminho.stat().st_size
                    for caminho in arquivos
                    if (intervalo := _intervalo_nome(caminho)) is not None
                    and intervalo[1] > inicio_requerido
                    and intervalo[0] < fim_exclusivo
                )
            ),
            "intervalos": [
                {"inicio": str(inicio), "fim_exclusivo": str(fim)}
                for inicio, fim in fundidos
            ],
            "lacunas": [
                {"inicio": str(inicio), "fim_exclusivo": str(fim)}
                for inicio, fim in lacunas
            ],
            "cobertura_completa": not lacunas,
        },
        erros,
    )


def _auditar_clima() -> tuple[dict[str, Any], list[str]]:
    relatorio = {
        "arquivos": [str(caminho) for caminho in SPACE_WEATHER_FILES],
        "cobertura_completa": not SPACE_WEATHER_FILES,
        "modo": "sem_clima_espacial" if not SPACE_WEATHER_FILES else "com_clima_espacial",
    }
    if not SPACE_WEATHER_FILES:
        return relatorio, []
    ausentes = [str(caminho) for caminho in SPACE_WEATHER_FILES if not caminho.is_file()]
    if ausentes:
        relatorio["arquivos_ausentes"] = ausentes
        return relatorio, ["Clima espacial ausente: " + ", ".join(ausentes)]

    try:
        modulo = runpy.run_path(str(SCRIPT_DIRECTORY / "05_integrar_clima_espacial.py"))
        quadros = [
            modulo["carregar_clima_espacial"](caminho)
            for caminho in SPACE_WEATHER_FILES
        ]
        padronizado = modulo["padronizar_clima_espacial"](
            pd.concat(quadros, ignore_index=True)
        )
    except Exception as erro:  # a mensagem detalhada é registrada no preflight
        relatorio["erro"] = f"{type(erro).__name__}: {erro}"
        return relatorio, [f"Clima espacial inválido: {erro}"]

    relatorio.update(
        {
            "cobertura_completa": True,
            "epocas_30min": len(padronizado),
            "inicio": str(padronizado["datetime"].min()),
            "fim": str(padronizado["datetime"].max()),
        }
    )
    return relatorio, []


def _salvar_atomico(conteudo: dict[str, Any]) -> None:
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    descritor, temporario = tempfile.mkstemp(
        prefix=".preflight_", suffix=".tmp", dir=SAIDA.parent
    )
    try:
        with os.fdopen(descritor, "w", encoding="utf-8") as arquivo:
            json.dump(conteudo, arquivo, ensure_ascii=False, indent=2)
        os.replace(temporario, SAIDA)
    except Exception:
        if os.path.exists(temporario):
            os.remove(temporario)
        raise


def main() -> None:
    erros: list[str] = []
    avisos: list[str] = []
    estacoes: dict[str, Any] = {}
    for estacao in STATIONS:
        estacoes[estacao], erros_estacao = _auditar_estacao(estacao)
        falhas, alertas = politica_lacunas(estacoes[estacao], erros_estacao)
        erros.extend(falhas)
        avisos.extend(alertas)

    clima, erros_clima = _auditar_clima()
    erros.extend(erros_clima)
    dependencias_ausentes = [
        nome for nome in DEPENDENCIAS if importlib.util.find_spec(nome) is None
    ]
    if dependencias_ausentes:
        erros.append("Dependências ausentes: " + ", ".join(dependencias_ausentes))

    uso_disco = shutil.disk_usage(PROJECT_ROOT)
    minimo_gb = float(os.environ.get("GNSS_MIN_FREE_SPACE_GB", "20"))
    livre_gb = uso_disco.free / 1024**3
    if livre_gb < minimo_gb:
        erros.append(
            f"Espaço livre insuficiente: {livre_gb:.1f} GiB; mínimo {minimo_gb:.1f} GiB."
        )

    relatorio = {
        "periodo": {
            "inicio": STUDY_START,
            "fim": TEST_END,
        },
        "estacoes": estacoes,
        "clima_espacial": clima,
        "dependencias_ausentes": dependencias_ausentes,
        "disco_livre_gib": round(livre_gb, 2),
        "disco_minimo_gib": minimo_gb,
        "erros": erros,
        "avisos": avisos,
        "lacunas_de_desenvolvimento_permitidas": ALLOW_ISMR_GAPS,
        "aprovado": not erros,
    }
    _salvar_atomico(relatorio)
    linhas_lacunas = []
    for estacao, auditoria in estacoes.items():
        for lacuna in auditoria["lacunas"]:
            linhas_lacunas.append(
                {
                    "tipo": "ISMR",
                    "estacao": estacao,
                    "inicio": lacuna["inicio"],
                    "fim_exclusivo": lacuna["fim_exclusivo"],
                    "observacao": "Arquivo ISMR ausente no intervalo",
                }
            )
    if not clima.get("cobertura_completa"):
        linhas_lacunas.append(
            {
                "tipo": "clima_espacial",
                "estacao": "TODAS",
                "inicio": "",
                "fim_exclusivo": "",
                "observacao": clima.get("erro", "Cobertura ausente"),
            }
        )
    pd.DataFrame(
        linhas_lacunas,
        columns=["tipo", "estacao", "inicio", "fim_exclusivo", "observacao"],
    ).to_csv(SAIDA_LACUNAS, index=False)
    print("Pré-validação salva:", SAIDA)
    for aviso in avisos:
        print("Aviso (experimento com dados disponíveis):", aviso)
    if erros:
        print("Falhas encontradas:")
        for erro in erros:
            print(" -", erro)
        raise RuntimeError(
            f"Pré-validação reprovada com {len(erros)} falha(s). "
            "Nenhum treinamento foi iniciado."
        )
    print("Pré-validação aprovada para:", ", ".join(STATIONS))


def politica_lacunas(auditoria: dict, erros: list[str]) -> tuple[list[str], list[str]]:
    """Aceita lacunas pré-teste somente quando declaradas no experimento."""
    permitido = (
        ALLOW_ISMR_GAPS
        and auditoria["arquivos_no_periodo"] > 0
        and all(pd.Timestamp(gap["fim_exclusivo"]) <= pd.Timestamp(TEST_START)
                for gap in auditoria["lacunas"])
    )
    return ([], erros) if permitido else (erros, [])


if __name__ == "__main__":
    main()
