"""Localiza somente os arquivos ISMR necessários ao recorte do estudo."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import tempfile


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from config import (  # noqa: E402
    ISMR_DIRECTORIES,
    ISMR_FILE_EXTENSIONS,
    PROCESSED_DATA,
    STATION,
    STUDY_START,
    TEST_END,
)


PASTA_ESTACAO = Path(PROCESSED_DATA) / STATION
ARQUIVO_SAIDA = PASTA_ESTACAO / "lista_arquivos_ismr.txt"
PADRAO_DATA = re.compile(r"\d{4}-\d{2}-\d{2}")


def _sobrepoe_periodo(nome: str) -> bool:
    """Usa as duas primeiras datas do nome como início/fim de cobertura."""

    datas = PADRAO_DATA.findall(nome)
    if len(datas) < 2:
        # Um nome não padronizado não é descartado sem inspecionar seu conteúdo.
        return True
    inicio_arquivo = datas[0]
    fim_arquivo = datas[1]
    inicio_estudo = str(STUDY_START)[:10]
    fim_estudo = str(TEST_END)[:10]
    return inicio_arquivo <= fim_estudo and fim_arquivo >= inicio_estudo


def localizar_arquivos(estacao: str = STATION) -> list[Path]:
    prefixo = estacao.upper() + "_"
    extensoes = tuple(extensao.lower() for extensao in ISMR_FILE_EXTENSIONS)
    encontrados: set[Path] = set()

    for diretorio in ISMR_DIRECTORIES:
        raiz = Path(diretorio)
        if not raiz.is_dir():
            continue
        pasta_estacao = raiz / estacao.upper()
        bases = (pasta_estacao,) if pasta_estacao.is_dir() else (raiz,)
        for base in bases:
            for caminho in base.rglob("*"):
                nome = caminho.name
                if (
                    caminho.is_file()
                    and nome.upper().startswith(prefixo)
                    and nome.lower().endswith(extensoes)
                    and _sobrepoe_periodo(nome)
                ):
                    encontrados.add(caminho.resolve())

    return sorted(encontrados, key=lambda caminho: caminho.name)


def salvar_lista(arquivos: list[Path], destino: Path = ARQUIVO_SAIDA) -> None:
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum ISMR de {STATION} cobre {STUDY_START} a {TEST_END}."
        )
    destino.parent.mkdir(parents=True, exist_ok=True)
    descritor, temporario = tempfile.mkstemp(
        prefix=".lista_", suffix=".tmp", dir=destino.parent
    )
    try:
        with os.fdopen(descritor, "w", encoding="utf-8", newline="\n") as arquivo:
            arquivo.writelines(f"{caminho}\n" for caminho in arquivos)
        os.replace(temporario, destino)
    except Exception:
        if os.path.exists(temporario):
            os.remove(temporario)
        raise


def main() -> None:
    arquivos = localizar_arquivos()
    salvar_lista(arquivos)
    print(f"Estação: {STATION}")
    print(f"Arquivos no período do estudo: {len(arquivos)}")
    print("Lista salva:", ARQUIVO_SAIDA)


if __name__ == "__main__":
    main()
