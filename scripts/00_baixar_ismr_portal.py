"""Baixa, com login manual, os ISMR ausentes no portal da FCT/UNESP.

O navegador é usado somente para o usuário autenticar-se no site oficial e
resolver o Cloudflare Turnstile. A senha nunca é lida pelo script. Depois da
confirmação manual, o token de sessão é lido do perfil temporário, não aparece
em logs nem é copiado para o projeto e é enviado ao mesmo endpoint usado pela
interface oficial. O perfil temporário é descartado ao encerrar.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable
from urllib.parse import unquote
import zipfile

try:
    import truststore
except ImportError as erro:
    raise RuntimeError(
        "Dependência truststore ausente. Execute: "
        ".venv\\Scripts\\python -m pip install -r requirements.txt"
    ) from erro

# Usa as mesmas autoridades certificadoras confiáveis do Windows/Edge. Isso
# corrige cadeias com intermediário no repositório do sistema sem enfraquecer
# a validação TLS e sem recorrer a verify=False.
truststore.inject_into_ssl()

import requests


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from config import (  # noqa: E402
    DATA_ROOT,
    PROJECT_ROOT,
    RESULTS_DIRECTORY,
    STATIONS,
    STUDY_START,
    TEST_END,
)


PORTAL_URL = "https://ismrquerytool.fct.unesp.br"
API_URL = "https://api-ismrquerytool.fct.unesp.br"
ENDPOINT_DOWNLOAD = "/api/v1/data/download/ismr/file"
ENDPOINT_USUARIO = "/api/v1/user/token"
ENDPOINT_ESTACOES = "/api/v1/station/list-all"
FORMATO_API = "%Y-%m-%d %H:%M"
EXTENSOES_ISMR = (".ismr", ".ismr.gz")
PADRAO_DATA = re.compile(r"\d{4}-\d{2}-\d{2}")
TAMANHO_BLOCO = 1024 * 1024

# O fim das consultas é exclusivo. O preset oficial resolve este intervalo
# como 01/01/2024--29/03/2024; configurações derivadas usam o próprio recorte.
_download_start = datetime.fromisoformat(STUDY_START).date().isoformat()
_download_end = (
    datetime.fromisoformat(TEST_END).date() + timedelta(days=1)
).isoformat()
BLOCOS_EXPERIMENTO = {
    estacao: ((_download_start, _download_end),) for estacao in STATIONS
}


class SessaoExpirada(RuntimeError):
    """Indica que o login deve ser repetido sem perder os downloads concluídos."""


@dataclass(frozen=True)
class Consulta:
    estacao: str
    inicio: datetime
    fim: datetime

    @property
    def chave(self) -> str:
        return (
            f"{self.estacao}_{self.inicio:%Y-%m-%dT%H_%M}_"
            f"{self.fim:%Y-%m-%dT%H_%M}"
        )

    def parametros(self) -> dict[str, str]:
        return {
            "station": self.estacao,
            "start": self.inicio.strftime(FORMATO_API),
            "end": self.fim.strftime(FORMATO_API),
        }


def dividir_em_semanas(inicio: datetime, fim: datetime) -> Iterable[tuple[datetime, datetime]]:
    """Divide [início, fim] em consultas contíguas de no máximo sete dias."""

    if fim <= inicio:
        raise ValueError("O fim do intervalo precisa ser posterior ao início.")
    atual = inicio
    while atual < fim:
        seguinte = min(atual + timedelta(days=7), fim)
        yield atual, seguinte
        atual = seguinte


def criar_manifesto(estacoes: Iterable[str] | None = None) -> list[Consulta]:
    selecionadas = (
        {item.strip().upper() for item in estacoes if item.strip()}
        if estacoes is not None
        else set(BLOCOS_EXPERIMENTO)
    )
    desconhecidas = selecionadas.difference(BLOCOS_EXPERIMENTO)
    if desconhecidas:
        raise ValueError(
            "Estações sem manifesto: " + ", ".join(sorted(desconhecidas))
        )

    manifesto: list[Consulta] = []
    for estacao, blocos in BLOCOS_EXPERIMENTO.items():
        if estacao not in selecionadas:
            continue
        for inicio_texto, fim_texto in blocos:
            inicio = datetime.fromisoformat(inicio_texto)
            fim = datetime.fromisoformat(fim_texto)
            manifesto.extend(
                Consulta(estacao, parte_inicio, parte_fim)
                for parte_inicio, parte_fim in dividir_em_semanas(inicio, fim)
            )
    return manifesto


def pasta_destino(consulta: Consulta, raiz_dados: Path) -> Path:
    return raiz_dados / f"Dados ismr {consulta.inicio.year}" / consulta.estacao


def _datas_no_nome(caminho: Path) -> tuple[datetime, datetime] | None:
    datas = PADRAO_DATA.findall(caminho.name)
    if len(datas) < 2:
        return None
    return datetime.fromisoformat(datas[0]), datetime.fromisoformat(datas[1])


def localizar_cobertura_existente(consulta: Consulta, raiz_dados: Path) -> Path | None:
    """Evita repetir uma consulta já coberta por um arquivo local."""

    pasta = raiz_dados / "Dados ismr 2024" / consulta.estacao
    if pasta.is_dir():
        for caminho in pasta.rglob("*"):
            if not caminho.is_file() or not caminho.name.lower().endswith(EXTENSOES_ISMR):
                continue
            intervalo = _datas_no_nome(caminho)
            if intervalo and intervalo[0] <= consulta.inicio and intervalo[1] >= consulta.fim:
                return caminho
    return None


def _carregar_status(caminho: Path) -> dict[str, Any]:
    if not caminho.is_file():
        return {}
    try:
        with caminho.open("r", encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
    except (OSError, json.JSONDecodeError):
        return {}
    return conteudo if isinstance(conteudo, dict) else {}


def _salvar_status(caminho: Path, status: dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    descritor, temporario = tempfile.mkstemp(
        prefix=".download_ismr_", suffix=".tmp", dir=caminho.parent
    )
    try:
        with os.fdopen(descritor, "w", encoding="utf-8") as arquivo:
            json.dump(status, arquivo, ensure_ascii=False, indent=2)
        os.replace(temporario, caminho)
    except Exception:
        if os.path.exists(temporario):
            os.remove(temporario)
        raise


def _registro_concluido(registro: Any) -> bool:
    if not isinstance(registro, dict) or registro.get("situacao") != "concluido":
        return False
    arquivos = registro.get("arquivos", [])
    return bool(arquivos) and all(Path(item).is_file() for item in arquivos)


def _espaco_livre_gib(caminho: Path) -> float:
    caminho.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(caminho).free / (1024**3)


def _executavel_navegador(preferencia: str) -> Path:
    """Localiza Chrome/Edge sem reutilizar o perfil pessoal do usuário."""

    programas = Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
    programas_x86 = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    candidatos = {
        "chrome": (
            programas / "Google/Chrome/Application/chrome.exe",
            local / "Google/Chrome/Application/chrome.exe",
        ),
        "msedge": (
            programas_x86 / "Microsoft/Edge/Application/msedge.exe",
            programas / "Microsoft/Edge/Application/msedge.exe",
            local / "Microsoft/Edge/Application/msedge.exe",
        ),
    }
    # Edge primeiro evita que uma instância pessoal já aberta do Chrome tente
    # absorver a nova janela e descarte a porta de depuração do perfil dedicado.
    ordem = ("msedge", "chrome") if preferencia == "auto" else (preferencia,)
    for navegador in ordem:
        for caminho in candidatos[navegador]:
            if caminho.is_file():
                return caminho.resolve()
    raise FileNotFoundError("Chrome ou Microsoft Edge não encontrado.")


def _porta_local_livre() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as servidor:
        servidor.bind(("127.0.0.1", 0))
        return int(servidor.getsockname()[1])


def _iniciar_navegador_manual(executavel: Path, perfil: Path, porta: int) -> subprocess.Popen:
    """Abre navegador comum; Playwright ainda não está conectado nesta etapa."""

    argumentos = (
        str(executavel),
        f"--remote-debugging-port={porta}",
        f"--user-data-dir={perfil}",
        "--no-first-run",
        "--no-default-browser-check",
        "--new-window",
        f"{PORTAL_URL}/login",
    )
    return subprocess.Popen(argumentos)


def _esperar_depurar_navegador(porta: int, processo: subprocess.Popen, segundos: int = 30) -> None:
    url = f"http://127.0.0.1:{porta}/json/version"
    limite = time.monotonic() + segundos
    while time.monotonic() < limite:
        if processo.poll() is not None:
            raise RuntimeError("O navegador foi fechado antes da autenticação.")
        try:
            resposta = requests.get(url, timeout=1)
            if resposta.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise TimeoutError("O navegador não disponibilizou a sessão local de depuração.")


def _aguardar_login_manual(browser: Any, contexto: Any, minutos: float) -> tuple[str, Any]:
    """Procura o token em todas as abas, somente após confirmação manual."""

    limite = time.monotonic() + minutos * 60
    while time.monotonic() < limite:
        if not browser.is_connected():
            raise RuntimeError(
                "O navegador dedicado foi fechado ou desconectado. "
                "Interrompa com Ctrl+C e inicie novamente."
            )
        paginas = [pagina for pagina in contexto.pages if not pagina.is_closed()]
        if not paginas:
            raise RuntimeError("Todas as abas do navegador dedicado foram fechadas.")
        urls_portal: list[str] = []
        for page in paginas:
            if not page.url.startswith(PORTAL_URL):
                continue
            urls_portal.append(page.url)
            try:
                token = page.evaluate("() => localStorage.getItem('accessToken')")
            except Exception:
                token = None
            if token:
                print("Autenticação detectada; iniciando as consultas.")
                return str(token), page
        print("O login ainda não foi confirmado pelo portal.")
        if urls_portal:
            print("Abas do portal detectadas:", "; ".join(urls_portal))
        else:
            print("Nenhuma aba do ISMR Query Tool foi encontrada nessa janela.")
        input("Conclua login/CAPTCHA nessa janela e pressione Enter novamente...")
    raise TimeoutError(f"Login não concluído em {minutos:g} minutos.")


def _esperar_login(page: Any, minutos: float, token_anterior: str | None = None) -> str:
    """Espera o usuário autenticar-se; não lê campos nem registra credenciais."""

    if token_anterior:
        page.evaluate(
            """() => {
                localStorage.removeItem('accessToken');
                localStorage.removeItem('expiresAt');
            }"""
        )
    page.goto(f"{PORTAL_URL}/login", wait_until="domcontentloaded")
    print("\nFaça o login e resolva o Turnstile na janela oficial do portal.")
    print("A senha não é acessada pelo programa. Aguardando autenticação...")
    limite = time.monotonic() + minutos * 60
    while time.monotonic() < limite:
        if page.is_closed():
            raise RuntimeError("A janela do navegador foi fechada antes do login.")
        try:
            token = page.evaluate("() => localStorage.getItem('accessToken')")
        except Exception:
            token = None
        if token and token != token_anterior:
            print("Autenticação detectada; iniciando as consultas.")
            return str(token)
        page.wait_for_timeout(1000)
    raise TimeoutError(f"Login não concluído em {minutos:g} minutos.")


def _configurar_sessao(token: str) -> requests.Session:
    sessao = requests.Session()
    sessao.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json, application/zip, application/octet-stream",
            "User-Agent": "GNSS-ML-research-downloader/1.0",
        }
    )
    return sessao


def _validar_conta_e_estacoes(
    sessao: requests.Session,
    page: Any,
    estacoes: Iterable[str],
    minutos: float,
) -> None:
    resposta_usuario = sessao.get(f"{API_URL}{ENDPOINT_USUARIO}", timeout=30)
    if resposta_usuario.status_code == 401:
        raise SessaoExpirada("A sessão expirou antes da primeira consulta.")
    resposta_usuario.raise_for_status()
    usuario = resposta_usuario.json()
    if usuario.get("accepted_terms") is False:
        page.goto(PORTAL_URL, wait_until="domcontentloaded")
        print("Leia e aceite os termos de uso na janela oficial para continuar.")
        limite = time.monotonic() + minutos * 60
        while time.monotonic() < limite:
            page.wait_for_timeout(1500)
            verificacao = sessao.get(f"{API_URL}{ENDPOINT_USUARIO}", timeout=30)
            if verificacao.ok and verificacao.json().get("accepted_terms") is not False:
                break
        else:
            raise TimeoutError("Os termos de uso não foram aceitos dentro do prazo.")

    resposta_estacoes = sessao.get(f"{API_URL}{ENDPOINT_ESTACOES}", timeout=30)
    resposta_estacoes.raise_for_status()
    bruto = resposta_estacoes.json().get("stations", [])
    valores = bruto.values() if isinstance(bruto, dict) else bruto
    disponiveis = {
        str(item.get("name", "")).upper()
        for item in valores
        if isinstance(item, dict)
    }
    ausentes = set(estacoes).difference(disponiveis)
    if ausentes:
        raise ValueError(
            "Estações não oferecidas atualmente pelo portal: "
            + ", ".join(sorted(ausentes))
        )


def _mensagem_erro_http(resposta: requests.Response) -> str:
    try:
        corpo = resposta.json()
        if isinstance(corpo, dict):
            return str(corpo.get("message") or corpo.get("detail") or corpo)
    except (requests.exceptions.JSONDecodeError, ValueError):
        pass
    texto = resposta.text.strip().replace("\n", " ")
    return texto[:500] or resposta.reason


def _nome_servidor(resposta: requests.Response) -> str | None:
    disposicao = resposta.headers.get("Content-Disposition", "")
    utf8 = re.search(r"filename\*=UTF-8''([^;]+)", disposicao, re.IGNORECASE)
    if utf8:
        return unquote(utf8.group(1)).strip('"')
    simples = re.search(r'filename="?([^";]+)', disposicao, re.IGNORECASE)
    return simples.group(1).strip() if simples else None


def _baixar_zip(
    sessao: requests.Session,
    consulta: Consulta,
    destino: Path,
    tentativas: int,
) -> tuple[Path, int, str | None]:
    destino.mkdir(parents=True, exist_ok=True)
    arquivo_final = destino / f".{consulta.chave}.zip"
    arquivo_parcial = destino / f".{consulta.chave}.zip.part"
    url = f"{API_URL}{ENDPOINT_DOWNLOAD}"

    for tentativa in range(1, tentativas + 1):
        if arquivo_parcial.exists():
            arquivo_parcial.unlink()
        try:
            with sessao.get(
                url,
                params=consulta.parametros(),
                stream=True,
                timeout=(30, 1800),
            ) as resposta:
                if resposta.status_code == 401:
                    raise SessaoExpirada("Sessão expirada durante o download.")
                if resposta.status_code == 429 or resposta.status_code >= 500:
                    mensagem = _mensagem_erro_http(resposta)
                    raise requests.HTTPError(
                        f"HTTP {resposta.status_code}: {mensagem}", response=resposta
                    )
                if not resposta.ok:
                    mensagem = _mensagem_erro_http(resposta)
                    raise RuntimeError(f"HTTP {resposta.status_code}: {mensagem}")

                esperado = int(resposta.headers.get("Content-Length", "0") or 0)
                if esperado and shutil.disk_usage(destino).free < esperado * 2:
                    raise OSError(
                        "Espaço insuficiente para armazenar e extrair o ZIP "
                        f"de {esperado / (1024**3):.2f} GiB."
                    )

                total = 0
                ultima_exibicao = time.monotonic()
                with arquivo_parcial.open("wb") as arquivo:
                    for bloco in resposta.iter_content(chunk_size=TAMANHO_BLOCO):
                        if not bloco:
                            continue
                        arquivo.write(bloco)
                        total += len(bloco)
                        if time.monotonic() - ultima_exibicao >= 10:
                            if esperado:
                                percentual = 100 * total / esperado
                                print(
                                    f"    {total / (1024**2):.1f} MiB / "
                                    f"{esperado / (1024**2):.1f} MiB "
                                    f"({percentual:.1f}%)"
                                )
                            else:
                                print(f"    {total / (1024**2):.1f} MiB recebidos")
                            ultima_exibicao = time.monotonic()

                if esperado and total != esperado:
                    raise OSError(
                        f"Download incompleto: {total} de {esperado} bytes."
                    )
                os.replace(arquivo_parcial, arquivo_final)
                return arquivo_final, total, _nome_servidor(resposta)
        except SessaoExpirada:
            raise
        except Exception:
            if arquivo_parcial.exists():
                arquivo_parcial.unlink()
            if tentativa >= tentativas:
                raise
            espera = min(60, 5 * 2 ** (tentativa - 1))
            print(
                f"    tentativa {tentativa}/{tentativas} falhou; "
                f"nova tentativa em {espera} s"
            )
            time.sleep(espera)
    raise AssertionError("Fluxo de tentativas inconsistente.")


def _extrair_zip_seguro(arquivo_zip: Path, destino: Path) -> list[Path]:
    if not zipfile.is_zipfile(arquivo_zip):
        raise zipfile.BadZipFile("O portal não retornou um ZIP válido.")

    extraidos: list[Path] = []
    raiz = destino.resolve()
    with zipfile.ZipFile(arquivo_zip) as pacote:
        membros = [
            item
            for item in pacote.infolist()
            if not item.is_dir() and item.filename.lower().endswith(EXTENSOES_ISMR)
        ]
        if not membros:
            raise ValueError("O ZIP não contém arquivos .ismr ou .ismr.gz.")
        tamanho_extraido = sum(item.file_size for item in membros)
        if shutil.disk_usage(destino).free < tamanho_extraido:
            raise OSError(
                "Espaço insuficiente para extrair "
                f"{tamanho_extraido / (1024**3):.2f} GiB de ISMR."
            )

        for membro in membros:
            relativo = Path(membro.filename.replace("\\", "/"))
            if relativo.is_absolute() or ".." in relativo.parts:
                raise ValueError(f"Caminho inseguro dentro do ZIP: {membro.filename}")
            alvo = (destino / relativo).resolve()
            if raiz != alvo.parent and raiz not in alvo.parents:
                raise ValueError(f"Caminho fora do destino: {membro.filename}")
            alvo.parent.mkdir(parents=True, exist_ok=True)
            if alvo.exists():
                if alvo.stat().st_size == membro.file_size:
                    extraidos.append(alvo)
                    continue
                raise FileExistsError(
                    f"Já existe um arquivo diferente no destino: {alvo}"
                )
            temporario = alvo.with_name(alvo.name + ".part")
            if temporario.exists():
                temporario.unlink()
            try:
                with pacote.open(membro) as origem, temporario.open("xb") as saida:
                    shutil.copyfileobj(origem, saida, length=TAMANHO_BLOCO)
                os.replace(temporario, alvo)
            except Exception:
                if temporario.exists():
                    temporario.unlink()
                raise
            extraidos.append(alvo)
    return extraidos


def _mostrar_manifesto(manifesto: list[Consulta], raiz_dados: Path) -> None:
    contagens: dict[str, int] = {}
    for consulta in manifesto:
        contagens[consulta.estacao] = contagens.get(consulta.estacao, 0) + 1
        print(
            f"{consulta.estacao}: {consulta.inicio:%Y-%m-%d %H:%M} -> "
            f"{consulta.fim:%Y-%m-%d %H:%M} | {pasta_destino(consulta, raiz_dados)}"
        )
    resumo = ", ".join(f"{estacao}={total}" for estacao, total in contagens.items())
    print(f"\nTotal: {len(manifesto)} consultas ({resumo}).")


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download assistido dos ISMR ausentes no portal FCT/UNESP."
    )
    parser.add_argument(
        "--executar",
        action="store_true",
        help="abre o navegador e realiza os downloads; sem esta opção só lista",
    )
    parser.add_argument(
        "--estacoes",
        nargs="+",
        default=None,
        help="subconjunto: FRTZ PALM POAL PRU2",
    )
    parser.add_argument(
        "--raiz-dados",
        type=Path,
        default=Path(DATA_ROOT),
        help="pasta que contém 'Dados ismr 2024'",
    )
    parser.add_argument(
        "--navegador",
        choices=("auto", "chrome", "msedge"),
        default="auto",
    )
    parser.add_argument("--espera-login-min", type=float, default=15.0)
    parser.add_argument("--intervalo-seg", type=float, default=3.0)
    parser.add_argument("--tentativas", type=int, default=3)
    parser.add_argument(
        "--max-consultas",
        type=int,
        default=None,
        help="limita esta execução; útil para validar primeiro arquivo",
    )
    parser.add_argument(
        "--manter-zips",
        action="store_true",
        help="preserva os ZIPs após a extração (consome espaço adicional)",
    )
    return parser.parse_args()


def main() -> int:
    args = _argumentos()
    if args.tentativas <= 0 or args.intervalo_seg < 0 or args.espera_login_min <= 0:
        raise ValueError("Tentativas e espera de login devem ser positivas.")

    raiz_dados = args.raiz_dados.expanduser().resolve()
    manifesto = criar_manifesto(args.estacoes)
    if args.max_consultas is not None:
        if args.max_consultas <= 0:
            raise ValueError("--max-consultas deve ser positivo.")
        manifesto = manifesto[: args.max_consultas]
    _mostrar_manifesto(manifesto, raiz_dados)

    if not args.executar:
        print("\nSimulação concluída. Nenhum login ou download foi realizado.")
        print("Acrescente --executar para iniciar com autenticação manual.")
        return 0

    espaco = _espaco_livre_gib(raiz_dados)
    minimo = float(os.environ.get("GNSS_MIN_FREE_SPACE_GB", "20"))
    if espaco < minimo:
        raise OSError(
            f"Há somente {espaco:.1f} GiB livres em {raiz_dados}; "
            f"são exigidos ao menos {minimo:.1f} GiB. Libere espaço ou use "
            "--raiz-dados em outra unidade."
        )

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as erro:
        raise RuntimeError(
            "Playwright não instalado. Execute: "
            ".venv\\Scripts\\python -m pip install -r requirements.txt"
        ) from erro

    status_path = Path(RESULTS_DIRECTORY) / "download_ismr_status.json"
    status = _carregar_status(status_path)
    falhas: list[str] = []
    concluidas = 0
    ignoradas = 0
    executavel_navegador = _executavel_navegador(args.navegador)

    with tempfile.TemporaryDirectory(
        prefix="ismr_login_", ignore_cleanup_errors=True
    ) as perfil_temporario, sync_playwright() as playwright:
        porta = _porta_local_livre()
        processo_navegador = _iniciar_navegador_manual(
            executavel_navegador, Path(perfil_temporario), porta
        )
        browser = None
        token: str | None = None
        sessao: requests.Session | None = None
        try:
            _esperar_depurar_navegador(porta, processo_navegador)
            print("\nO navegador foi aberto sem controle automatizado.")
            print("Faça o login e resolva o Turnstile diretamente no portal.")
            input("Quando a conta estiver autenticada, volte aqui e pressione Enter...")
            browser = playwright.chromium.connect_over_cdp(
                f"http://127.0.0.1:{porta}"
            )
            if not browser.contexts:
                raise RuntimeError("Não foi possível acessar a sessão do navegador.")
            contexto = browser.contexts[0]
            token, page = _aguardar_login_manual(
                browser, contexto, args.espera_login_min
            )
            sessao = _configurar_sessao(token)
            _validar_conta_e_estacoes(
                sessao,
                page,
                {consulta.estacao for consulta in manifesto},
                args.espera_login_min,
            )

            for numero, consulta in enumerate(manifesto, start=1):
                existente = localizar_cobertura_existente(consulta, raiz_dados)
                if existente or _registro_concluido(status.get(consulta.chave)):
                    origem = existente or "registro de execução anterior"
                    print(f"[{numero}/{len(manifesto)}] {consulta.chave}: já existe ({origem})")
                    ignoradas += 1
                    continue

                print(f"[{numero}/{len(manifesto)}] Baixando {consulta.chave}")
                destino = pasta_destino(consulta, raiz_dados)
                inicio_tentativa = datetime.now().astimezone().isoformat()
                try:
                    while True:
                        try:
                            pacote, bytes_recebidos, nome_servidor = _baixar_zip(
                                sessao, consulta, destino, args.tentativas
                            )
                            break
                        except SessaoExpirada:
                            print("Sessão expirada. Refaça o login na janela oficial.")
                            sessao.close()
                            token = _esperar_login(
                                page, args.espera_login_min, token_anterior=token
                            )
                            sessao = _configurar_sessao(token)
                            _validar_conta_e_estacoes(
                                sessao,
                                page,
                                {consulta.estacao},
                                args.espera_login_min,
                            )

                    extraidos = _extrair_zip_seguro(pacote, destino)
                    if not args.manter_zips:
                        pacote.unlink()
                    status[consulta.chave] = {
                        "situacao": "concluido",
                        "estacao": consulta.estacao,
                        "inicio": consulta.inicio.strftime(FORMATO_API),
                        "fim": consulta.fim.strftime(FORMATO_API),
                        "inicio_tentativa": inicio_tentativa,
                        "conclusao": datetime.now().astimezone().isoformat(),
                        "bytes_zip": bytes_recebidos,
                        "nome_servidor": nome_servidor,
                        "arquivos": [str(caminho) for caminho in extraidos],
                    }
                    _salvar_status(status_path, status)
                    concluidas += 1
                    print(f"    concluído: {len(extraidos)} arquivo(s) extraído(s)")
                except Exception as erro:
                    mensagem = f"{type(erro).__name__}: {erro}"
                    status[consulta.chave] = {
                        "situacao": "falhou",
                        "estacao": consulta.estacao,
                        "inicio": consulta.inicio.strftime(FORMATO_API),
                        "fim": consulta.fim.strftime(FORMATO_API),
                        "inicio_tentativa": inicio_tentativa,
                        "falha": datetime.now().astimezone().isoformat(),
                        "erro": mensagem,
                    }
                    _salvar_status(status_path, status)
                    falhas.append(f"{consulta.chave}: {mensagem}")
                    print(f"    FALHA: {mensagem}")
                if numero < len(manifesto) and args.intervalo_seg:
                    time.sleep(args.intervalo_seg)
        finally:
            if sessao is not None:
                sessao.close()
            if browser is not None:
                browser.close()
            if processo_navegador.poll() is None:
                processo_navegador.terminate()
                try:
                    processo_navegador.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    processo_navegador.kill()

    print(
        f"\nResumo: {concluidas} concluída(s), {ignoradas} já existente(s), "
        f"{len(falhas)} falha(s)."
    )
    print("Status:", status_path)
    if falhas:
        print("Falhas que serão retomadas na próxima execução:")
        for falha in falhas:
            print(" -", falha)
        return 1
    print("Execute agora: .venv\\Scripts\\python scripts/01_validar_insumos.py")
    return 0


if __name__ == "__main__":
    try:
        codigo_saida = main()
    except KeyboardInterrupt:
        print("\nDownload interrompido pelo usuário; os lotes concluídos serão retomados.")
        codigo_saida = 130
    raise SystemExit(codigo_saida)
