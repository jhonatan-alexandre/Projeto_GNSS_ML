# APÊNDICE A — CÓDIGOS E ARQUIVOS DOS EXPERIMENTOS

Este apêndice identifica os códigos e arquivos associados aos experimentos,
complementando os procedimentos descritos na metodologia. O repositório publica
o código e os arquivos necessários para descrever e executar os fluxos; não
publica os modelos treinados, logs nem resultados completos das execuções
históricas.

## A.1 Acesso ao código

Repositório: <https://github.com/jhonatan-alexandre/Projeto_GNSS_ML>.

O histórico Git deste projeto começa com a publicação do código reunido neste
repositório. O material local não continha tags ou commits que associassem
separadamente as execuções históricas a versões específicas do código. Por
isso, não se atribui retrospectivamente um hash de commit ou uma tag distinta a
cada experimento. As configurações abaixo identificam os protocolos registrados
no código publicado; consulte o histórico de commits do repositório para
identificar a versão pública utilizada em uma reprodução posterior.

**Experimento 1 — protocolo original do Estudo Dirigido.** Configuração
principal: `configs/estudo_dirigido.json`. Execução: `scripts/12_multistation_pipeline.py`.
O protocolo usa resolução de 30 minutos, desenvolvimento de 01/01/2024 a
26/03/2024, gap em 27/03/2024 e teste final em 28/03/2024, nas estações FRTZ,
PALM, PRU2 e POAL.

**Experimento 2 — comparação comum com teste em 18/02/2024.** Configurações:
`configs/xie_frtz_comum_18fev2024.json`,
`configs/xie_palm_comum_18fev2024.json`,
`configs/xie_poal_comum_18fev2024.json` e
`configs/xie_pru2_comum_18fev2024.json`. Execução e módulos de apoio:
`scripts/18_continuar_fevereiro.py` e `scripts/fevereiro/`. O protocolo usa
histórico comum desde dezembro de 2023, desenvolvimento até 16/02/2024 e
avaliação em 18/02/2024. Trata-se de um experimento derivado, não de uma
reprodução literal do teste de março.

As instruções de instalação, preparação dos dados, validação de insumos e
execução encontram-se em [`README.md`](../README.md) e
[`CHECKLIST_DOWNLOAD_SEMANAL.md`](CHECKLIST_DOWNLOAD_SEMANAL.md). Os parâmetros
metodológicos e as limitações estão em
[`COERENCIA_METODOLOGICA.md`](COERENCIA_METODOLOGICA.md). As dependências estão
declaradas como faixas de versão em `requirements.txt`; não há arquivo de lock
nem manifesto que fixe as versões exatas instaladas nas execuções históricas.

## A.2 Localização dos arquivos

**Quadro A.1 — Arquivos de suporte aos experimentos**

| Material | Experimento 1 | Experimento 2 |
|---|---|---|
| Configurações utilizadas | `configs/estudo_dirigido.json` | `configs/xie_*_comum_18fev2024.json` (uma configuração por estação) |
| Código principal de execução | `scripts/12_multistation_pipeline.py` | `scripts/18_continuar_fevereiro.py`; módulos em `scripts/fevereiro/` |
| Registro das execuções | Logs gerados localmente em `resultados/logs/`; não publicados | Logs da execução local em `estudo_20240218/logs/`; não publicados |
| Previsões e valores de referência | Dados gerados localmente em `resultados/<ESTAÇÃO>/`; não publicados | Dados gerados localmente em `estudo_20240218/estacoes/` e `experimentos/`; não publicados |
| Métricas consolidadas | Resultados locais em `resultados/<ESTAÇÃO>/nucleo1/` e `nucleo2/`; não publicados | Tabelas locais em `estudo_20240218/artigo/tabelas/`; não publicadas |
| Código de geração de tabelas e figuras | `scripts/11_generate_report.py`, `scripts/13_generate_article_assets.py` e scripts `15` a `17` | `scripts/fevereiro/` e `scripts/13_generate_article_assets.py` |
| Dependências e versões das bibliotecas | `requirements.txt` (faixas de versões) | `requirements.txt` (faixas de versões) |

O repositório contém todos os arquivos Python sob `scripts/` e as configurações
JSON sob `configs/`. Os caminhos de resultados no
quadro indicam destinos produzidos localmente; esses resultados não fazem parte
da publicação. O `.gitignore` exclui diretórios locais de dados processados,
modelos, resultados e experimentos para evitar o envio acidental desses
artefatos.

Fonte: O autor, 2026.

## A.3 Disponibilidade dos dados

Os arquivos de clima espacial presentes em `Geomagnéticos/` acompanham o
repositório. São arquivos horários em formato texto, com ano, dia do ano, hora
e valores usados para Kp, número de manchas solares, Dst, Ap e F10.7. O
integrador documenta o formato legado como exportação compatível com NASA
OMNIWeb. As cópias locais não contêm metadados de origem individuais; antes de
redistribuir esses dados separadamente, confirme a origem exata e as condições
aplicáveis. A NASA recomenda citar a fonte original e validar as condições de
uso dos dados, que podem variar conforme o provedor
([política de dados científicos da NASA](https://science.data.nasa.gov/about/license);
[NASA OMNIWeb](https://omniweb.gsfc.nasa.gov/)).

Os arquivos ISMR não acompanham o repositório. Devem ser obtidos no portal
[ISMR Query Tool da FCT/UNESP](https://ismrquerytool.fct.unesp.br/), com acesso e
autenticação manual conforme exigido pelo portal. O script
`scripts/00_baixar_ismr_portal.py` auxilia a localizar e baixar as consultas
faltantes. Para o fluxo documentado, os arquivos são esperados localmente em
`../Dados/Dados ismr 2024/<ESTAÇÃO>/`; eles não são enviados ao GitHub.
Consulte as condições e atribuições definidas pelo portal e pelos provedores
antes de redistribuir os arquivos obtidos.

Os arquivos processados, modelos, previsões, métricas e logs das execuções
históricas também não são distribuídos. Para reproduzir os experimentos, obtenha
os dados de entrada nas fontes indicadas, instale as dependências e execute as
instruções do README. A visibilidade pública do código não concede, por si só,
licença de reutilização do software: esta publicação não inclui um arquivo de
licença de software.
