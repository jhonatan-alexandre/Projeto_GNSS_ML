# Checklist de download — reprodução do relatório

## Período oficial

- Desenvolvimento: 01/01/2024 00:00 a 26/03/2024 23:59.
- Gap temporal: 27/03/2024.
- Teste final: 28/03/2024 00:00 a 23:59.
- Estações: FRTZ, PALM, PRU2 e POAL.

O limite final das consultas é exclusivo. Portanto, `29/03/2024 00:00` cobre
todo o dia de teste de 28/03.

## Consultas ISMR

Faça as 13 consultas abaixo para cada estação, totalizando 52 consultas. O
script ignora automaticamente uma consulta quando encontra um arquivo local que
cobre todo o intervalo.

| Nº | Início UTC | Fim UTC |
|---:|---:|---:|
| 1 | 2024-01-01 00:00 | 2024-01-08 00:00 |
| 2 | 2024-01-08 00:00 | 2024-01-15 00:00 |
| 3 | 2024-01-15 00:00 | 2024-01-22 00:00 |
| 4 | 2024-01-22 00:00 | 2024-01-29 00:00 |
| 5 | 2024-01-29 00:00 | 2024-02-05 00:00 |
| 6 | 2024-02-05 00:00 | 2024-02-12 00:00 |
| 7 | 2024-02-12 00:00 | 2024-02-19 00:00 |
| 8 | 2024-02-19 00:00 | 2024-02-26 00:00 |
| 9 | 2024-02-26 00:00 | 2024-03-04 00:00 |
| 10 | 2024-03-04 00:00 | 2024-03-11 00:00 |
| 11 | 2024-03-11 00:00 | 2024-03-18 00:00 |
| 12 | 2024-03-18 00:00 | 2024-03-25 00:00 |
| 13 | 2024-03-25 00:00 | 2024-03-29 00:00 |

Conferir o manifesto:

```powershell
.venv\Scripts\python scripts/00_baixar_ismr_portal.py
```

Executar o download assistido:

```powershell
.venv\Scripts\python scripts/00_baixar_ismr_portal.py --executar
```

Os arquivos são armazenados em:

```text
../Dados/Dados ismr 2024/<ESTAÇÃO>/
```

## Conteúdo obrigatório do ISMR

Baixe a exportação completa, não apenas figuras ou resumos. Ela deve permitir a
obtenção de:

- instante UTC;
- SVID;
- elevação e azimute;
- S4 e correção de ruído térmico, quando disponível;
- VTEC direto ou TEC/STEC para conversão pelo modelo de camada fina.

Extensões aceitas: `.ismr` e `.ismr.gz`.

## Clima espacial

O arquivo deve cobrir 01/01/2024 00:00 a 28/03/2024 23:30 e conter:

- Kp;
- Dst;
- F10.7;
- Ap;
- SSN.

O arquivo legado esperado é
`Geomagnéticos/dados_solar_geomag_2024.txt`. Também são aceitos
`dados_externos/omniweb.csv` ou arquivos `omniweb_*.csv`.

## Verificação antes do treinamento

```powershell
.venv\Scripts\python scripts/01_validar_insumos.py
```

O treinamento só deve começar quando o preflight confirmar:

- cobertura integral das quatro estações;
- cobertura integral do clima espacial;
- dependências instaladas;
- espaço livre suficiente.

## Execução do experimento

```powershell
.venv\Scripts\python scripts/12_multistation_pipeline.py
```

Esse comando usa os hiperparâmetros completos do relatório: 150 árvores no
XGBoost e até 50 épocas nas LSTM, com patience 5. Não existe modo piloto no
fluxo oficial.
