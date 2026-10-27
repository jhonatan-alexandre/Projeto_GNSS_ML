# Duplo núcleo GNSS + Machine Learning

Repositório de código e documentação para os experimentos de classificação de
cintilação ionosférica (S4) e previsão de VTEC nas estações FRTZ, PALM, PRU2 e
POAL. O [Apêndice A](docs/APENDICE_A_CODIGOS_E_ARQUIVOS.md) identifica os
protocolos, configurações, fontes de dados e arquivos de saída.

Esta publicação contém scripts, configurações, documentação selecionada e
arquivos auxiliares de clima espacial. Dados ISMR, modelos treinados, logs e
resultados gerados não são distribuídos neste repositório. Consulte
[disponibilidade e preparação dos dados](docs/CHECKLIST_DOWNLOAD_SEMANAL.md)
antes de executar o pipeline.

Reprodução computacional do experimento descrito no relatório de Estudo
Dirigido: classificação de cintilação ionosférica e previsão de VTEC nas
estações FRTZ, PALM, PRU2 e POAL.

## Protocolo original do Estudo Dirigido

- Período de desenvolvimento: 01/01/2024 a 26/03/2024.
- Gap temporal: 27/03/2024.
- Teste final isolado: 28/03/2024.
- Resolução: 30 minutos.
- Estações: FRTZ, PALM, PRU2 e POAL.
- Núcleo 1: previsão de `S4(t + 30 min) >= 0,5` com XGBoost.
- Núcleo 2: previsão do VTEC com LSTM/LSTM-Attention, nas organizações
  convencional e piecewise.

O projeto avalia classificação de S4 e regressão de VTEC. Ele não executa SPP,
PPP nem estima erro de coordenadas GNSS.

## Retreinamento anterior com os hiperparâmetros de Xie

O experimento derivado de setembro de 2026 preserva os resultados anteriores
e usa os períodos com dados disponíveis de cada estação. Execute:

```powershell
.venv\Scripts\python -X utf8 scripts/14_retreinar_xie.py --workers 1 --min-free-space-gb 5
```

O comando lê novamente as pastas ISMR, executa os dois núcleos e gera a
comparação em `experimentos/xie_20260920/relatorio_comparativo.html`.
Use `--preflight-only` para validar entradas e `--resume` para reaproveitar
estações integralmente concluídas, com configuração e completude verificadas;
uma estação incompleta é
preservada em arquivo e reexecutada. O estado fica em `runner_status.json`, junto ao manifesto
de configuração, código e entradas. Os logs e artefatos anteriores permanecem
preservados. As configurações estão em `configs/xie_<estação>.json`.

O modo `--resume` não substitui a auditoria de integridade dos arquivos: ele
confere a proveniência e a completude, mas não recalcula os hashes de todos
os modelos. Na execução de setembro, essa verificação adicional integra a
auditoria final dos checkpoints e das fontes dos gráficos.

O treinamento executa a seleção de taxa e cada configuração LSTM em processos
separados para liberar a memória do TensorFlow. A recuperação dos modelos
parciais da execução interrompida em 20/09 usa evidências explícitas:

```powershell
.venv\Scripts\python -X utf8 scripts/14_retreinar_xie.py --resume-training --recovery-manifest experimentos/xie_20260920/recuperacao_memoria_20260920/manifesto_retomada_verificada.json
```

Esse manifesto é específico daquela interrupção: valida código, dados,
configuração e hashes dos modelos antes de retomar a FRTZ na etapa 10. As
demais estações seguem o fluxo completo. Não reutilize o manifesto depois
que a execução avançar; a validação rejeita estados e arquivos divergentes.

A LSTM passa a usar três camadas de 31 unidades, Dense com LeakyReLU,
dropout 0,2, Adam/MSE, até 50 épocas, batch de até 400, treino/validação 15:1
e early stopping com paciência 3. As seis taxas de aprendizado da tabela de
Xie são comparadas na validação de uma rede convencional multivariada original
por estação; a escolhida é compartilhada na matriz M1–M8. Isso adapta os
hiperparâmetros ao presente estudo; não reproduz integralmente os dados,
atributos geográficos ou experimentos de posicionamento do artigo.

O XGBoost compara quatro configurações e seleciona o limiar por F2 nos quatro
primeiros folds temporais. O quinto fold permanece reservado, assim como o
teste final. Os resultados incluem precisão, recall, F1, F2, average precision
e contagens de eventos. Aumentar recall pode aumentar os falsos positivos.

Os painéis são gerados após salvar as métricas dos dois núcleos. A revisão
manual exporta decisões separadas: um falso alerta deve ser desmarcado para
restaurar o VTEC; uma medição considerada inválida pode ser excluída com motivo.
Essa revisão não altera as métricas automáticas nem retreina modelos.

Detalhes dos protocolos e configurações: [Apêndice A](docs/APENDICE_A_CODIGOS_E_ARQUIVOS.md).

## Núcleo 1 — XGBoost

As entradas são elevação, seno/cosseno do azimute, seno/cosseno da hora e do
mês, média e desvio-padrão de S4 em 3 horas, lag de S4 de 24 horas e Kp, Dst,
F10.7, Ap e SSN.

Parâmetros do relatório:

| Parâmetro | Valor |
|---|---:|
| `n_estimators` | 150 |
| `max_depth` | 6 |
| `learning_rate` | 0,05 |
| `scale_pos_weight` | 1 |
| `subsample` | 0,8 |
| `colsample_bytree` | 0,8 |
| folds temporais | 5 |
| gap entre treino e validação | 24 h |
| limiar de S4 severo | 0,5 |
| limiar probabilístico de filtragem | 0,75 |

O SMOTE e a imputação são ajustados somente no treino. Para evitar que o
classificador filtre o próprio conjunto de ajuste com previsões dentro da
amostra, o período de desenvolvimento usa probabilidades out-of-fold. O modelo
final, treinado até 26/03, gera as probabilidades do gap e do teste.

## Núcleo 2 — LSTM

| Parâmetro | Valor |
|---|---:|
| divisão treino/validação | 80% / 20%, cronológica |
| normalização | Min-Max `[0, 1]` |
| unidades LSTM | 64 |
| dropout | 0,2 |
| otimizador | Adam |
| perda | MSE |
| épocas máximas | 50 |
| batch size máximo | 400 |
| early stopping | patience 5 |
| look-back convencional | 48 épocas |
| look-back piecewise | mesmo horário nos 15 dias anteriores |
| redes piecewise | 48 por configuração |

São avaliadas oito configurações por série, combinando:

1. convencional ou piecewise;
2. LSTM padrão ou LSTM-Attention;
3. entrada univariada ou multivariada.

Cada configuração é executada para a série original e para a série tratada. Em
conformidade com o relatório, o ramo original usa `VTEC_original` como entrada e
alvo; o ramo tratado usa `VTEC_filtered` como entrada e alvo. As métricas são
MAE, MSE, RMSE e R².

## Dados

Os ISMR são procurados em `../Dados/Dados ismr 2024`. Cada arquivo deve conter
tempo UTC, SVID, elevação, azimute, S4 e TEC/STEC ou VTEC. O processamento reduz
os dados a `datetime`, `svid`, `elev`, `azim`, `S4` e `VTEC`.

O clima espacial deve cobrir todo o período e conter Kp, Dst, F10.7, Ap e SSN.
Por padrão, o pipeline usa, nesta ordem:

1. `dados_externos/omniweb.csv`;
2. a união de `dados_externos/omniweb_*.csv`;
3. `Geomagnéticos/dados_solar_geomag_2024.txt`.

## Preparação

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Para conferir as 52 consultas semanais do experimento:

```powershell
.venv\Scripts\python scripts/00_baixar_ismr_portal.py
```

Para efetuar os downloads com autenticação manual no portal:

```powershell
.venv\Scripts\python scripts/00_baixar_ismr_portal.py --executar
```

Downloads já cobertos por arquivos locais são ignorados.

## Execução oficial

O comando abaixo valida os insumos e executa as quatro estações:

```powershell
.venv\Scripts\python scripts/12_multistation_pipeline.py
```

O fluxo executado é:

1. validação de cobertura, dependências e espaço;
2. integração do clima espacial;
3. localização e extração dos ISMR;
4. controle de qualidade e EDA;
5. engenharia de atributos e alvo futuro de S4;
6. XGBoost, validação temporal e probabilidades out-of-fold;
7. filtragem e interpolação linear do VTEC;
8. matriz LSTM/LSTM-Attention;
9. métricas, gráficos e tabelas para publicação.

Resultados por estação ficam em `resultados/<ESTAÇÃO>/`. A consolidação fica em
`resultados/artigo/`, e os logs em `resultados/logs/`.

O relatório inclui um [painel de Visual Analytics](docs/VISUAL_ANALYTICS.md)
offline, com histórico e dispersão de VTEC, épocas do Núcleo 1 circuladas e
revisão manual por época/SVID. Para refazer os gráficos da FRTZ usando os
modelos já treinados:

```powershell
.venv/Scripts/python scripts/11_generate_report.py --station FRTZ
```

Abra `resultados/FRTZ/relatorio/visual_analytics/painel_visual_analytics.html`.

## Alterações posteriores

As variáveis de ambiente `GNSS_STUDY_START`, `GNSS_TRAIN_END`,
`GNSS_TEST_START`, `GNSS_TEST_END`, `GNSS_STATIONS`,
`GNSS_LSTM_ORGANIZATIONS`, `GNSS_LSTM_ATTENTION` e
`GNSS_LSTM_MULTIVARIATE` permitem criar experimentos posteriores. Qualquer uso
dessas opções deve ser identificado como experimento derivado, não como a
reprodução do relatório.

## Verificação de sintaxe

```powershell
.venv\Scripts\python -c "import ast; from pathlib import Path; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in Path('scripts').rglob('*.py')]"
```

Para os protocolos, limitações e localização dos arquivos de cada experimento,
consulte o [Apêndice A](docs/APENDICE_A_CODIGOS_E_ARQUIVOS.md) e a
[documentação metodológica](docs/COERENCIA_METODOLOGICA.md).
