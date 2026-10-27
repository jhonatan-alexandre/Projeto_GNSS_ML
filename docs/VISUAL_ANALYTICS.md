# Visual Analytics de S4 e VTEC

O relatório da estação agora inclui a dispersão das observações por SVID,
as séries comparativas de S4/VTEC, a cintilação severa (CS, S4 ≥ 0,5)
e as oito configurações M1–M8, em ambos os ramos.
As figuras são exportadas em PNG (300 dpi), SVG e PDF.

## Gerar novamente, usando os modelos já treinados

```powershell
.venv/Scripts/python scripts/11_generate_report.py --station FRTZ
```

Para gerar somente a parte de análise visual:

```powershell
.venv/Scripts/python scripts/visual_analytics.py --station FRTZ
```

Abra `resultados/FRTZ/relatorio/visual_analytics/painel_visual_analytics.html`
no navegador. O painel funciona offline, sem servidor ou bibliotecas externas.

No experimento Xie, informe o manifesto e a raiz da estação:

```powershell
.venv/Scripts/python -X utf8 scripts/11_generate_report.py --config configs/xie_pru2.json --output-root experimentos/xie_20260920/PRU2
```

O painel fica em
`experimentos/xie_20260920/PRU2/resultados/PRU2/relatorio/visual_analytics/`.
Só é gerado depois das métricas de ambos os núcleos. Os pontos observados são
médias de 30 minutos por SVID; CS é classificada depois dessa média. A mediana
observada não contém interpolação. As linhas das séries de treinamento podem
conter interpolação e são identificadas como tal.

## Exploração e revisão

1. Selecione o período e, opcionalmente, um SVID. Arrastar sobre os gráficos
   temporais também altera o recorte. As linhas de mediana sempre representam
   a estação; o seletor de SVID afeta a dispersão e a tabela.
2. Inspecione os círculos e o valor de S4 e de probabilidade no mesmo instante.
   O Núcleo 1 sinaliza `P(S4(t+30 min) >= 0,5)` pelo limiar salvo da execução
   (0,75 no experimento anterior; selecionado por estação no novo). Os círculos ficam em
   `target_datetime`, associados ao VTEC do mesmo SVID. Probabilidades ausentes
   permanecem ausentes, especialmente no início do histórico sem OOF.
3. Na revisão, escolha **probabilidade**, **S4 observado** ou **VTEC** e informe
   o limiar. O corte é estrito (`valor < limiar`). A tabela pode restringir a
   seleção às observações sinalizadas pelo Núcleo 1.
4. Escolha **Falso positivo · restaurar VTEC** para retirar um alerta e
   reincluir a medição na série revisada. Use **Excluir medição da série
   derivada** somente para uma medição considerada inválida. Cada decisão
   registra época UTC, SVID, ação, critério, limiar e motivo. Restaurações
   aparecem como losangos e exclusões com ×; a série revisada aparece em roxo.
   As decisões podem ser desfeitas individualmente ou em conjunto.
5. Exporte as decisões em JSON para guardar/retomar a revisão, ou exporte a
   prévia da série inteira em CSV. O recorte da tela não limita a exportação.

O critério inicial é S4 observado abaixo de 0,5. A ação de falso positivo
exige um alerta originalmente salvo. O limiar da tabela filtra candidatos à
inspeção; não modifica o classificador nem a sinalização original auditada.

## Aplicar as decisões e gerar as figuras da revisão

```powershell
.venv/Scripts/python scripts/visual_analytics.py --station FRTZ --revisao caminho/FRTZ_decisoes_revisao.json
```

Para a nova PRU2:

```powershell
.venv/Scripts/python -X utf8 scripts/visual_analytics.py --config configs/xie_pru2.json --output-root experimentos/xie_20260920/PRU2 --revisao caminho/PRU2_decisoes_revisao.json
```

Os resultados ficam em `resultados/<ESTACAO>/relatorio/revisao_manual/`, dentro
da raiz de saída utilizada. No exemplo novo da PRU2, o caminho começa com
`experimentos/xie_20260920/PRU2/`. Incluem decisões, observações removidas,
série revisada e figuras. O arquivo JSON
inclui uma assinatura do conjunto de dados para impedir sua aplicação em
outro experimento. Cada nova execução com `--revisao` atualiza essa pasta;
guarde o JSON de cada revisão que desejar reproduzir.

A revisão atua sobre a chave época/SVID. O registro automático é preservado;
na série derivada, um falso positivo desmarcado restaura a medição, e uma
exclusão a remove. A mediana é recalculada por época e interpolada separadamente
antes e durante o teste. As fontes, métricas e dataset de treinamento permanecem
preservados, com hashes registrados no manifesto. A auditoria inclui
`auditoria_revisao.csv` e `falsos_positivos_restaurados.csv`.

Novos JSONs usam `schema_version: 2`, lista `decisoes`, ações `falso_positivo`
ou `excluir_observacao` e `metricas_sha256`. JSONs legados de versão 1 com
`exclusoes` continuam aceitos, sempre com a semântica antiga de excluir medição.

A revisão é uma análise derivada. O uso do S4 observado é retrospectivo.
As curvas previstas e métricas exibidas pertencem aos modelos já treinados,
com seus alvos originais (`VTEC_original` e `VTEC_filtered`). Nenhuma nova
métrica LSTM é atribuída à série revisada sem novo treinamento.

## Figuras

| Arquivo (PNG, SVG e PDF) | Conteúdo |
|---|---|
| `vtec_historico_deteccoes` | Histórico completo, detalhe dos últimos três dias e diferença entre medianas |
| `nucleo1_epocas_circuladas` | S4, probabilidade e VTEC alinhados no instante-alvo |
| `vtec_dispersao_original_filtrado` | Dispersão original × filtrado e diferença × média |
| `regressao_series_todos_original` | Séries das configurações do ramo original |
| `regressao_series_todos_filtrado` | Séries do ramo filtrado sobrepostas aos originais em dispersão |
| `regressao_dispersao_todos_original` | Observado × previsto, reta 1:1, RMSE e R² por configuração |
| `regressao_dispersao_todos_filtrado` | Mesmo diagnóstico contra o alvo filtrado |
| `vtec_real_serie_temporal` | Histórico e dia de teste com observações reais em dispersão; lacunas sem pontos |
| `vtec_real_previsoes_teste` | VTEC real por SVID e mediana observada em pontos, previsões M1–M8 em linhas |
| `s4_vtec_cs_30min_historico` | S4, VTEC e proporção de CS observada no histórico |
| `s4_vtec_cs_30min_teste` | Mesma comparação no teste final |
| `s4_cintilacao_real_alertas_30min` | S4 em dispersão, CS por SVID e alertas circulados, seguindo o Estudo Dirigido |

O relatório também exporta distribuição de classes antes/depois do balanceamento
e curvas de precisão/recall/F2 usadas na seleção. Se forem escolhidos pesos de
classe, as contagens ficam intactas e não se apresentam amostras como sintéticas.

Nas séries e dispersões dos modelos, os círculos identificam épocas em que
pelo menos um SVID foi sinalizado. Na dispersão bruta, cada círculo identifica
o próprio SVID. Os limites dos dados disponíveis são respeitados: no teste
atual da FRTZ, o histórico termina em 28/03/2024.

Os gráficos das estações recuperam `VTEC_observado` diretamente das
observações válidas, antes da interpolação, para os pontos
reais. Essa série também é exportada em `vtec_observado_30min.csv`.
