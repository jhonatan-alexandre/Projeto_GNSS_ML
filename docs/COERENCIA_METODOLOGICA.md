# Coerência metodológica da reprodução

Revisão alinhada ao relatório de Estudo Dirigido.

## Contrato experimental

| Item | Implementação oficial |
|---|---|
| Estações | FRTZ, PALM, PRU2 e POAL |
| Desenvolvimento | 01/01/2024 a 26/03/2024 |
| Gap | 27/03/2024 |
| Teste final | 28/03/2024 |
| Resolução | 30 minutos |
| Núcleo 1 | XGBoost para `S4(t+30 min) >= 0,5` |
| Núcleo 2 | LSTM/LSTM-Attention para VTEC |

As datas padrão de `config.py` constituem o experimento do relatório. Datas
fornecidas por variáveis de ambiente caracterizam um experimento derivado.

## Encadeamento dos núcleos

1. Os ISMR são padronizados por estação e SVID.
2. As features de S4 no instante `t` usam somente informações disponíveis até
   `t`.
3. O alvo do XGBoost corresponde a `S4(t+30 min) >= 0,5`.
4. SMOTE e imputação são ajustados somente nos subconjuntos de treinamento.
5. A validação usa cinco divisões temporais com gap de 24 horas.
6. As probabilidades usadas para filtrar o período de desenvolvimento são
   out-of-fold; épocas iniciais sem passado suficiente permanecem sem filtro.
7. O modelo final do XGBoost, ajustado até 26/03, produz os alertas do gap e do
   teste.
8. Alertas com probabilidade `>= 0,75` removem o VTEC do SVID na época-alvo.
9. Os valores ausentes são tratados por interpolação linear e os SVIDs são
   agregados pela mediana da estação.
10. As séries original e tratada alimentam separadamente a matriz LSTM.

## Contrato da regressão

Para reproduzir a comparação apresentada no relatório:

- ramo sem filtragem: entrada e alvo `VTEC_original`;
- ramo com filtragem: entrada e alvo `VTEC_filtered`.

Assim, as métricas quantificam o desempenho de cada conjunto conforme ele entra
na modelagem descrita no trabalho. Uma ablação futura com alvo comum deve ser
registrada como alteração metodológica posterior.

As métricas exportadas são MAE, MSE, RMSE e R². Como os alvos dos ramos são
distintos, a variação entre eles representa o efeito conjunto do tratamento da
série e da modelagem, exatamente como no relatório, e não uma ablação isolada da
entrada.

## Hiperparâmetros

### XGBoost

- 150 estimadores;
- profundidade máxima 6;
- learning rate 0,05;
- subsample 0,8;
- colsample_bytree 0,8;
- scale_pos_weight 1;
- random state 42;
- limiar probabilístico 0,75.

### LSTM

- normalização Min-Max `[0, 1]` ajustada no treino;
- divisão cronológica 80/20;
- 64 unidades por camada;
- dropout 0,2;
- Adam e perda MSE;
- até 50 épocas;
- batch size máximo 400;
- early stopping com patience 5;
- janela contínua de 48 épocas;
- janela piecewise do mesmo horário nos 15 dias anteriores;
- 48 redes por configuração piecewise.

## Matriz experimental

Cada estação executa oito combinações para cada uma das duas séries:

- convencional/piecewise;
- LSTM/LSTM-Attention;
- univariada/multivariada;
- original/filtrada.

Isso resulta em 16 resultados comparáveis por estação. As configurações
piecewise materializam 48 redes cada.

## Decisões que precisam permanecer explícitas no texto do trabalho

- correção térmica do S4 quando a coluna está disponível;
- conversão de TEC/STEC para VTEC com camada fina a 450 km quando não há VTEC;
- limites do controle de qualidade;
- média em 30 minutos por SVID;
- mediana dos SVIDs para formar a série de VTEC da estação;
- interpolação linear offline;
- probabilidades out-of-fold no período de desenvolvimento;
- `subsample=0,8`, `colsample_bytree=0,8` e semente 42;
- uso de apenas um dia no teste final.

O código avalia S4 e VTEC. Qualquer conclusão sobre melhora de coordenadas GNSS
exige uma etapa posterior de posicionamento que não integra esta reprodução.
