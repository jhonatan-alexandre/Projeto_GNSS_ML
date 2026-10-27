"""Dossiê autossuficiente em Markdown e DOCX, usando resultados auditados."""
import json
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from PIL import Image
from core_convlstm import ROOT, OUT, sha

DEST=OUT/'artigo/documentacao/apresentacao'
ART=OUT/'artigo'


def table(df):
    def fmt(v):
        if pd.isna(v):return 'não informado'
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])+'\n'


def docx(text,images,path):
    """Minimal OOXML document with genuine tables and embedded reference figures."""
    w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    body=[]
    def para(t,style=None):
        t=re.sub(r'\*\*(.*?)\*\*',r'\1',t)
        t=re.sub(r'`([^`]+)`',r'\1',t)
        prop=f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
        return f'<w:p>{prop}<w:r><w:t xml:space="preserve">{escape(t)}</w:t></w:r></w:p>'
    lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('| '):
            rows=[]
            while i<len(lines) and lines[i].startswith('| '):
                cells=[s.strip() for s in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',s) for s in cells):
                    rows.append('<w:tr>'+''.join('<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/></w:tcPr>'+para(c)+'</w:tc>' for c in cells)+'</w:tr>')
                i+=1
            body.append('<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders><w:bottom w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="2"/></w:tblBorders></w:tblPr>'+''.join(rows)+'</w:tbl>');continue
        level=len(line)-len(line.lstrip('#'))
        if level:body.append(para(line[level:].strip(),'Title' if level==1 else 'Heading'+str(min(level-1,3))))
        elif line.strip():body.append(para(line))
        i+=1
    relationships=['<Relationship Id="rStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>']
    for i,(label,img) in enumerate(images,1):
        body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>');body.append(para(label,'Heading1'))
        with Image.open(img) as im:width,height=im.size
        cx=5943600;cy=int(cx*height/width)
        body.append(f'<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{i}" name="Figura {i}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{i}" name="Figura {i}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rImg{i}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
        body.append(para('Arquivo: '+str(img.relative_to(ROOT))))
        relationships.append(f'<Relationship Id="rImg{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/figura{i}.png"/>')
    section='<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1000" w:right="1000" w:bottom="1000" w:left="1000"/></w:sectPr>'
    namespaces=f'xmlns:w="{w}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
    styles=f'<w:styles xmlns:w="{w}"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="21"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120"/></w:pPr></w:pPrDefault></w:docDefaults>'
    for name,size in [('Title',34),('Heading1',28),('Heading2',24),('Heading3',22)]:
        styles+=f'<w:style w:type="paragraph" w:styleId="{name}"><w:name w:val="{name}"/><w:pPr><w:keepNext/></w:pPr><w:rPr><w:b/><w:sz w:val="{size}"/></w:rPr></w:style>'
    styles+='</w:styles>'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rDoc" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr('word/document.xml',f'<?xml version="1.0" encoding="UTF-8"?><w:document {namespaces}><w:body>'+''.join(body)+section+'</w:body></w:document>')
        z.writestr('word/styles.xml',styles)
        z.writestr('word/_rels/document.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(relationships)+'</Relationships>')
        for i,(_,img) in enumerate(images,1):z.write(img,f'word/media/figura{i}.png')


def main():
    DEST.mkdir(parents=True,exist_ok=True);sources={}
    def read(name):
        p=ART/name;sources[str(p.relative_to(ROOT))]=sha(p)
        return pd.read_csv(p)
    n1=read('tabelas/nucleo1_18fev.csv');best=read('tabelas/menor_rmse_retrospectivo.csv')
    effect=read('tabelas/efeito_filtragem_rmse.csv');hp=read('tabelas/hiperparametros_nucleo1.csv')
    history=read('documentacao/LINHA_DO_TEMPO_CONSOLIDADA.csv').fillna('')
    timing=read('figuras/tempos_treinamento/tempos_finais_por_modelo_estacao.csv')
    audit=json.loads((ART/'auditoria.json').read_text(encoding='utf-8'));assert audit['status']=='completo' and audit['final_configurations']==96
    text='''# Dossiê de atualização do estudo dirigido e briefing para uma apresentação

Autor do estudo: Jhonatan. Consolidação documental: 23/09/2026. Datas dos dados científicos: 2023–2024. A execução computacional em 2026 não significa que as observações GNSS sejam de 2026.

## 1. Como usar este documento

Envie o DOCX ou o Markdown a um chatbot. O documento contém contexto, alterações, resultados atuais, limitações, inventário de figuras e um pedido pronto para elaborar um plano de apresentação. O DOCX inclui oito exemplos visuais ao final, dois por estação. O Markdown contém o mesmo texto, sem incorporar as imagens. Os caminhos citados são referências locais: um chatbot externo não consegue abri-los sem receber os arquivos.

Escopo: evolução documentada desde o estudo dirigido, antecedentes computacionais registrados e alterações da rodada atual, incluindo gráficos e auditoria. Não se presume um registro de toda alteração linha a linha do código. O histórico resumido no apêndice conserva a distinção entre experimentos anteriores e o estudo vigente.

## 2. Resumo executivo e estado atual

O trabalho integra dois núcleos: o primeiro classifica a ocorrência futura de cintilação severa pelo índice S4; o segundo prevê VTEC e compara o uso de séries originais e filtradas pelo primeiro núcleo. A pergunta científica é se a detecção de épocas potencialmente degradadas contribui para a previsão de VTEC, e como o resultado varia por estação, arquitetura e organização temporal.

O conjunto vigente foi concluído e auditado: quatro estações (FRTZ, PALM, POAL e PRU2), 12 modelos, dois ramos por modelo, 96 configurações finais, 2.352 redes individuais e 4.608 previsões de regressão. Cada configuração possui 48 épocas no dia 18/02/2024. Há quatro classificadores de S4. A auditoria final é a referência de completude; estados antigos marcados como “executando” são registros desatualizados.

Melhorias comprovadas: calendário comum entre estações, ampliação da comparação, documentação da seleção e dos limiares, verificação dos artefatos, comparação em referência observada comum e figuras com dispersão real. Isso não equivale a uma melhora universal de acurácia. A filtragem reduziu o RMSE em 13 de 48 pares modelo–estação; nos demais 35 o RMSE aumentou. O menor RMSE retrospectivo do dia foi obtido por M10 original nas quatro estações. Esses resultados não demonstram superioridade universal nem desempenho operacional futuro.

## 3. O que mudou em relação ao relatório e às execuções anteriores

| Aspecto | Relatório/antecedentes | Rodada vigente e contribuição |
| --- | --- | --- |
| Calendário | O relatório apresenta resultados de 28/03/2024; execuções derivadas também usaram calendários distintos por disponibilidade | Histórico comum desde 01/12/2023, ajuste final até 16/02/2024 e teste comum em 18/02/2024 |
| Matriz de modelos | Oito configurações M1–M8 de LSTM e LSTM com atenção | Inclusão de M9–M12 ConvLSTM: 12 modelos e 24 configurações por estação |
| Limiar de classificação | Texto do relatório descreve probabilidade 0,75 e seleção por F1 | Limiar por estação, selecionado temporalmente com F2; não confundir com o limiar físico S4 ≥ 0,5 |
| Desbalanceamento | Relatório descreve SMOTE e scale_pos_weight = 1 | FRTZ/POAL/PRU2 usam SMOTE; PALM usa pesos de classe, conforme seleção registrada |
| Referência de regressão | Figuras antigas rotulam a curva preta como média de 30 min; antecedentes têm alvos próprios por ramo | Métricas atuais usam mediana observada comum entre satélites; rótulos foram corrigidos para essa referência |
| Ajuste final | Artefatos históricos incluem seleção e execuções preliminares | Separação entre busca/validação e redes finais reajustadas até 16/02, com épocas congeladas |
| Comparabilidade | Diferentes períodos, famílias e versões poderiam ser misturados | Artigo atual separa fevereiro dos antecedentes e declara diferenças de alvo de treino e horizonte |
| POAL | Busca M11/M12 interrompida por ausência de alvos de validação | Janela anterior autorizada somente nos horários afetados; teste final preservado |
| Execução | Interrupções e estado persistido “executando” sem processos ativos | Retomada com checagem de hashes, reutilização compatível, travas e auditoria final |
| Visualização | Figuras do relatório servem como referência de linguagem visual | Dispersão observada ao fundo, alertas de S4 circulados, VTEC com M1–M12, hiperparâmetros e tempos |
| Interpretação | Algumas formulações antigas sugerem estabilização e redução geral do erro pela filtragem | Benefício tratado como hipótese avaliada: ganhos e perdas apresentados por modelo e estação |

Não comparar numericamente o RMSE do relatório antigo com o atual como se fosse um experimento controlado: mudaram datas, disponibilidade, modelos e aspectos do protocolo. Não transportar conclusões de março para fevereiro.

## 4. Protocolo que deve aparecer na apresentação

Núcleo 1: prever a classe S4(t + 30 min) ≥ 0,5 com XGBoost; alinhar o alerta à época-alvo e ao SVID; avaliar precisão, recall, F1, F2, TP, FP e FN sem reclassificação manual posterior. O desenvolvimento utiliza previsões temporais fora da amostra (OOF); os períodos sem passado suficiente não possuem previsão e não podem ser contados como acertos ou erros.

Integração: alertas identificam observações potencialmente degradadas e produzem um ramo filtrado para comparação com o original. Não são prova de outliers estatísticos nem de erro de coordenadas. O estudo prevê S4/VTEC; não executa SPP/PPP e não quantifica diretamente erro de posicionamento.

Núcleo 2: modelos convencionais recebem as 48 épocas anteriores de 30 min; modelos piecewise recebem o mesmo horário nos 15 dias anteriores. Os horizontes informacionais são diferentes. As previsões convencionais são sequenciais, não uma previsão integral do dia emitida à meia-noite. O dia 17/02 não ajusta pesos, mas pode fornecer contexto de entrada. O teste é 18/02, com 48 épocas observadas.

M1–M8 preservam a seleção de taxa e épocas do fluxo ativo; redes e normalizadores finais são reajustados em todo o desenvolvimento elegível. M9–M12 comparam dois candidatos: 16 filtros/taxa 0,001 e 32 filtros/taxa 0,0003. Early stopping usual: 03–09/02; seleção: 10–16/02. A escolha considera os dois ramos conjuntamente; o teste não participa. M1–M8 mantêm alvos de treinamento próprios do ramo; ConvLSTM usa alvo observado comum. Portanto, comparar famílias não isola apenas a arquitetura.

Exceção POAL: M11/M12, slots 14–41 (07:00–20:30 UTC), não possuíam alvos observados para seleção em 10–16/02. Após autorização, a busca nesses slots ajusta até 26/01, usa 27/01–02/02 para early stopping e 03–09/02 para seleção. Demais slots mantêm a partição usual. Ajuste final até 16/02 e teste de 18/02 permanecem. Os suportes de validação podem diferir por horário. A adaptação não preencheu alvos de validação com dados sintéticos.

A interpolação offline de entradas foi herdada da preparação; isso limita alegações de operação em tempo real. O alvo comum é uma referência observacional agregada, não uma verdade independente.

## 5. Mapa dos modelos e hiperparâmetros

| Modelos | Família | Organização | Entradas |
| --- | --- | --- | --- |
| M1 / M2 | LSTM | Convencional | Uni / multivariada |
| M3 / M4 | LSTM com atenção | Convencional | Uni / multivariada |
| M5 / M6 | LSTM | Piecewise por horário | Uni / multivariada |
| M7 / M8 | LSTM com atenção | Piecewise por horário | Uni / multivariada |
| M9 / M10 | ConvLSTM | Convencional | Uni / multivariada |
| M11 / M12 | ConvLSTM | Piecewise por horário | Uni / multivariada |

Cada modelo possui ramo original e filtrado. Convencional: uma rede por ramo; piecewise: 48 redes por ramo. Por estação são 12 redes convencionais e 576 piecewise, totalizando 588. Nas quatro estações: 2.352.

Taxas selecionadas do fluxo LSTM: FRTZ 0,05; PALM 0,0005; POAL e PRU2 0,005. A apresentação deve remeter às figuras dos hiperparâmetros para as épocas por modelo/ramo e candidatos ConvLSTM, evitando sugerir uma única quantidade de épocas para todas as redes.

Hiperparâmetros XGBoost registrados (campos ausentes no resumo não devem ser inventados):

'''
    text+=table(hp)
    text+='''
## 6. Resultados atuais do Núcleo 1

Os números abaixo são do teste de 18/02/2024, por observação satélite–época, não por número de dias ou redes. As métricas estão em proporções entre zero e um.

'''+table(n1[['station','n','positivos','TP','FP','FN','precision','recall','f1','f2']])
    text+='''
Leitura: PALM e POAL apresentam recall acima de 0,92, acompanhado de precisão de aproximadamente 0,223 e 0,240. Em PALM, são 41 verdadeiros positivos e 143 falsos positivos. FRTZ detecta 32 de 40 positivos; PRU2 detecta 65 de 88. F2 prioriza recall, mas não transforma falsos positivos em benefício comprovado. Explicar o custo de filtrar observações sem cintilação severa e avaliar seu efeito na regressão. Não usar apenas acurácia para sustentar qualidade em uma classe rara.

## 7. Resultados atuais do Núcleo 2 e efeito da filtragem

Menor RMSE retrospectivo por estação entre as 24 configurações avaliadas (TECU para RMSE/MAE):

'''+table(best[['station','model','variant','RMSE','MAE','R2']])
    summary=effect.groupby('station').apply(lambda d:pd.Series({'modelos_com_reducao_RMSE':int((d.ganho_percentual>0).sum()),'modelos_com_aumento_RMSE':int((d.ganho_percentual<0).sum()),'mediana_ganho_percentual':d.ganho_percentual.median()}),include_groups=False).reset_index()
    text+='\nEfeito pareado dentro de cada modelo; ganho positivo significa redução do RMSE com filtragem:\n\n'+table(summary)
    text+='''
Exemplos para discussão: FRTZ/M1 reduz RMSE de 29,8759 para 21,5350 TECU (27,92%), mas isso não torna o modelo o melhor da estação. FRTZ/M2 aumenta de 4,7131 para 22,1205 TECU: degradação importante que deve aparecer junto dos ganhos. M10 original é o mínimo retrospectivo nas quatro estações; não usar esse ranking do teste para alegar uma seleção independente de implantação. A mediana dos ganhos entre modelos é negativa nas quatro estações.

A identificação de curvas quase constantes ou R² negativo mostra baixo desempenho, não prova de interrupção de treinamento. A auditoria confirma artefatos finais, mas não explica a causa de uma previsão ruim. Não atribuir automaticamente o comportamento a taxa alta, atenção, perda de variabilidade ou falha do otimizador sem análise adicional.

## 8. Tempos de treinamento: o que pode ser afirmado

As figuras comparam M1–M12 por estação e distinguem desenvolvimento/seleção de reajuste final. Também apresentam segundos médios por rede para contextualizar as duas redes convencionais versus 96 redes piecewise ao somar os ramos de um modelo. Somatórios finais registrados em horas:

'''
    totals=timing.groupby('station').seconds.sum().div(3600).reset_index(name='horas_registradas_reajuste_final');text+=table(totals)
    text+='''
Os cronômetros incluem preparação, ajuste, previsão, salvamento e verificações. Execuções concorrentes disputaram recursos; a soma não é a duração global da sessão nem um benchmark isolado de arquitetura. FRTZ/M9 tem reinício identificado: o tempo salvo não cobre necessariamente a tentativa anterior. A busca compartilhada da taxa M1–M8 e partes reutilizadas da tentativa interrompida de POAL não possuem custo completo recuperável. Ausência de medição não significa custo zero. Não apresentar esses somatórios como todo o tempo histórico investido.

## 9. Melhorias de apresentação e inventário de figuras

Todas as figuras são estáticas e exportadas em PNG, PDF e SVG, com CSVs e referências de origem. O estilo final foi adaptado das figuras 6.1–6.2 e 6.9–6.10 do PDF do estudo dirigido fornecido pelo autor.

- Núcleo 1: S4 real em cinza ao fundo; eventos reamostrados com S4 ≥ 0,5 coloridos pelo SVID; círculos vermelhos nos alertas alinhados à época-alvo. A altura do círculo é S4 observado: o classificador não prevê uma regressão contínua de S4. Círculos abaixo de 0,5 são falsos positivos; eventos sem círculo só são falsos negativos quando há probabilidade disponível.
- Núcleo 1 histórico: 01/12/2023–18/02/2024, fundo com observações reamostradas a cada 30 min. No detalhe do dia de teste, o fundo usa todos os pontos nativos. Não confundir essas resoluções.
- Núcleo 2: dados nativos de todos os satélites em dispersão cinza; mediana observada comum em preto; M1–M12 em cores. Versões separadas original/filtrado incluem RMSE; a versão conjunta possui 24 curvas. O fundo é mantido igual nos dois ramos. Nos painéis individuais, tracejado distingue piecewise; no gráfico conjunto, tracejado distingue filtrado.
- Hiperparâmetros: curvas de seleção do limiar, parâmetros XGBoost, seleção da taxa das redes e quadro por estação/modelo.
- Tempos: mapa modelo × estação e separação das etapas de desenvolvimento e ajuste final.
- Material complementar: matrizes de confusão, comparações temporais em painéis, observado versus predito, tabelas de RMSE/MAE/R² e efeito pareado da filtragem.

Pastas relativas à raiz do projeto:

| Material | Caminho |
| --- | --- |
| Estilo final preferencial: 20 figuras, cinco por estação | estudo_20240218/artigo/figuras/estilo_estudo_dirigido/README.md |
| Cada estação: S4 dia e histórico; VTEC original, filtrado e todos | estudo_20240218/artigo/figuras/estilo_estudo_dirigido/{ESTACAO}/ |
| Versão complementar de séries e cobertura: 16 figuras | estudo_20240218/artigo/figuras/series_nucleos/README.md |
| Hiperparâmetros: duas figuras principais | estudo_20240218/artigo/figuras/hiperparametros/README.md |
| Tempo de treinamento: duas figuras principais | estudo_20240218/artigo/figuras/tempos_treinamento/README.md |
| Tabelas atuais | estudo_20240218/artigo/tabelas/ |
| Auditoria final do experimento | estudo_20240218/artigo/auditoria.json |

Os 20 gráficos no estilo final e os 16 gráficos complementares são versões de visualização dos mesmos resultados, não experimentos adicionais. Para slides, priorizar a versão final; não exibir todos os gráficos de uma vez. A sobreposição de 24 curvas funciona melhor como panorama, com versões separadas por ramo para discussão.

## 10. Limitações, pendências e perguntas para discussão

Concluído: treinamento, auditoria, tabelas, seleção/limiares documentados, figuras temporais de S4/VTEC, tempos e adaptação visual ao relatório.

Pendente: explicabilidade global/local do XGBoost; análise de falsos alertas e eventos perdidos; diagnóstico causal dos modelos com baixo desempenho; inspeção de perdas, escalas, amplitude e sensibilidade por oclusão/perturbações. Existe uma triagem preliminar de 80 configurações que encontrou 17 casos com R² negativo e desvio-padrão previsto abaixo de 10% do observado. Ela não foi atualizada para os 96 resultados e não deve ser apresentada como diagnóstico final.

Limitações científicas a explicitar: um único dia comum de teste nesta comparação; suporte desigual por estação e horário; validação adaptada de POAL; interpolação offline; horizontes convencionais/piecewise distintos; diferenças de alvo de treino entre famílias; classificação com baixa prevalência e falsos positivos; referência agregada em vez de verdade independente; ausência de validação direta de posicionamento.

Limitação de software: o registro da retomada informa 11 testes específicos de fevereiro aprovados. A suíte geral executou 95 testes, com cinco falhas e quatro erros ligados a módulos históricos ausentes em analises/. Não afirmar que toda a suíte passou. A auditoria de artefatos e as verificações dos gráficos atuais são evidências diferentes desse estado da suíte geral.

Perguntas úteis: por que a filtragem ajuda algumas combinações e piora outras? A disponibilidade e a interpolação alteram a informação útil? Como separar efeito da arquitetura do alvo e do horizonte? As curvas quase constantes indicam subajuste ou outro mecanismo? Os achados persistem em múltiplos dias e condições ionosféricas? Qual seria o custo de falsos positivos em operação real? São perguntas abertas, não conclusões.

## 11. Mensagens que a apresentação pode e não pode sustentar

Pode sustentar: “Concluímos uma comparação auditada com calendário comum, 12 modelos e dois ramos”; “A filtragem tem efeito heterogêneo e melhorou 13 de 48 pares”; “M10 original apresentou o menor RMSE observado neste dia nas quatro estações”; “O desenho visual evidencia dados reais, alertas, previsões e limitações”.

Evitar: “A filtragem sempre melhora”; “ConvLSTM é universalmente superior”; “Todas as falhas foram explicadas”; “Todos os modelos têm o mesmo horizonte”; “O sistema já funciona em tempo real”; “S4 alto comprova erro de coordenadas”; “O estudo foi feito com observações de 2026”; “O limiar de probabilidade é 0,5 ou 0,75 em todas as estações”.

## 12. Pedido pronto para enviar ao chatbot

Use este dossiê como base factual para elaborar um PLANO de apresentação acadêmica do estudo dirigido de Jhonatan, em português. Ainda não escreva os slides completos. Considere público de pós-graduação com conhecimentos gerais de aprendizado de máquina, mas sem assumir domínio de GNSS. Se duração e público específicos não forem informados, proponha uma versão de 15–20 minutos, com 12–15 slides principais e apêndices; identifique essa duração como sugestão, não como requisito do autor.

Organize uma narrativa que conecte problema, hipótese, dois núcleos, evolução metodológica, protocolo vigente, resultados, limitações e próximos passos. Distinga claramente resultados históricos de março e resultados atuais de fevereiro. Diferencie melhorias de rigor/reprodutibilidade de melhora numérica de desempenho. Não invente explicabilidade, causalidade, testes adicionais, significância estatística, resultados operacionais ou ganhos universais.

Entregue: (1) objetivo e mensagem central; (2) sequência de slides com título, pergunta respondida, mensagem principal, evidência numérica, figura/tabela indicada por nome, tópicos de fala e duração; (3) transições entre seções; (4) itens que devem ficar no apêndice; (5) perguntas prováveis da banca e respostas ancoradas no dossiê; (6) lista de informações ainda necessárias para finalizar a apresentação. Para os gráficos, indique quais usar, quais ampliar e quais simplificar sem apagar resultados ruins. Inclua o balanço dos falsos positivos e o efeito misto da filtragem. Separe explicitamente o que está comprovado, o que é hipótese e o que é trabalho futuro.

Os caminhos locais não estarão acessíveis por padrão. Use as imagens incorporadas ao DOCX como exemplos e solicite apenas os arquivos adicionais indispensáveis. Não declare ter visto imagens que não recebeu. Preserve os nomes M1–M12, estações, datas, unidades e definições de mediana/limiar descritos aqui.

## 13. Apêndice: histórico documentado de alterações

O inventário abaixo resume 43 registros existentes. Alguns são antecedentes concluídos, outros são propostas ou estados intermediários. A coluna “situação” e os limites originais impedem tratá-los como mudanças automaticamente ativas em fevereiro. Datas não comprovadas permanecem não inferidas. Em particular, o registro histórico N15 possui inconsistências de contagem: a ampliação M10–M12 adiciona seis configurações, 288 previsões e 194 redes por estação; M9–M12 totalizam oito configurações, 384 previsões e 196 redes por estação. Os números atuais auditados prevalecem.

'''
    for r in history.to_dict('records'):
        text+=f"### {r['id']} — {r['etapa']}\n\nSituação histórica: {r['situacao']}. Referência temporal: {r['data_referencia']}.\n\nMudança: {r['mudanca']}\n\nMotivação: {r['motivacao']}\n\nResultado registrado: {r['resultado']}\n\nLimites: {r['limites']}\n\nFontes: {r['fontes']}\n\n"
    text+='''### Complementos finais de 23/09/2026

1. Retomada das execuções interrompidas com reutilização de artefatos compatíveis; preservação das fontes e hashes.
2. Autorização e implementação isolada da janela anterior para POAL M11/M12 nos slots afetados.
3. Conclusão de 96 configurações, 2.352 redes e 4.608 previsões, registrada na auditoria final.
4. Figuras de seleção do limiar e hiperparâmetros das redes, acompanhadas de dados e fontes.
5. Figuras de tempos registrados, separando fases e identificando lacunas históricas de medição.
6. Séries históricas de S4, detalhe do teste e sobreposição de todos os modelos de VTEC.
7. Adaptação ao estilo visual do PDF do estudo dirigido, com observações reais ao fundo e RMSE nas legendas.
8. Preparação deste dossiê para orientar o planejamento da apresentação. A explicabilidade permanece pendente.

## 14. Fontes principais e prioridade em caso de conflito

Prioridade para resultados atuais: auditoria final e CSVs do artigo; protocolo de fevereiro e autorização POAL; auditorias específicas das figuras. O PDF original e as notas antigas descrevem o histórico e a linguagem visual, não substituem os resultados atuais. Registros de progresso não devem ser usados como contagem final.

Fontes textuais: estudo_20240218/README.md; RETOMADA_20260923.md; validacao_POAL_autorizada.json; PROXIMA_ETAPA_EXPLICABILIDADE.md; artigo/documentacao/LINHA_DO_TEMPO_CONSOLIDADA.csv. Fonte anterior: “Estudo dirigido - relatório (Jhonatan).pdf”, em especial seções metodológicas dos dois núcleos e figuras 6.1–6.16. Este dossiê não altera o PDF original.

O DOCX inclui a seguir oito exemplos: S4 do dia de teste e VTEC original de cada estação. As versões filtradas, históricas, de hiperparâmetros e tempos estão nas pastas indicadas. Os exemplos são ilustrações dos resultados, não uma proposta de que todos devem entrar no corpo da apresentação.
'''
    images=[]
    for st in ('FRTZ','PALM','POAL','PRU2'):
        for name,label in [('nucleo1_dia_teste','Núcleo 1: S4 e alertas'),('nucleo2_vtec_original','Núcleo 2: VTEC e modelos — original')]:
            img=ART/'figuras/estilo_estudo_dirigido'/st/(name+'.png');images.append((st+' — '+label,img));sources[str(img.relative_to(ROOT))]=sha(img)
    for p in [OUT/'README.md',OUT/'RETOMADA_20260923.md',OUT/'validacao_POAL_autorizada.json',OUT/'PROXIMA_ETAPA_EXPLICABILIDADE.md',ART/'auditoria.json',Path(__file__).resolve()]:sources[str(p.relative_to(ROOT))]=sha(p)
    pdf=ROOT.parent/'Documentação'/'Estudo dirigido - relatório (Jhonatan).pdf';sources[str(pdf)]=sha(pdf)
    name='DOSSIE_MUDANCAS_E_BRIEFING_APRESENTACAO'
    (DEST/(name+'.md')).write_text(text,encoding='utf-8')
    docx(text,images,DEST/(name+'.docx'))
    (DEST/'fontes_sha256.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'document':str(DEST/(name+'.docx')),'words':len(text.split()),'embedded_figures':len(images),'historical_records':len(history)},ensure_ascii=False))


if __name__=='__main__':main()
