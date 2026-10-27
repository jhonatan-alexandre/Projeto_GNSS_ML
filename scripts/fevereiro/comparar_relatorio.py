"""Compara tabelas publicadas no PDF com resultados atuais, sem inferir causalidade."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core_convlstm import ROOT,OUT,sha
from relatorio import export

PDF=ROOT.parent/'Documentação'/'Estudo dirigido - relatório (Jhonatan).pdf'
ART=OUT/'artigo'
DEST=ART/'figuras/comparacao_relatorio_nova_modelagem'
ORDER=[1,5,2,6,3,7,4,8]
PAGES={'FRTZ':83,'PALM':84,'POAL':86,'PRU2':88}
OLD,NEW='#7F8C8D','#0072B2'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})
    reader=PdfReader(PDF);rows=[];sources={str(PDF):sha(PDF)}
    for station,page in PAGES.items():
        t=reader.pages[page-1].extract_text()
        (DEST/f'tabela_relatorio_{station}_pagina_{page}.txt').write_text(t,encoding='utf-8')
        pairs=re.findall(r'(\d+,\d{2})\s+(\d+,\d{2})\s+[+−-]',re.sub(r'\s+',' ',t))
        assert len(pairs)==8,(station,pairs)
        for number,(original,filtered) in zip(ORDER,pairs):
            for variant,value in [('original',original),('filtrado',filtered)]:
                rows.append(dict(station=station,model=f'M{number}',variant=variant,RMSE_relatorio=float(value.replace(',','.')),pagina_pdf=page,tabela=f'6.{list(PAGES).index(station)+3}'))
    old=pd.DataFrame(rows);old.to_csv(DEST/'rmse_transcrito_relatorio.csv',index=False)
    t=reader.pages[79].extract_text();(DEST/'tabela_6_2_classificacao.txt').write_text(t,encoding='utf-8')
    n1old=[]
    for st in PAGES:
        match=re.search(st+r'\s+([0-9.,]+)\s+([0-9.,]+)\s+([0-9.,]+)\s+([0-9.,]+)',t)
        assert match,st
        n1old.append(dict(station=st,**dict(zip(['accuracy','precision','recall','f1'],[float(x.replace(',','.')) for x in match.groups()]))))
    n1old=pd.DataFrame(n1old)
    # Tabela 6.2 e figuras 6.5–6.8 não publicam AUC-ROC.
    n1old['auc_roc']=np.nan
    n1old.to_csv(DEST/'classificacao_transcrita_relatorio.csv',index=False)
    def read(name):
        p=ART/'tabelas'/name;sources[str(p.relative_to(ROOT))]=sha(p);return pd.read_csv(p)
    current=read('nucleo2_18fev_referencia_comum.csv');n1=read('nucleo1_18fev.csv')
    n1['accuracy']=(n1.TP+n1.TN)/n1.n
    auc_records=[]
    for st in PAGES:
        path=ROOT/'experimentos'/f'{st.lower()}_comum_18fev2024'/'resultados'/st/'nucleo1/predicoes_xgboost.csv'
        sources[str(path.relative_to(ROOT))]=sha(path)
        probabilities=pd.read_csv(path,parse_dates=['target_datetime'])
        q=probabilities.loc[probabilities.target_datetime.between('2024-02-18','2024-02-19',inclusive='left')].dropna(subset=['classe_future','probabilidade_severa'])
        assert len(q)==int(n1.loc[n1.station==st,'n'].iloc[0])
        assert q.origem_probabilidade.eq('modelo_final').all()
        assert not q.duplicated(['target_datetime','svid']).any()
        y=q.classe_future.astype(int);p=q.probabilidade_severa
        assert set(y.unique())=={0,1} and p.between(0,1).all()
        assert int(y.sum())==int(n1.loc[n1.station==st,'positivos'].iloc[0])
        auc=float(roc_auc_score(y,p));n1.loc[n1.station==st,'auc_roc']=auc
        fpr,tpr,thresholds=roc_curve(y,p)
        pd.DataFrame({'false_positive_rate':fpr,'true_positive_rate':tpr,'threshold':thresholds}).to_csv(DEST/f'curva_roc_nova_{st}.csv',index=False)
        auc_records.append(dict(station=st,n=len(q),positivos=int(y.sum()),auc_roc=auc,source=str(path.relative_to(ROOT))))
    pd.DataFrame(auc_records).to_csv(DEST/'auc_roc_nova_modelagem.csv',index=False)
    comparison=current[['station','model','variant','RMSE']].rename(columns={'RMSE':'RMSE_novo'}).merge(old,on=['station','model','variant'],how='left',validate='one_to_one')
    comparison['diferenca_descritiva_novo_menos_relatorio']=comparison.RMSE_novo-comparison.RMSE_relatorio
    comparison['status_comparacao']=np.where(comparison.RMSE_relatorio.notna(),'periodos_protocolos_distintos','sem_modelo_anterior')
    comparison.to_csv(DEST/'comparacao_rmse.csv',index=False)
    cls=n1old.merge(n1[['station','accuracy','precision','recall','f1','auc_roc']],on='station',suffixes=('_relatorio','_novo'),validate='one_to_one')
    cls.to_csv(DEST/'comparacao_classificacao.csv',index=False)
    maximum=float(max(comparison.RMSE_novo.max(),comparison.RMSE_relatorio.max()))*1.2
    for st in PAGES:
        fig=plt.figure(figsize=(15,11));gs=fig.add_gridspec(2,2,height_ratios=[1,2.5],hspace=.42,wspace=.22)
        ax=fig.add_subplot(gs[0,:]);row=cls.loc[cls.station==st].iloc[0];x=np.arange(5)
        for offset,suffix,color,label in [(-.18,'relatorio',OLD,'Relatório anterior'),(.18,'novo',NEW,'Nova modelagem — 18/02/2024')]:
            bars=ax.bar(x+offset,[row[m+'_'+suffix] for m in ['accuracy','precision','recall','f1','auc_roc']],width=.34,color=color,label=label)
            ax.bar_label(bars,labels=[f'{b.get_height():.3f}' if np.isfinite(b.get_height()) else '' for b in bars],padding=3,fontsize=10)
        ax.text(4-.18,.10,'Não informado\nno relatório',ha='center',va='bottom',fontsize=8,color='#536779')
        ax.set_xticks(x,['Acurácia','Precisão','Recall','F1','AUC-ROC']);ax.set_ylim(0,1.3);ax.set_yticks(np.arange(0,1.01,.2));ax.set_ylabel('Métrica (0–1)')
        ax.set_title('A | Núcleo 1 — métricas publicadas e atuais',loc='left',weight='bold',fontsize=12)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,fontsize=9);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        for col,(variant,label) in enumerate([('original','sem filtragem'),('filtrado','com filtragem')]):
            ax=fig.add_subplot(gs[1,col]);q=comparison.loc[(comparison.station==st)&(comparison.variant==variant)].set_index('model').reindex([f'M{i}' for i in range(1,13)])
            y=np.arange(12)
            ax.axhspan(7.5,11.5,color='#EAF3F8',zorder=0)
            before=ax.barh(y-.18,q.RMSE_relatorio,height=.32,color=OLD,label='Relatório')
            after=ax.barh(y+.18,q.RMSE_novo,height=.32,color=NEW,label='Nova modelagem')
            for bars in (before,after):
                for bar in bars:
                    if np.isfinite(bar.get_width()):ax.text(bar.get_width()+.15,bar.get_y()+bar.get_height()/2,f'{bar.get_width():.2f}',va='center',fontsize=8)
            for i in range(8,12):ax.text(.15,i-.18,'sem valor no relatório',fontsize=7.5,color='#536779',va='center')
            ax.set_yticks(y,[f'M{i}'+(' *' if i>=9 else '') for i in range(1,13)]);ax.invert_yaxis();ax.set_xlim(0,maximum)
            ax.set_title(f'{"B" if col==0 else "C"} | Núcleo 2 — RMSE {label}',loc='left',fontsize=12,weight='bold')
            ax.set_xlabel('RMSE (TECU) — menor é melhor');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        fig.suptitle(f'{st} | Relatório anterior × nova modelagem',fontsize=19,weight='bold',y=.985)
        fig.subplots_adjust(left=.07,right=.97,top=.92,bottom=.15)
        fig.text(.07,.09,'Comparação descritiva: Núcleo 2 do relatório = 28/03/2024; nova rodada = 18/02/2024. Dados, alvos e protocolo diferem.',fontsize=10,weight='bold')
        fig.text(.07,.06,'Núcleo 1 anterior: tabela 6.2 (suporte/período não discriminados na tabela); atual: teste comum de 18/02/2024.\n'
            '* M9–M12: arquiteturas acrescentadas; ausência de resultado antigo não significa erro zero. Valores antigos arredondados no PDF.',fontsize=9)
        fig.text(.07,.022,'Diferenças não isolam o efeito do retreinamento. M1/M2: LSTM conv.; M3/M4: atenção conv.; M5/M6: LSTM piecewise;\n'
            'M7/M8: atenção piecewise; M9/M10: ConvLSTM conv.; M11/M12: ConvLSTM piecewise. Ímpar: uni; par: multivariado.',fontsize=8.5)
        export(fig,DEST/f'{st}_relatorio_vs_nova_modelagem_auc_roc')
    sources[str(Path(__file__).resolve().relative_to(ROOT))]=sha(Path(__file__).resolve())
    for p,h in sources.items():assert sha(ROOT/p)==h
    (DEST/'auditoria.json').write_text(json.dumps({'sources_sha256':sources,'old_rmse_values':len(old),'current_rmse_values':len(current),'classification_stations':len(cls),'mapping_pdf_row_order':ORDER,'pdf_pages':PAGES},ensure_ascii=False,indent=2),encoding='utf-8')
    text='# Relatório anterior × nova modelagem\n\nUma figura por estação, com três painéis: classificação do Núcleo 1; RMSE original; RMSE filtrado.\n\n'
    for st in PAGES:text+='- '+st+': '+' · '.join(f'[{ext.upper()}]({st}_relatorio_vs_nova_modelagem_auc_roc.{ext})' for ext in ['png','pdf','svg'])+'\n'
    text+='\n## AUC-ROC acrescentada\n\nAUC-ROC atual calculada com classe_future e probabilidade_severa, exclusivamente nas épocas-alvo de 18/02/2024 e previsões do modelo final. O suporte e os positivos coincidem com a tabela auditada. Usa probabilidades contínuas, sem aplicar o limiar de classificação; não é average precision/AUC-PR. O relatório não informa AUC-ROC na tabela 6.2 nem nas figuras 6.5–6.8: o valor anterior permanece ausente, não zero. CSVs de AUC e curvas ROC acompanham as figuras.\n'
    text+='\n## Fontes e interpretação\n\nValores antigos extraídos das tabelas 6.2 a 6.6 do PDF fornecido pelo autor. Tabela 6.2: página física 80 (impressa 71); FRTZ: 83 (74); PALM: 84 (75); POAL: 86 (77); PRU2: 88 (79). A ordem das linhas do PDF corresponde a M1, M5, M2, M6, M3, M7, M4, M8; a extração foi remapeada por arquitetura. Textos extraídos e CSVs acompanham as figuras.\n\nNúcleo 2 anterior é apresentado no relatório como teste em 28/03/2024; o atual usa 18/02/2024, 48 épocas e referência comum observada. A tabela antiga do Núcleo 1 não discrimina suporte ou intervalo temporal: não se assume que compartilha o teste da regressão. O relatório usa seleção por F1 e limiar 0,75; a rodada atual usa F2 e limiares por estação. As diferenças de período, dados, referência e seleção impedem atribuir diferenças de métricas apenas ao modelo.\n\nM9–M12 não constam no relatório e são apresentados somente com valores novos, nunca com zero como referência antiga. Valores antigos são arredondados a duas casas. Acurácia atual foi calculada por (TP + TN)/n; as demais métricas vêm das tabelas auditadas. Acurácia alta com eventos raros não substitui precisão e recall.\n\nLegenda sugerida: Comparação descritiva entre métricas publicadas no estudo dirigido e a rodada atual, por estação. Cinza: relatório; azul: nova modelagem. Painel A: métricas da classificação. Painéis B/C: RMSE sem/com filtragem. Diferenças refletem experimentos com períodos e protocolos distintos; não constituem evidência isolada de ganho pelo retreinamento.\n\nReprodução: `.venv/Scripts/python.exe -X utf8 scripts/fevereiro/comparar_relatorio.py`.\n'
    (DEST/'README.md').write_text(text,encoding='utf-8')
    print('4 figuras; 64 RMSE antigos; 96 RMSE atuais; 4 classificadores comparados.')


if __name__=='__main__':main()
