"""Seleção temporal dos hiperparâmetros e limiares do classificador."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core_convlstm import ROOT,OUT,sha
from relatorio import export

DEST=OUT/'artigo/figuras/comparacao_relatorio_nova_modelagem'
STATIONS=('FRTZ','PALM','POAL','PRU2')
METRICS=[('precision','Precisão','#0072B2'),('recall','Recall','#D55E00'),('f1','F1','#9966AA'),('f2','F₂','#009E73')]


def threshold_plot(ax,curve,w):
    # Todas as linhas participam da verificação; redução só na renderização.
    idx=np.unique(np.r_[np.linspace(0,len(curve)-1,min(1800,len(curve)),dtype=int),np.argmin(abs(curve.limiar-w.limiar))])
    q=curve.iloc[idx]
    for key,label,color in METRICS:ax.plot(q.limiar,q[key],color=color,lw=2 if key=='f2' else 1.2,label=label)
    ax.axvline(w.limiar,color='#333333',ls='--',lw=1.3,label=f'τ escolhido = {w.limiar:.4f}')
    ax.scatter([w.limiar],[w.f2],s=65,color='#009E73',edgecolor='white',zorder=5)
    ax.set(xlim=(0,1),ylim=(0,1.04),xlabel='Limiar de probabilidade τ',ylabel='Métrica na seleção (OOF)')
    ax.legend(loc='lower left',fontsize=8,ncol=2);ax.grid(alpha=.2)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})
    sources={};summary=[];curves={};winners={}
    def source(p):sources[str(p.relative_to(ROOT))]=sha(p);return p
    for st in STATIONS:
        base=ROOT/'experimentos'/f'{st.lower()}_comum_18fev2024'/'resultados'/st/'nucleo1'
        config=json.loads(source(base/'hiperparametros_xgboost.json').read_text(encoding='utf-8'))
        candidates=pd.read_csv(source(base/'busca_hiperparametros.csv'))
        full=pd.read_csv(source(base/'curvas_limiar_validacao.csv'))
        folds=pd.read_csv(source(base/'validacao_temporal.csv'))
        assert not config['teste_usado_na_selecao']
        w=candidates.sort_values(['f2','precision','candidato'],ascending=[False,False,True]).iloc[0]
        assert np.isclose(w.limiar,config['limiar_probabilidade'],rtol=0,atol=1e-10)
        for k in ['n_estimators','max_depth','learning_rate']:assert np.isclose(w[k],config['parametros'][k])
        assert len(candidates)==4 and candidates.folds_selecao.eq(4).all()
        for row in candidates.itertuples():
            curve=full.loc[full.candidato==row.candidato]
            top=curve.sort_values(['f2','limiar'],ascending=[False,False]).iloc[0]
            np.testing.assert_allclose([top.limiar,top.f2,top.precision,top.recall],[row.limiar,row.f2,row.precision,row.recall],rtol=1e-10)
        curve=full.loc[full.candidato==w.candidato].sort_values('limiar').copy()
        curve['f1']=np.divide(2*curve.precision*curve.recall,curve.precision+curve.recall,out=np.zeros(len(curve)),where=(curve.precision+curve.recall)>0)
        curve.to_csv(DEST/f'{st}_selecao_curva_limiar.csv',index=False)
        candidates.to_csv(DEST/f'{st}_selecao_candidatas.csv',index=False)
        folds.to_csv(DEST/f'{st}_selecao_folds.csv',index=False)
        assert folds.fold.tolist()==[1,2,3,4,5]
        assert pd.to_datetime(folds.fim_validacao).max()<pd.Timestamp('2024-02-18')
        curves[st]=curve;winners[st]=w
        summary.append(dict(station=st,**w.to_dict()))
        fig,axes=plt.subplots(2,2,figsize=(16,11.5),gridspec_kw={'hspace':.48,'wspace':.24})
        fig.subplots_adjust(left=.06,right=.97,top=.88,bottom=.17)
        fig.suptitle(f'{st} | Núcleo 1 — seleção do limiar e dos hiperparâmetros',fontsize=18,weight='bold',y=.975)
        fig.text(.06,.93,f'Candidata C{int(w.candidato)} selecionada • τ = {w.limiar:.6f} • F₂ = {w.f2:.4f} • Precisão = {w.precision:.4f} • Recall = {w.recall:.4f}',fontsize=11)
        ax=axes[0,0];threshold_plot(ax,curve,w);ax.set_title('A | Curvas de limiar da candidata vencedora',loc='left',fontsize=12,weight='bold')
        ax=axes[0,1];x=np.arange(len(candidates));cs=candidates.sort_values('candidato')
        for offset,key,label,color in [(-.23,'precision','Precisão','#0072B2'),(0,'recall','Recall','#D55E00'),(.23,'f2','F₂','#009E73')]:
            bars=ax.bar(x+offset,cs[key],width=.22,color=color,label=label)
            ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
        ax.set_xticks(x,[f'C{int(r.candidato)}'+(' ★' if r.candidato==w.candidato else '')+f'\nτ={r.limiar:.3f}' for r in cs.itertuples()])
        ax.set_ylim(0,1.15);ax.set_ylabel('Métrica no melhor limiar de cada candidata');ax.legend(ncol=3,fontsize=8,loc='upper center');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_title('B | Comparação das quatro candidatas',loc='left',fontsize=12,weight='bold')
        ax=axes[1,0]
        for key,label,color in METRICS+[('roc_auc','AUC-ROC','#555555')]:
            ax.plot(folds.fold,folds[key],'-o',color=color,lw=1.4,ms=4,label=label)
        ax.axvspan(4.5,5.5,color='#F4E3BD',alpha=.6,zorder=0)
        ax.set(xlim=(.8,5.5),ylim=(0,1.04),ylabel='Métrica por bloco',xlabel='Bloco temporal (fold)')
        ax.set_xticks([1,2,3,4,5],['1','2','3','4','5\nreservado'])
        ax.legend(ncol=3,fontsize=8,loc='lower left');ax.grid(alpha=.2)
        ax.set_title('C | Validação temporal da candidata escolhida',loc='left',fontsize=12,weight='bold')
        ax=axes[1,1];ax.axis('off');ax.set_title('D | Configurações avaliadas',loc='left',fontsize=12,weight='bold')
        records=[]
        def val(v):return '—' if pd.isna(v) else f'{v:g}'
        for r in cs.itertuples():records.append([f'C{int(r.candidato)}'+(' *' if r.candidato==w.candidato else ''),str(r.n_estimators),str(r.max_depth),f'{r.learning_rate:g}',r.balanceamento,val(r.min_child_weight),val(r.reg_lambda)])
        t=ax.table(cellText=records,colLabels=['ID','Árvores','Prof.','Taxa η','Balanceam.','Peso filho','λ L2'],cellLoc='center',bbox=[0,.38,1,.58],colWidths=[.08,.13,.09,.1,.23,.19,.18])
        t.auto_set_font_size(False);t.set_fontsize(8.5)
        for (i,j),cell in t.get_celld().items():
            cell.set_edgecolor('white');cell.set_facecolor('#DCEAF2' if i==0 else '#E4F1E9' if i>0 and cs.iloc[i-1].candidato==w.candidato else '#F2F4F5')
        params=config['parametros']
        ax.text(0,.28,f"* Selecionada; — parâmetro não explicitado no registro.\nSubsample={params['subsample']:g}; colsample_bytree={params['colsample_bytree']:g}; semente={params['random_state']}.\nPeso positivo no ajuste final: {params['scale_pos_weight']:.4f}.\n{int(w.n_selecao):,} observações OOF de seleção; quatro folds.".replace(',',' '),va='top',fontsize=9,linespacing=1.6)
        fig.text(.06,.105,'Critério: maior F₂ nos folds 1–4. Empate entre candidatas: maior precisão, depois menor ID; empate entre limiares: maior τ.',fontsize=10)
        fig.text(.06,.072,'F₂ = 5PR/(4P + R). O fold 5 é reservado; o teste de 18/02/2024 não participa da escolha.\n'
            'Curvas A/B: métricas agrupadas das previsões OOF; painel C: métricas de cada fold, não uma média usada como objetivo.',fontsize=9)
        fig.text(.06,.029,'Alerta: P(S4 futuro ≥ 0,5) ≥ τ. S4 = 0,5 define a classe; τ define a decisão probabilística.\n'
            'AUC-ROC utiliza probabilidades e não depende do limiar: é diagnóstico, não o critério de seleção desta busca.',fontsize=9)
        export(fig,DEST/f'{st}_selecao_limiar_hiperparametros_nucleo1')
        print(st,'concluída',flush=True)
    fig,axes=plt.subplots(2,2,figsize=(14,10));fig.subplots_adjust(hspace=.4,wspace=.22,bottom=.12,top=.89)
    for st,ax in zip(STATIONS,axes.flat):
        threshold_plot(ax,curves[st],winners[st]);ax.set_title(f'{st} | C{int(winners[st].candidato)} | τ={winners[st].limiar:.4f}',loc='left',weight='bold')
    fig.suptitle('Núcleo 1 | Escolha do limiar nas quatro estações',fontsize=18,weight='bold')
    fig.text(.08,.035,'Máximo F₂ nas previsões OOF dos quatro primeiros folds; quinto fold reservado.\nValores de validação: não são as métricas do teste de 18/02/2024. F1 derivado de precisão e recall.',fontsize=10)
    export(fig,DEST/'todas_estacoes_selecao_limiar_nucleo1')
    pd.DataFrame(summary).to_csv(DEST/'resumo_selecao_nucleo1.csv',index=False)
    source(Path(__file__).resolve());source(ROOT/'scripts/08_train_xgboost.py')
    for p,h in sources.items():assert sha(ROOT/p)==h
    (DEST/'auditoria_selecao_nucleo1.json').write_text(json.dumps({'sources_sha256':sources,'test_used_for_selection':False,'figures':5,'criterion':'F2 agrupado OOF folds 1-4; fold 5 reservado'},ensure_ascii=False,indent=2),encoding='utf-8')
    text='# Seleção de limiares e hiperparâmetros — Núcleo 1\n\n'
    for st in STATIONS:
        text+='- '+st+': '+' · '.join(f'[{ext.upper()}]({st}_selecao_limiar_hiperparametros_nucleo1.{ext})' for ext in ('png','pdf','svg'))+'\n'
    text+='- Visão conjunta: '+' · '.join(f'[{ext.upper()}](todas_estacoes_selecao_limiar_nucleo1.{ext})' for ext in ('png','pdf','svg'))+'\n'
    text+='\nPainel A: precisão, recall, F1 e F2 versus limiar da candidata vencedora. F1 é calculado como 2PR/(P+R). Painel B: métricas agrupadas das quatro candidatas, cada qual no seu melhor limiar. Painel C: métricas da candidata selecionada nos cinco folds, com o quinto reservado marcado. AUC-ROC por fold vem de validacao_temporal.csv; não é otimizada por limiar. Painel D: configurações e parâmetros selecionados.\n\nA escolha maximiza F2 nas previsões OOF agrupadas dos folds 1–4, não a média das métricas por fold. Desempate de candidatas: precisão decrescente, ID crescente. Desempate entre limiares: maior limiar. O teste final não foi usado na seleção. As métricas descritivas dos folds de seleção não constituem uma avaliação independente da escolha.\n\nAs curvas completas e tabelas foram exportadas; apenas a renderização das curvas pode usar até 1800 pontos, incluindo sempre o limiar selecionado. Todas as linhas participaram da verificação dos máximos. Valores ausentes de parâmetros não foram preenchidos artificialmente.\n\nFontes: busca_hiperparametros.csv, curvas_limiar_validacao.csv, hiperparametros_xgboost.json e validacao_temporal.csv de cada estação, com hashes em auditoria_selecao_nucleo1.json. Reprodução: `.venv/Scripts/python.exe -X utf8 scripts/fevereiro/figuras_selecao_nucleo1.py`.\n'
    (DEST/'SELECAO_NUCLEO1.md').write_text(text,encoding='utf-8')
    index=DEST/'README.md';text=index.read_text(encoding='utf-8');link='SELECAO_NUCLEO1.md'
    if link not in text:index.write_text(text+'\n[Gráficos de seleção dos limiares e hiperparâmetros do Núcleo 1]('+link+').\n',encoding='utf-8')


if __name__=='__main__':main()
