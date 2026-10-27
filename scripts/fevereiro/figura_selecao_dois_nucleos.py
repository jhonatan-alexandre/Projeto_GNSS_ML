"""Visão conjunta da seleção efetiva de XGBoost, LSTM e ConvLSTM."""
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


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    sources={};records=[]
    def read(p):sources[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text(encoding='utf-8'))
    fig,axes=plt.subplots(3,4,figsize=(19,13))
    fig.subplots_adjust(left=.065,right=.98,top=.88,bottom=.25,hspace=.8,wspace=.35)
    fig.suptitle('Escolha dos hiperparâmetros | Dois núcleos · quatro estações',fontsize=22,weight='bold',y=.975)
    fig.text(.065,.932,'A | Núcleo 1 — XGBoost: maior F₂ nas previsões OOF de seleção',fontsize=13,weight='bold')
    fig.text(.065,.672,'B | Núcleo 2 — LSTM/Attention M1–M8: menor MSE normalizado de validação',fontsize=13,weight='bold')
    fig.text(.065,.438,'C | Núcleo 2 — ConvLSTM M9–M12: menor RMSE médio dos ramos na seleção',fontsize=13,weight='bold')
    for col,st in enumerate(STATIONS):
        base=ROOT/'experimentos'/f'{st.lower()}_comum_18fev2024'/'resultados'/st
        cfg=read(base/'nucleo1/hiperparametros_xgboost.json');assert not cfg['teste_usado_na_selecao']
        path=base/'nucleo1/busca_hiperparametros.csv';sources[str(path.relative_to(ROOT))]=sha(path)
        c=pd.read_csv(path).sort_values('candidato');w=c.sort_values(['f2','precision','candidato'],ascending=[False,False,True]).iloc[0]
        assert np.isclose(w.limiar,cfg['limiar_probabilidade'])
        ax=axes[0,col];bars=ax.bar(c.candidato,c.f2,color=['#009E73' if n==w.candidato else '#9AB8CC' for n in c.candidato])
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
        ax.set(ylim=(0,1),xticks=c.candidato,xticklabels=[f'C{int(n)}' for n in c.candidato],ylabel='F₂ — maior é melhor' if col==0 else '',title=f'{st} | C{int(w.candidato)} • τ={w.limiar:.4f}')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for row in c.to_dict('records'):records.append(dict(station=st,family='XGBoost',candidate=f'C{int(row["candidato"])}',metric='F2',score=row['f2'],selected=row['candidato']==w.candidato,learning_rate=row['learning_rate'],n_estimators=row['n_estimators'],max_depth=row['max_depth'],balanceamento=row['balanceamento'],threshold=row['limiar']))
        cfg=read(base/'nucleo2/hiperparametros_lstm.json');search=cfg['selecao_taxa_aprendizado'];assert not search['dados_teste_utilizados']
        c=sorted(search['candidatas'],key=lambda r:r['learning_rate']);rates=np.array([r['learning_rate'] for r in c]);values=np.array([r['melhor_val_loss'] for r in c]);idx=int(values.argmin())
        assert rates[idx]==cfg['hiperparametros']['learning_rate']
        ax=axes[1,col];ax.plot(rates,values,'-o',color='#0072B2',ms=4);ax.scatter(rates[idx],values[idx],color='#D55E00',s=80,zorder=5,edgecolor='white')
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Taxa η (escala log)');ax.set_ylabel('MSE normalizado (log)' if col==0 else '')
        ax.set_title(f'{st} | η={rates[idx]:g}\nMSE mínimo={values[idx]:.5g}',fontsize=10);ax.grid(alpha=.2)
        for row in c:records.append(dict(station=st,family='LSTM_M1_M8',candidate=f'lr={row["learning_rate"]:g}',metric='MSE_normalizado',score=row['melhor_val_loss'],selected=row['learning_rate']==rates[idx],learning_rate=row['learning_rate']))
        base=OUT/'estacoes'/st/'convlstm';frozen=read(base/'selecao_congelada.json');assert not frozen['test_used_for_selection']
        scores=np.zeros((4,2));selected=[]
        for i,n in enumerate(range(9,13)):
            for j in (0,1):
                label=f'c{j}'+('_validacao_anterior' if st=='POAL' and n>=11 else '')
                results=[read(base/'busca'/f'M{n}'/v/label/'resumo.json') for v in ('original','filtrado')]
                assert all(r['phase']=='search' for r in results)
                scores[i,j]=np.mean([r['metrics_common']['RMSE'] for r in results])
            win=int(scores[i].argmin());selected.append(win)
            for v in ('original','filtrado'):
                choice=frozen['selected'][f'M{n}_{v}'];assert choice['candidate'].startswith(f'c{win}')
            for j in (0,1):records.append(dict(station=st,family=f'ConvLSTM_M{n}',candidate=f'C{j}',metric='RMSE_medio_ramos',score=scores[i,j],selected=j==win,learning_rate=.001 if j==0 else .0003,filters=16 if j==0 else 32))
        ax=axes[2,col];x=np.arange(4)
        for j,color in [(0,'#0072B2'),(1,'#E69F00')]:
            bars=ax.bar(x+(-.18 if j==0 else .18),scores[:,j],width=.34,color=color,label=f'C{j}')
            for i,b in enumerate(bars):
                if selected[i]==j:ax.text(b.get_x()+b.get_width()/2,b.get_height()+scores.max()*.025,'★',ha='center',fontsize=12)
        ax.set_xticks(x,['M9','M10','M11','M12']);ax.set_ylim(0,scores.max()*1.28)
        ax.set_title(st+(' *' if st=='POAL' else ''),fontsize=10);ax.set_ylabel('RMSE de seleção (TECU)' if col==0 else '')
        ax.legend(ncol=2,fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.065,.193,'XGBoost — C0: 150 árvores, profundidade 6, η=0,05, SMOTE. C1: 300 árvores, profundidade 3, η=0,05, SMOTE.\n'
        'C2: 400 árvores, profundidade 4, η=0,03, pesos de classe. C3: 300 árvores, profundidade 3, η=0,05, pesos de classe.\n'
        'C1–C3: min_child_weight=5 e λ=5. Desempate: precisão, depois ID. Limiar próprio otimizado por F₂ em cada candidata.',fontsize=9,linespacing=1.5)
    fig.text(.065,.127,'LSTM — taxa escolhida na rede convencional multivariada ORIGINAL e compartilhada entre M1–M8 e os dois ramos.\n'
        'Demais parâmetros fixos: 3×31 unidades, dropout 0,2, Adam, MSE, batch ≤400; busca até 50 épocas e patience 3.',fontsize=9,linespacing=1.5)
    fig.text(.065,.083,'ConvLSTM — C0: 16 filtros, η=0,001. C1: 32 filtros, η=0,0003. ★ = candidata escolhida por modelo para os dois ramos.\n'
        'Adam, MSE, dropout 0,2, batch 64; até 60 épocas e patience 8. Média aritmética dos RMSE original e filtrado.',fontsize=9,linespacing=1.5)
    fig.text(.065,.035,'* POAL M11/M12, slots 14–41: validação anterior autorizada (seleção 03–09/02). Demais slots/estações: seleção 10–16/02.\n'
        'Valores de seleção, não de teste. Teste de 18/02/2024 preservado. As escalas e unidades dos três painéis não são intercambiáveis.',fontsize=9,linespacing=1.5)
    export(fig,DEST/'todas_estacoes_escolha_hiperparametros_dois_nucleos')
    pd.DataFrame(records).to_csv(DEST/'escolha_hiperparametros_dois_nucleos.csv',index=False)
    sources[str(Path(__file__).resolve().relative_to(ROOT))]=sha(Path(__file__).resolve())
    for p,h in sources.items():assert sha(ROOT/p)==h
    (DEST/'auditoria_hiperparametros_dois_nucleos.json').write_text(json.dumps({'sources_sha256':sources,'test_used':False,'candidates':len(records)},ensure_ascii=False,indent=2),encoding='utf-8')
    p=DEST/'SELECAO_NUCLEO1.md';t=p.read_text(encoding='utf-8')
    if 'todas_estacoes_escolha_hiperparametros_dois_nucleos' not in t:
        t+='\n## Escolha dos hiperparâmetros dos dois núcleos — quatro estações\n\n'+' · '.join(f'[{e.upper()}](todas_estacoes_escolha_hiperparametros_dois_nucleos.{e})' for e in ('png','pdf','svg'))+'\n\nXGBoost: F2 da seleção por candidata. LSTM/Attention: escolha da taxa por MSE normalizado. ConvLSTM: escolha conjunta dos ramos por RMSE médio. Marcadores identificam escolhas congeladas, conferidas com os registros. Os parâmetros fixos não são apresentados como se tivessem sido otimizados. CSV e auditoria acompanham a figura.\n'
        p.write_text(t,encoding='utf-8')
    print('Figura conjunta concluída;',len(records),'candidatas verificadas.')


if __name__=='__main__':main()
