"""S4 observado/alertas e sobreposição das previsões finais, sem retreinamento."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from core_convlstm import ROOT, OUT, sha
from relatorio import style, export, time_axis, GRID, DAY

DEST=OUT/'artigo/figuras/series_nucleos'
STATIONS=('FRTZ','PALM','POAL','PRU2')
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9',
        '#332288','#882255','#44AA99','#999933','#AA4499','#661100']


def intervals(ax,index,mask,color,alpha):
    """Shade contiguous 30-minute bins; missing bins remain unshaded."""
    selected=np.flatnonzero(np.asarray(mask,dtype=bool))
    for group in np.split(selected,np.flatnonzero(np.diff(selected)>1)+1):
        if len(group):ax.axvspan(index[group[0]],index[group[-1]]+pd.Timedelta(minutes=30),color=color,alpha=alpha,lw=0)


def s4figure(station,s,threshold,day=False):
    d=s.loc[GRID] if day else s
    fig,axes=plt.subplots(3,1,figsize=(15,8),sharex=True,
        gridspec_kw={'height_ratios':[3,1.5,1]},layout='constrained')
    axes[0].fill_between(d.index,d.s4_q10,d.s4_q90,color='#0072B2',alpha=.18,label='Percentis 10–90 entre satélites')
    axes[0].plot(d.index,d.s4_mediana,color='#0072B2',lw=1,label='Mediana entre satélites')
    axes[0].plot(d.index,d.s4_maximo,color='#666666',lw=.7,alpha=.75,label='Máximo entre satélites')
    axes[0].axhline(.5,color='#B22222',ls='--',lw=1,label='S4 severo ≥ 0,5')
    intervals(axes[0],d.index,d.n_severos.fillna(0)>0,'#D55E00',.17)
    axes[0].plot([],[],color='#D55E00',lw=7,alpha=.25,label='Época com ≥ 1 satélite severo')
    axes[0].set_ylabel('S4');axes[0].legend(ncol=3,fontsize=8,loc='upper left')
    axes[1].plot(d.index,100*d.fracao_severa,color='#B22222',lw=1,label='Satélites com S4 ≥ 0,5 / observados')
    axes[1].plot(d.index,100*d.fracao_alerta,color='#7B3294',lw=1,label='Alertas / previsões disponíveis')
    intervals(axes[1],d.index,d.n_alertas.fillna(0)>0,'#7B3294',.12)
    axes[1].set_ylabel('Fração (%)');axes[1].set_ylim(-2,102);axes[1].legend(ncol=2,fontsize=8)
    axes[2].plot(d.index,d.n_observados,color='#222222',lw=1,label='Satélites com S4 observado')
    axes[2].plot(d.index,d.n_previsoes,color='#7B3294',lw=1,label='Satélites com previsão disponível')
    axes[2].set_ylabel('Satélites');axes[2].legend(ncol=2,fontsize=8)
    if day:time_axis(axes[-1])
    else:
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        axes[-1].set_xlabel('Data UTC — 01/12/2023 a 18/02/2024')
        axes[-1].set_xlim(d.index.min(),d.index.max()+pd.Timedelta(minutes=30))
    fig.suptitle(f'{station} | Núcleo 1 — '+('18/02/2024' if day else 'histórico de S4 e épocas degradadas'),fontsize=17,weight='bold')
    fig.supxlabel(f'Alerta: probabilidade ≥ {threshold:.4f}, alinhado à época-alvo (+30 min). Sem previsão: lacuna, não ausência de alerta.\n'
        'Degradação indica cintilação pelo S4; não mede erro de posicionamento. Histórico: previsões OOF e do modelo final conforme CSV.',fontsize=9)
    export(fig,DEST/station/('nucleo1_s4_dia_teste' if day else 'nucleo1_s4_historico'))


def vtecfigure(station,p,observed,split=False):
    fig,axes=plt.subplots(2 if split else 1,1,figsize=(15,10 if split else 7),sharex=True,sharey=True,squeeze=False)
    axes=axes[:,0]
    for i,ax in enumerate(axes):
        variants=[('original','Original')] if split and i==0 else [('filtrado','Filtrado')] if split else [('original','Original'),('filtrado','Filtrado')]
        for variant,label in variants:
            for n in range(1,13):
                q=p.loc[(p.model==f'M{n}')&(p.variant==variant)].sort_values('datetime')
                assert pd.DatetimeIndex(q.datetime).equals(GRID)
                ax.plot(q.datetime,q.predito,color=COLORS[n-1],ls='-' if variant=='original' else '--',lw=1.5,alpha=.85)
        ax.plot(GRID,observed,color='black',lw=2.6,marker='o',ms=3,zorder=20)
        ax.set_ylabel('VTEC (TECU)');time_axis(ax)
        if split:ax.set_title(variants[0][1],loc='left',weight='bold')
    handles=[Line2D([],[],color='black',lw=2.6,marker='o',ms=3,label='Observado comum')]
    handles += [Line2D([],[],color=c,lw=2,label=f'M{n}') for n,c in enumerate(COLORS,1)]
    handles += [Line2D([],[],color='#666666',ls=ls,label=label) for ls,label in [('-','Original'),('--','Filtrado')]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=8,fontsize=9,frameon=False)
    fig.suptitle(f'{station} | VTEC observado e M1–M12 — 18/02/2024',fontsize=17,weight='bold',y=.995)
    fig.subplots_adjust(top=.82 if not split else .87,bottom=.16 if not split else .12,hspace=.22,left=.07,right=.98)
    fig.text(.07,.025,'Referência comum: mediana observada entre satélites, a cada 30 min. Cor = modelo; linha = ramo.\n'
        'Previsões finais congeladas: modelos convencionais e por horário têm horizontes distintos, conforme o protocolo.',fontsize=9)
    export(fig,DEST/station/('nucleo2_vtec_paineis' if split else 'nucleo2_vtec_todos_sobrepostos'))


def main():
    style();DEST.mkdir(parents=True,exist_ok=True);sources={};counts={}
    def source(path):sources[str(path.relative_to(ROOT))]=sha(path);return path
    pred=pd.read_csv(source(OUT/'artigo/tabelas/predicoes_finais_18fev.csv'),parse_dates=['datetime'])
    assert len(pred)==4608 and np.isfinite(pred.predito).all()
    for station in STATIONS:
        folder=DEST/station;folder.mkdir(exist_ok=True)
        base=ROOT/'experimentos'/f'{station.lower()}_comum_18fev2024'
        f=pd.read_csv(source(base/'dados_processados'/station/f'{station}_features.csv'),usecols=['datetime','svid','S4'],parse_dates=['datetime'])
        f=f.loc[f.datetime.between('2023-12-01','2024-02-18 23:30')]
        assert not f.duplicated(['datetime','svid']).any()
        grid=pd.date_range('2023-12-01','2024-02-18 23:30',freq='30min')
        s=f.groupby('datetime').S4.agg(n_observados='count',s4_mediana='median',s4_maximo='max',
            s4_q10=lambda x:x.quantile(.1),s4_q90=lambda x:x.quantile(.9),n_severos=lambda x:int((x>=.5).sum())).reindex(grid)
        s['fracao_severa']=s.n_severos/s.n_observados.replace(0,np.nan)
        c=pd.read_csv(source(base/'resultados'/station/'nucleo1/predicoes_xgboost.csv'),parse_dates=['target_datetime'])
        c=c.loc[c.target_datetime.isin(grid)].copy()
        assert not c.duplicated(['target_datetime','svid']).any()
        thresholds=c.limiar_probabilidade.dropna().unique();assert len(thresholds)==1
        threshold=float(thresholds[0]);available=c.probabilidade_severa.notna()
        c['alerta']=c.sinalizar_filtro.astype(str).str.lower().eq('true')
        assert (c.loc[available,'alerta']==(c.loc[available,'probabilidade_severa']>=threshold)).all()
        c['disponivel']=available
        a=c.groupby('target_datetime').agg(n_previsoes=('disponivel','sum'),n_alertas=('alerta','sum'))
        s=s.join(a).fillna({'n_previsoes':0,'n_alertas':0})
        s['fracao_alerta']=s.n_alertas/s.n_previsoes.replace(0,np.nan)
        s.index.name='datetime';s.to_csv(folder/'nucleo1_serie_30min.csv')
        c.to_csv(folder/'nucleo1_alertas_por_satelite.csv',index=False)
        s4figure(station,s,threshold);s4figure(station,s,threshold,day=True)
        p=pred.loc[pred.station==station].copy();assert len(p)==1152
        obs=pd.read_csv(source(OUT/'artigo/dados_dos_graficos'/station/'referencia_observada.csv'),parse_dates=['datetime']).set_index('datetime').VTEC_observado.reindex(GRID)
        assert obs.notna().all()
        for _,q in p.groupby(['model','variant']):
            np.testing.assert_allclose(q.sort_values('datetime').observado_comum,obs,rtol=1e-9,atol=1e-8)
        p.to_csv(folder/'nucleo2_previsoes_sobrepostas.csv',index=False)
        vtecfigure(station,p,obs);vtecfigure(station,p,obs,split=True)
        counts[station]={'epocas_historico':len(s),'epocas_s4_severo':int((s.n_severos>0).sum()),'epocas_alerta':int((s.n_alertas>0).sum()),'previsoes_vtec':len(p),'limiar_probabilidade':threshold}
        print(station,'concluída',flush=True)
    source(Path(__file__).resolve())
    for path,digest in sources.items():assert sha(ROOT/path)==digest
    (DEST/'auditoria_series.json').write_text(json.dumps({'sources_sha256':sources,'counts':counts,'figuras':16,'formatos':['png','pdf','svg']},ensure_ascii=False,indent=2),encoding='utf-8')
    text='# Séries dos Núcleos 1 e 2\n\n'
    for station in STATIONS:
        text+=f'## {station}\n\n'
        for name,label in [('nucleo1_s4_historico','S4 histórico e épocas degradadas'),('nucleo1_s4_dia_teste','S4 e alertas no dia de teste'),('nucleo2_vtec_todos_sobrepostos','VTEC: todos os modelos sobrepostos'),('nucleo2_vtec_paineis','VTEC: original e filtrado em painéis')]:
            text+=f'- {label}: '+ ' · '.join(f'[{ext.upper()}]({station}/{name}.{ext})' for ext in ('png','pdf','svg'))+'\n'
    text+='\n## Critérios e legendas\n\nS4: a faixa laranja identifica intervalos de 30 minutos com pelo menos um satélite com S4 ≥ 0,5 nos dados reamostrados. A linha azul é a mediana, a faixa azul representa percentis 10–90 e a linha cinza é o máximo. A fração severa usa apenas satélites com S4 disponível. Degradação refere-se à cintilação, não a erro de coordenadas.\n\nAlertas: probabilidades alinhadas por target_datetime e svid; o limiar é específico da estação. A faixa roxa identifica épocas com alertas. A fração de alertas usa somente previsões disponíveis. Períodos sem previsão permanecem lacunas; o painel de cobertura explicita a disponibilidade. O CSV preserva a origem das probabilidades (OOF/modelo final/indisponível). As faixas são intervalos gráficos de 30 min, não durações comprovadas de eventos contínuos.\n\nVTEC: 24 curvas finais (12 modelos × 2 ramos), com observado comum em preto; cores identificam modelos e traços distinguem original/filtrado. Apenas 18/02/2024 possui as previsões finais desta comparação. Não são extrapoladas previsões para todo o histórico. Modelos convencionais e por horário possuem horizontes diferentes conforme o protocolo.\n\nCada pasta contém os CSVs utilizados. Fontes e hashes em auditoria_series.json. Figuras complementares: não alteram modelos, limiares nem métricas auditadas.\n\nReprodução: `.venv/Scripts/python.exe -X utf8 scripts/fevereiro/figuras_series.py`.\n'
    (DEST/'README.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
