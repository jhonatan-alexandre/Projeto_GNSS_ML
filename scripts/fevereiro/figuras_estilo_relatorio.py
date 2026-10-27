"""Figuras no estilo visual do estudo dirigido, com observações em dispersão."""
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
from relatorio import export, GRID, time_axis
from figuras_series import COLORS, STATIONS

DEST=OUT/'artigo/figuras/estilo_estudo_dirigido'
REFERENCE=ROOT.parent/'Documentação'/'Estudo dirigido - relatório (Jhonatan).pdf'
NAMES=['Conv. LSTM uni','Conv. LSTM multi','Conv. Attention uni','Conv. Attention multi',
       'Piecewise LSTM uni','Piecewise LSTM multi','Piecewise Attention uni','Piecewise Attention multi',
       'Conv. ConvLSTM uni','Conv. ConvLSTM multi','Piecewise ConvLSTM uni','Piecewise ConvLSTM multi']


def s4_plot(station,f,c,raw,day):
    obs=f.loc[f.datetime.isin(GRID)].copy() if day else f.copy()
    q=obs.merge(c[['target_datetime','svid','probabilidade_severa','limiar_probabilidade','sinalizar_filtro']],
        left_on=['datetime','svid'],right_on=['target_datetime','svid'],how='left',validate='one_to_one')
    q=q.dropna(subset=['S4']);severe=q.S4>=.5
    alert=q.probabilidade_severa.notna() & q.sinalizar_filtro.astype(str).str.lower().eq('true')
    fig,ax=plt.subplots(figsize=(15,6.8))
    background=raw if day else obs
    ax.scatter(background.datetime,background.S4,s=3 if day else 2,color='#888888',alpha=.22,
        linewidths=0,rasterized=True,zorder=1)
    points=ax.scatter(q.loc[severe,'datetime'],q.loc[severe,'S4'],c=q.loc[severe,'svid'],
        cmap='viridis',vmin=float(f.svid.min()),vmax=float(f.svid.max()),s=17 if day else 5,
        linewidths=0,rasterized=True,zorder=3)
    ax.scatter(q.loc[alert,'datetime'],q.loc[alert,'S4'],s=58 if day else 12,facecolors='none',
        edgecolors='#E31A1C',linewidths=1 if day else .45,rasterized=True,zorder=4)
    ax.axhline(.5,color='#333333',ls='--',lw=1,zorder=2)
    cb=fig.colorbar(points,ax=ax,pad=.018);cb.set_label('ID do satélite (SVID)')
    handles=[Line2D([],[],marker='.',ls='',color='#888888',label='S4 real — '+('dados nativos' if day else 'reamostrado 30 min')),
        Line2D([],[],marker='o',ls='',color='#440154',label='Cintilação real: S4 ≥ 0,5 (30 min)'),
        Line2D([],[],marker='o',ls='',markerfacecolor='none',markeredgecolor='#E31A1C',markersize=8,label='Alerta emitido — época-alvo'),
        Line2D([],[],color='#333333',ls='--',label='Limiar S4 = 0,5')]
    ax.legend(handles=handles,loc='upper right',fontsize=9,framealpha=.95)
    ax.set_ylabel('Índice S4');ax.set_title(f'{station} | Núcleo 1 — cintilação observada e alertas',loc='left',fontsize=13)
    if day:time_axis(ax)
    else:
        ax.set_xlim(pd.Timestamp('2023-12-01'),pd.Timestamp('2024-02-19'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.set_xlabel('Data UTC — 01/12/2023 a 18/02/2024')
    threshold=c.limiar_probabilidade.dropna().iloc[0]
    fig.subplots_adjust(left=.07,right=.96,top=.92,bottom=.18)
    fig.text(.07,.035,f'Círculos: P(S4 ≥ 0,5) ≥ {threshold:.4f}, previsão de +30 min alinhada por época-alvo e satélite.\n'
        'A altura do círculo é o S4 observado; o classificador prevê a classe, não o valor de S4. Sem probabilidade: sem avaliação de alerta.',fontsize=9)
    name='nucleo1_dia_teste' if day else 'nucleo1_historico'
    export(fig,DEST/station/name)
    q.to_csv(DEST/station/(name+'_pontos.csv'),index=False)
    return len(q),int(severe.sum()),int(alert.sum())


def vtec_plot(station,p,raw,variant):
    fig,ax=plt.subplots(figsize=(15,7.2))
    ax.scatter(raw.datetime,raw.VTEC,s=3,color='#888888',alpha=.25,linewidths=0,
        rasterized=True,zorder=1,label='VTEC real — pontos nativos')
    observed=p.groupby('datetime').observado_comum.first().reindex(GRID)
    ax.plot(GRID,observed,color='black',lw=2.2,zorder=10,label='VTEC real — mediana 30 min')
    selected=p if variant=='todos' else p.loc[p.variant==variant]
    for n in range(1,13):
        for branch in ('original','filtrado'):
            q=selected.loc[(selected.model==f'M{n}')&(selected.variant==branch)].sort_values('datetime')
            if q.empty:continue
            assert pd.DatetimeIndex(q.datetime).equals(GRID)
            np.testing.assert_allclose(q.observado_comum,observed,atol=1e-8)
            rmse=float(np.sqrt(np.mean((q.predito.to_numpy()-observed.to_numpy())**2)))
            label=f'M{n} {NAMES[n-1]} — RMSE: {rmse:.2f}'
            if variant=='todos':label=f'M{n} — '+branch
            ax.plot(q.datetime,q.predito,color=COLORS[n-1],lw=1.25,
                ls=('-' if branch=='original' else '--') if variant=='todos' else ('--' if n in (5,6,7,8,11,12) else '-'),
                label=label,zorder=4)
    title={'original':'sem filtragem','filtrado':'com filtragem','todos':'original e filtrado sobrepostos'}[variant]
    ax.set_title(f'{station} | VTEC observado e M1–M12 — {title}',loc='left',fontsize=13)
    ax.set_ylabel('VTEC (TECU)');time_axis(ax)
    ax.legend(loc='upper left',fontsize=7.4 if variant!='todos' else 7.2,ncol=2 if variant!='todos' else 4,framealpha=.94)
    fig.subplots_adjust(left=.07,right=.985,top=.93,bottom=.17)
    fig.text(.07,.035,'18/02/2024 • Pontos cinza: observações nativas de todos os satélites, preservadas nos dois ramos.\n'
        'Referência preta: mediana comum de 30 min. Horizontes convencional/piecewise distintos, conforme o protocolo; RMSE na referência comum.',fontsize=9)
    export(fig,DEST/station/f'nucleo2_vtec_{variant}')


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,'grid.alpha':.4,
        'grid.linewidth':.7,'axes.axisbelow':True,'axes.spines.top':True,'axes.spines.right':True,
        'axes.edgecolor':'#BBBBBB','pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})
    DEST.mkdir(parents=True,exist_ok=True);sources={};counts={}
    def source(path):sources[str(path)]=sha(path);return path
    source(REFERENCE)
    p=pd.read_csv(source(OUT/'artigo/tabelas/predicoes_finais_18fev.csv'),parse_dates=['datetime'])
    assert len(p)==4608
    for station in STATIONS:
        dest=DEST/station;dest.mkdir(exist_ok=True)
        base=ROOT/'experimentos'/f'{station.lower()}_comum_18fev2024'
        f=pd.read_csv(source(base/'dados_processados'/station/f'{station}_features.csv'),usecols=['datetime','svid','S4'],parse_dates=['datetime'])
        f=f.loc[f.datetime.between('2023-12-01','2024-02-18 23:30')]
        c=pd.read_csv(source(base/'resultados'/station/'nucleo1/predicoes_xgboost.csv'),parse_dates=['target_datetime'])
        raw=pd.read_csv(source(OUT/'artigo/dados_dos_graficos'/station/'satelites_nativos_18fev.csv'),parse_dates=['datetime'])
        raw.to_csv(dest/'observacoes_nativas_18fev.csv',index=False)
        counts[station]={'historico':s4_plot(station,f,c,raw,False),'dia_teste':s4_plot(station,f,c,raw,True),'pontos_nativos':len(raw)}
        q=p.loc[p.station==station];assert len(q)==1152
        for variant in ('original','filtrado','todos'):vtec_plot(station,q,raw,variant)
        q.to_csv(dest/'previsoes_vtec.csv',index=False)
        print(station,'concluída',flush=True)
    source(Path(__file__).resolve())
    for path,h in sources.items():assert sha(Path(path))==h
    (DEST/'auditoria.json').write_text(json.dumps({'sources_sha256':sources,'counts':counts,'reference_figures':['6.1','6.2','6.9','6.10'],'figures':20},ensure_ascii=False,indent=2),encoding='utf-8')
    text='# Figuras no estilo do estudo dirigido\n\nReferência visual: figuras 6.1–6.2 e 6.9–6.10 do relatório fornecido. Dados e resultados atuais: fevereiro de 2024, M1–M12.\n\n'
    for station in STATIONS:
        text+=f'## {station}\n\n'
        for name,label in [('nucleo1_dia_teste','Núcleo 1 — S4 real e alertas em 18/02'),('nucleo1_historico','Núcleo 1 — histórico completo'),('nucleo2_vtec_original','Núcleo 2 — sem filtragem'),('nucleo2_vtec_filtrado','Núcleo 2 — com filtragem'),('nucleo2_vtec_todos','Núcleo 2 — 24 curvas sobrepostas')]:
            text+='- '+label+': '+' · '.join(f'[{ext.upper()}]({station}/{name}.{ext})' for ext in ('png','pdf','svg'))+'\n'
    text+='\n## Leitura e legendas\n\nNúcleo 1: observações reais ao fundo em cinza; observações reamostradas com S4 ≥ 0,5 coloridas pelo SVID; círculos vermelhos indicam alertas alinhados por satélite e época-alvo. O círculo usa o valor observado de S4, não uma regressão predita. Círculos abaixo de 0,5 representam falsos positivos; pontos severos sem círculo são eventos perdidos apenas quando existe probabilidade disponível. No histórico, o fundo contém todos os dados reamostrados disponíveis (30 min); no dia de teste, todos os pontos nativos. A ausência de probabilidade não equivale à previsão negativa.\n\nNúcleo 2: pontos nativos de VTEC de todos os satélites em cinza; mediana observada comum em preto; previsões dos 12 modelos em cores, com RMSE na referência comum. Gráficos individuais por ramo usam tracejado para modelos piecewise. No gráfico com 24 curvas, contínuo identifica original e tracejado identifica filtrado. O fundo observado é o mesmo nos dois ramos, permitindo comparação direta. Os dados observados não foram recortados pela amplitude das previsões nem amostrados para a dispersão. Horizontes das arquiteturas diferem conforme o protocolo.\n\nA mediana atual foi preservada, embora a figura antiga descreva média. Não foram alterados treinamento, limiares, métricas ou dia de teste. CSVs de observações, pontos de S4/alertas e previsões acompanham as figuras; fontes SHA-256 em auditoria.json.\n\nReprodução: `.venv/Scripts/python.exe -X utf8 scripts/fevereiro/figuras_estilo_relatorio.py`.\n'
    (DEST/'README.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
