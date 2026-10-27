"""Tabelas e figuras estáticas; só incorpora resultados finais verificáveis."""
import argparse
import gc
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import average_precision_score
from core_convlstm import ROOT, OUT, sha, write, scores
from lock import acquire

STATIONS=('FRTZ','PALM','POAL','PRU2')
DAY=pd.Timestamp('2024-02-18')
GRID=pd.date_range(DAY,periods=48,freq='30min')
ARTICLE=OUT/'artigo'
COLORS={'original':'#0072B2','filtrado':'#D55E00'}
LABELS={'original':'Original','filtrado':'Filtrado'}


def validate_predictions(frame):
    q=frame.copy();q['datetime']=pd.to_datetime(q.datetime);q=q.sort_values('datetime')
    if not pd.DatetimeIndex(q.datetime).equals(GRID):
        raise ValueError('Previsões devem conter exatamente as 48 épocas de 18/02/2024')
    if not np.isfinite(q.predito).all(): raise ValueError('Previsões não finitas')
    return q


def classification_metrics(frame):
    d=frame.loc[frame.target_datetime.between(DAY,DAY+pd.Timedelta(days=1),inclusive='left')].copy()
    d=d.dropna(subset=['classe_future','probabilidade_severa'])
    if d.empty: raise ValueError('Classificação sem teste em 18/02')
    assert d.origem_probabilidade.eq('modelo_final').all()
    y=d.classe_future.astype(int);z=d.sinalizar_filtro.astype(str).str.lower().eq('true')
    tp=int(((y==1)&z).sum());fp=int(((y==0)&z).sum());fn=int(((y==1)&~z).sum());tn=int(((y==0)&~z).sum())
    p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
    return dict(n=len(d),TP=tp,FP=fp,FN=fn,TN=tn,positivos=tp+fn,precision=p,recall=r,
                f1=2*p*r/(p+r) if p+r else 0,f2=5*p*r/(4*p+r) if 4*p+r else 0,
                prevalence=(tp+fn)/len(d),average_precision=float(average_precision_score(y,d.probabilidade_severa)))


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.6,
                         'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})


def export(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf','svg'):
        fig.savefig(path.with_suffix('.'+extension),dpi=220,bbox_inches='tight')
    plt.close(fig)


def readable_table(frame,path):
    def value(x):
        if pd.isna(x): return '—'
        if isinstance(x,(float,np.floating)): return f'{x:.4f}'
        return str(x)
    lines=['| '+' | '.join(map(str,frame.columns))+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    lines.extend('| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    def tex(x): return value(x).replace('_',r'\_').replace('%',r'\%').replace('—','--')
    lines=[r'\begin{tabular}{'+'l'*len(frame.columns)+'}',r'\hline', ' & '.join(tex(x) for x in frame.columns)+r' \\',r'\hline']
    lines.extend(' & '.join(tex(x) for x in row)+r' \\' for row in frame.itertuples(index=False,name=None))
    lines.extend([r'\hline',r'\end{tabular}'])
    path.with_suffix('.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def time_axis(ax):
    ax.set_xlim(DAY,DAY+pd.Timedelta(days=1))
    ax.set_xticks(pd.date_range(DAY,periods=7,freq='4h'),['00','04','08','12','16','20','24'])
    ax.set_xlabel('Hora UTC — 18/02/2024')


def raw_day(path):
    blocks=[]
    for block in pd.read_csv(path,usecols=['datetime','svid','S4','VTEC'],chunksize=200000,parse_dates=['datetime']):
        selected=block.loc[block.datetime.between(DAY,DAY+pd.Timedelta(days=1),inclusive='left')]
        if not selected.empty: blocks.append(selected)
    if not blocks: raise ValueError(f'Sem dados nativos no dia: {path}')
    return pd.concat(blocks,ignore_index=True)


def station_figures(station,raw,features,series,predictions,metrics):
    directory=ARTICLE/'figuras'/station
    observed=features.groupby('datetime').VTEC.median().reindex(GRID)
    s4=features.groupby('datetime').S4.median().reindex(GRID)
    fig,axes=plt.subplots(2,1,figsize=(12,7.8),sharex=True,gridspec_kw={'height_ratios':[1,1.6]},layout='constrained')
    for ax,column in zip(axes,['S4','VTEC']):
        ax.scatter(raw.datetime,raw[column],s=3,color='#777777',alpha=.16,linewidths=0,
                   rasterized=True,label='Todos os satélites — dados nativos')
    axes[0].plot(GRID,s4,color='#009E73',lw=1.8,marker='.',ms=4,label='Mediana reamostrada — 30 min')
    axes[0].axhline(.5,color='#A32135',ls='--',lw=1,label='Limiar de S4 severo (0,5)')
    axes[0].set_ylabel('S4');axes[0].set_title(f'{station}  |  Observações por satélite e séries de 30 minutos',loc='left',fontweight='bold')
    axes[1].plot(GRID,observed,color='#222222',lw=1.5,marker='o',ms=3,label='VTEC observado — mediana 30 min')
    filtered=series.set_index('datetime').VTEC_filtered.reindex(GRID)
    axes[1].plot(GRID,filtered,color='#009E73',ls=':',lw=1.6,label='VTEC filtrado — reamostrado')
    if metrics:
        best=min(metrics,key=lambda r:r['RMSE']);model=best['model']
        for variant in COLORS:
            q=predictions.get((model,variant))
            if q is not None: axes[1].plot(q.datetime,q.predito,color=COLORS[variant],lw=2,label=f'{model} — {LABELS[variant]}')
        axes[1].set_title(f'{model}: menor RMSE neste dia entre os resultados finais disponíveis (ranking retrospectivo)',loc='left',fontsize=10)
    else:
        axes[1].set_title('Predições finais ainda em treinamento',loc='left',fontsize=10)
    axes[1].set_ylabel('VTEC (TECU)');time_axis(axes[1])
    for ax in axes: ax.legend(loc='upper left',fontsize=8,framealpha=.9,ncol=2)
    export(fig,directory/'dia_s4_vtec')
    if not predictions: return
    fig,axes=plt.subplots(4,3,figsize=(15,14),sharex=True,sharey=True,layout='constrained')
    parity,paraxes=plt.subplots(4,3,figsize=(12,14),sharex=True,sharey=True,layout='constrained')
    all_values=np.concatenate([observed.to_numpy()]+[q.predito.to_numpy() for q in predictions.values()])
    lower=float(all_values.min())-2;upper=float(all_values.max())+2
    for number,(ax,px) in enumerate(zip(axes.flat,paraxes.flat),start=1):
        model=f'M{number}'
        ax.scatter(raw.datetime,raw.VTEC,s=1,color='#888888',alpha=.10,rasterized=True,linewidths=0)
        ax.plot(GRID,observed,color='#333333',lw=1.2,label='Observado 30 min')
        px.plot([lower,upper],[lower,upper],color='#666666',ls='--',lw=1)
        found=False
        for variant in COLORS:
            q=predictions.get((model,variant))
            if q is None: continue
            found=True;ax.plot(q.datetime,q.predito,color=COLORS[variant],lw=1.5,label=LABELS[variant])
            px.scatter(observed,q.predito,color=COLORS[variant],s=15,alpha=.75,label=LABELS[variant])
        ax.set_title(model if found else f'{model} — pendente',loc='left',fontweight='bold');time_axis(ax)
        px.set_title(model if found else f'{model} — pendente',loc='left',fontweight='bold')
        px.set_xlim(lower,upper);px.set_ylim(lower,upper);px.set_aspect('equal',adjustable='box')
        if found: ax.legend(fontsize=7,loc='upper left');px.legend(fontsize=7,loc='upper left')
        ax.set_ylabel('VTEC (TECU)');px.set_xlabel('Observado (TECU)');px.set_ylabel('Predito (TECU)')
    fig.suptitle(f'{station} | M1–M12 — 18/02/2024 | cinza: todos os satélites',fontsize=16)
    parity.suptitle(f'{station} | Observado × predito — 48 épocas comuns',fontsize=15)
    export(fig,directory/'comparacao_temporal_m1_m12');export(parity,directory/'dispersao_observado_predito')


def gather_documents():
    destination=ARTICLE/'documentacao'/'historico';destination.mkdir(parents=True,exist_ok=True)
    roots=[ROOT/'docs',ROOT/'estudo_20240328'/'documentacao',ROOT/'resultados_artigo_12_modelos_20260922'/'documentacao',
           ROOT/'resultados',ROOT/'experimentos']
    files=[ROOT/'README.md',OUT/'README.md',ROOT/'estudo_20240328'/'README.md']
    decision_documents = [OUT/name for name in ('validacao_POAL_autorizada.json',
                          'proposta_validacao_POAL.json', 'RETOMADA_20260923.md') if (OUT/name).exists()]
    files.extend(decision_documents)
    for root in roots:
        files.extend(p for p in root.rglob('*') if p.suffix.lower() in ('.md','.txt'))
    # Preserva também as evidências compactas e a árvore de caminhos dos
    # documentos históricos, para manter seus links relativos existentes.
    files.extend(p for p in (ROOT/'estudo_20240328'/'documentacao').rglob('*')
                 if p.suffix.lower() in ('.json','.csv') and p.stat().st_size < 5_000_000)
    records=[];copies={}
    for path in sorted(set(files)):
        if not path.is_file(): continue
        digest=sha(path)
        target=destination/'fontes'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or sha(target)!=digest: shutil.copy2(path,target)
        copies[digest]=str(target.relative_to(ARTICLE))
        records.append({'source':str(path.relative_to(ROOT)),'sha256':digest,'copy':str(target.relative_to(ARTICLE)),
                        'mtime_utc_not_decision_date':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()})
    pd.DataFrame(records).to_csv(ARTICLE/'documentacao'/'indice_documental.csv',index=False)
    shutil.copy2(OUT/'README.md',ARTICLE/'documentacao'/'PROTOCOLO_E_RETOMADA.md')
    for path in decision_documents:
        shutil.copy2(path, ARTICLE/'documentacao'/path.name)
    guide='# Histórico para a redação\n\n'
    guide+='[Linha do tempo anterior](historico/fontes/estudo_20240328/documentacao/historico/LINHA_DO_TEMPO.md) · '
    guide+='[Ampliação de março](historico/fontes/estudo_20240328/documentacao/LINHA_DO_TEMPO_ATUAL.md) · '
    guide+='[Protocolo e retomada de fevereiro](PROTOCOLO_E_RETOMADA.md).\n\n'
    guide+='O índice documental registra origem, cópia e SHA-256. A data de modificação não é uma data comprovada de decisão. '
    guide+='Os arquivos históricos preservam os textos originais, inclusive afirmações válidas apenas para o estado dos dados e calendário daquela execução. '
    guide+='As referências internas a modelos antigos removidos são registros históricos, não resultados vigentes.\n\n'
    guide+='Correção de leitura: a linha N15 da cronologia anterior tem contagens textuais inconsistentes. A expansão M10–M12 acrescentou 6 configurações, 288 previsões e 194 redes por estação; o conjunto M9–M12 tem 8 configurações, 384 previsões e 196 redes por estação. O documento original permanece intacto.\n'
    (ARTICLE/'documentacao'/'GUIA_DA_LINHA_DO_TEMPO.md').write_text(guide,encoding='utf-8')
    timeline=pd.read_csv(ROOT/'estudo_20240328/documentacao/historico/LINHA_DO_TEMPO.csv').fillna('')
    rows=timeline.to_dict('records');order=max(int(r['ordem']) for r in rows)
    march=ROOT/'estudo_20240328/documentacao/LINHA_DO_TEMPO_ATUAL.md'
    for line in march.read_text(encoding='utf-8').splitlines():
        cells=[c.strip() for c in line.split('|')[1:-1]]
        if len(cells)==3 and re.fullmatch(r'N\d+',cells[0]):
            order+=1
            change=cells[1]
            if cells[0]=='N15':
                change='Acrescentar M10–M12 a FRTZ/PALM: 6 configurações, 288 previsões e 194 redes por estação; correção editorial das contagens do texto-fonte.'
            rows.append(dict(id=cells[0],ordem=order,etapa='Ampliação de março',data_referencia='Consultar fonte',
                precisao_data='Datas das etapas documentadas na fonte; sem inferência por mtime',
                mudanca=change,fontes='estudo_20240328/documentacao/LINHA_DO_TEMPO_ATUAL.md',
                limites='Antecedente com teste em 28/03; não misturar com fevereiro'))
    updates=[
        ('Data comum de fevereiro','Usuário confirmou 18/02/2024 e continuidade do fluxo ativo.','decisão confirmada'),
        ('Recuperação de FRTZ','Preservar seleção antiga com início de teste em 18/02, refazer a busca para configuração com 17–18/02 e reutilizar M1 após conferir hashes.','implementado'),
        ('Preparação de PRU2','Criar configuração de fevereiro, ler os novos ISMR de dezembro a fevereiro e concluir Núcleo 1 e séries; 48 épocas observadas em 18/02.','concluído'),
        ('Histórico no ajuste final','Acrescentar reajuste M1–M8 com todo desenvolvimento elegível até 16/02 e épocas já selecionadas; preservar seleção anterior.','em execução; verificar estado atual'),
        ('Quatro ConvLSTM','Programar M9–M12, uni/multivariado e convencional/piecewise, original/filtrado, duas candidatas e seleção temporal antes do reajuste.','programado; verificar estado atual'),
        ('Classificação no dia comum','Consolidar precisão, recall, F1, F2, TP, FP, FN e suporte das quatro estações sem revisão manual dos alertas.','concluído'),
        ('Figuras e acervo','Gerar figuras estáticas em PNG/PDF/SVG, dados para reprodução, índice documental e cronologia consolidada; tabelas finais só incorporam modelos verificados.','atualização durante o treinamento')]
    for index,(title,change,situation) in enumerate(updates,1):
        order+=1
        rows.append(dict(id=f'F{index:02d}',ordem=order,etapa=title,data_referencia='22/09/2026 — retomada (Brasília)',
            precisao_data='Data da sessão; horários específicos devem ser consultados nos logs UTC',
            situacao=situation,mudanca=change,fontes='estudo_20240218/README.md;estudo_20240218/estado_execucao.json',
            limites='Teste em 18/02/2024; conclusão das redes depende da auditoria final'))
    pd.DataFrame(rows).to_csv(ARTICLE/'documentacao'/'LINHA_DO_TEMPO_CONSOLIDADA.csv',index=False)
    lines=['# Linha do tempo consolidada','',
           'Datas de 2024 são observacionais; datas de 2026 são de execução. Campos sem data exata permanecem sem inferência.','']
    for row in rows:
        lines.extend([f"## {row['id']} — {row.get('etapa','')}",'',
                      str(row.get('mudanca','')), '',
                      '**Data / precisão:** '+str(row.get('data_referencia',''))+'; '+str(row.get('precisao_data',''))+'.','',
                      '**Fontes:** '+str(row.get('fontes','')), '',
                      '**Limites:** '+str(row.get('limites','')), ''])
    (ARTICLE/'documentacao'/'LINHA_DO_TEMPO_CONSOLIDADA.md').write_text('\n'.join(lines),encoding='utf-8')
    preliminary=[dict(file=str(p.relative_to(ARTICLE)),sha256=sha(p),
                      description='Versão preliminar gerada nesta retomada; a documentação vigente está no índice principal')
                 for p in destination.iterdir() if p.is_file() and re.match(r'^[0-9a-f]{12}_',p.name)]
    pd.DataFrame(preliminary).to_csv(ARTICLE/'documentacao'/'versoes_preliminares.csv',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    report_lock=acquire(ARTICLE/'relatorio.lock')
    style();tables=ARTICLE/'tabelas';tables.mkdir(parents=True,exist_ok=True)
    metrics=[];classifiers=[];status=[];all_predictions=[];sources={str(Path(__file__).resolve()):sha(__file__)}
    hyperparameters=[];partitions=[];classifier_parameters=[];filtering=[]
    digest_cache={}
    def check_hash(path,digest):
        path=str(path)
        if path not in digest_cache: digest_cache[path]=sha(path)
        assert digest_cache[path]==digest,path
    for station in STATIONS:
        base=ROOT/'experimentos'/f'{station.lower()}_comum_18fev2024';processed=base/'dados_processados'/station
        featurepath=processed/f'{station}_features.csv';seriespath=processed/f'{station}_dataset_regressao.csv'
        n1path=base/'resultados'/station/'nucleo1'/'predicoes_xgboost.csv'
        count=0
        if n1path.exists():
            frame=pd.read_csv(n1path,parse_dates=['target_datetime']);classifiers.append(dict(station=station,test_date='2024-02-18',**classification_metrics(frame)))
            sources[str(n1path)]=sha(n1path)
            hpath=n1path.with_name('hiperparametros_xgboost.json');h=json.loads(hpath.read_text(encoding='utf-8'))
            assert not h['teste_usado_na_selecao'] and not h['revisao_manual_aplicada']
            classifier_parameters.append(dict(station=station,threshold=h['limiar_probabilidade'],
                sampling=h['amostragem_treino']['balanceamento'],
                **{k:h['parametros'].get(k) for k in ['n_estimators','max_depth','learning_rate','min_child_weight','reg_lambda','scale_pos_weight']}))
            sources[str(hpath)]=sha(hpath)
        if not featurepath.exists() or not seriespath.exists():
            status.append(dict(station=station,final_configurations=0,expected=24,status='preparacao_pendente'));continue
        features=pd.read_csv(featurepath,usecols=['datetime','svid','S4','VTEC'],parse_dates=['datetime'])
        observed=features.groupby('datetime').VTEC.median().reindex(GRID)
        if not observed.notna().all(): raise ValueError(f'{station}: dia observado incompleto')
        predictions={};station_metrics=[]
        paths=list((OUT/'estacoes'/station/'lstm_integral'/'finais').glob('*/*/resumo.json'))
        paths+=list((OUT/'estacoes'/station/'convlstm'/'finais').glob('*/*/selecionado/resumo.json'))
        for path in paths:
            summary=json.loads(path.read_text(encoding='utf-8'))
            assert summary['phase']=='final' and summary['all_development_refit']
            for artifact,digest in summary['artifacts'].items(): check_hash(artifact,digest)
            q=validate_predictions(pd.read_csv(path.parent/'predicoes.csv',parse_dates=['datetime']))
            np.testing.assert_allclose(q.observado_comum,observed,rtol=1e-9,atol=1e-8)
            model=summary['model'];variant=summary['variant'];assert (model,variant) not in predictions
            training=OUT/'estacoes'/station/('lstm_integral' if int(model[1:])<=8 else 'convlstm')
            taskname=f'{model}_{variant}.json' if int(model[1:])<=8 else f'{model}_{variant}_selecionado_final.json'
            taskpath=training/'tarefas'/taskname;check_hash(taskpath,summary['task_sha256'])
            task=json.loads(taskpath.read_text(encoding='utf-8'))
            assert task['phase']=='final' and task['test_date']=='2024-02-18'
            assert pd.Timestamp(task['train_end'])==pd.Timestamp('2024-02-16 23:30')
            for artifact,digest in task['hashes'].items(): check_hash(artifact,digest)
            histories=list(path.parent.glob('*/historico.json'))
            assert len(histories)==(48 if int(model[1:]) in (5,6,7,8,11,12) else 1)
            for historypath in histories:
                history=json.loads(historypath.read_text(encoding='utf-8'))
                partition=history['partition'];assert pd.Timestamp(partition['fit']['end']) < pd.Timestamp('2024-02-17')
                assert pd.Timestamp(partition['fit']['start']) >= pd.Timestamp('2023-12-01')
                assert partition['early']['n']==0 and partition['selection']['n']==0
                assert history['reload_weights_identical'] and history['reload_same_batch']
                assert history['epochs_executed']==history['best_epoch']
                partitions.append(dict(station=station,model=model,variant=variant,slot=history['slot'],
                    fit_start=partition['fit']['start'],fit_end=partition['fit']['end'],fit_n=partition['fit']['n'],
                    test_n=partition['test']['n'],epochs=history['epochs_executed'],
                    reload_max_difference=history['reload_max_difference']))
            settings=history['spec'];epochs=list(summary['selected_epochs'].values())
            hyperparameters.append(dict(station=station,model=model,variant=variant,**settings,
                                        epochs_min=min(epochs),epochs_max=max(epochs),networks=len(histories)))
            predictions[model,variant]=q
            row=dict(station=station,model=model,variant=variant,test_date='2024-02-18',
                     **scores(observed,q.predito),source=str(path.relative_to(ROOT)))
            station_metrics.append(row);metrics.append(row);count+=1
            all_predictions.append(q.assign(station=station,model=model,variant=variant))
            sources[str(path)]=sha(path)
        status.append(dict(station=station,final_configurations=count,expected=24,
                           observed_epochs=int(observed.notna().sum()),status='concluido' if count==24 else 'treinando'))
        series=pd.read_csv(seriespath,parse_dates=['datetime'])
        if n1path.exists():
            dayrows=features.loc[features.datetime.isin(GRID)].merge(
                frame[['target_datetime','svid','sinalizar_filtro','classe_future']].rename(columns={'target_datetime':'datetime'}),
                on=['datetime','svid'],how='left',validate='one_to_one')
            flags=dayrows.sinalizar_filtro.astype(str).str.lower().eq('true')
            valid=dayrows.VTEC.notna();removed=flags&valid
            before=dayrows.groupby('datetime').VTEC.median().reindex(GRID)
            after=dayrows.VTEC.mask(flags).groupby(dayrows.datetime).median().reindex(GRID)
            np.testing.assert_allclose(before,observed,rtol=1e-9,atol=1e-8)
            saved=series.set_index('datetime').reindex(GRID)
            np.testing.assert_allclose(saved.VTEC_original,observed,rtol=1e-9,atol=1e-8)
            np.testing.assert_allclose(saved.VTEC_filtered[after.notna()],after.dropna(),rtol=1e-9,atol=1e-8)
            difference=(saved.VTEC_filtered-saved.VTEC_original).abs()
            filtering.append(dict(station=station,observed_epochs=48,observed_satellite_epochs=int(valid.sum()),
                removed_satellite_epochs=int(removed.sum()),removed_percent=100*float(removed.sum())/int(valid.sum()),
                removed_true_positives=int((removed&dayrows.classe_future.eq(1)).sum()),
                removed_false_positives=int((removed&dayrows.classe_future.eq(0)).sum()),
                epochs_without_satellites_after_filter=int(after.isna().sum()),
                epochs_changed=int(difference.gt(1e-8).sum()),mean_absolute_change_TECU=float(difference.mean()),
                max_absolute_change_TECU=float(difference.max())))
        raw=raw_day(processed/f'{station}_clean.csv')
        station_figures(station,raw,features,series,predictions,station_metrics)
        daydir=ARTICLE/'dados_dos_graficos'/station;daydir.mkdir(parents=True,exist_ok=True)
        raw.to_csv(daydir/'satelites_nativos_18fev.csv',index=False)
        series.loc[series.datetime.isin(GRID)].to_csv(daydir/'series_30min_18fev.csv',index=False)
        observed.rename('VTEC_observado').to_csv(daydir/'referencia_observada.csv',index_label='datetime')
        sources[str(featurepath)]=sha(featurepath);sources[str(seriespath)]=sha(seriespath)
        del raw,features,series
        gc.collect()
    pd.DataFrame(status).to_csv(tables/'status_estacoes.csv',index=False)
    n1=pd.DataFrame(classifiers);n1.to_csv(tables/'nucleo1_18fev.csv',index=False)
    if classifiers: readable_table(n1[['station','precision','recall','f1','f2','TP','FP','FN','positivos']],tables/'nucleo1_18fev')
    pd.DataFrame(classifier_parameters).to_csv(tables/'hiperparametros_nucleo1.csv',index=False)
    pd.DataFrame(filtering).to_csv(tables/'efeito_filtragem_nos_dados_18fev.csv',index=False)
    pd.DataFrame(hyperparameters).to_csv(tables/'hiperparametros_nucleo2.csv',index=False)
    pd.DataFrame(partitions).to_csv(tables/'particoes_por_rede.csv',index=False)
    n2=pd.DataFrame(metrics,columns=['station','model','variant','test_date','n','RMSE','MAE','R2','source'])
    n2.to_csv(tables/'nucleo2_18fev_referencia_comum.csv',index=False)
    if all_predictions: pd.concat(all_predictions,ignore_index=True).to_csv(tables/'predicoes_finais_18fev.csv',index=False)
    if metrics:
        readable_table(n2[['station','model','variant','RMSE','MAE','R2']],tables/'nucleo2_18fev_referencia_comum')
        comparison=n2.pivot(index=['station','model'],columns='variant',values=['RMSE','MAE','R2'])
        comparison.columns=[f'{metric}_{variant}' for metric,variant in comparison.columns]
        comparison=comparison.reset_index()
        comparison.to_csv(tables/'comparacao_modelos_filtragens.csv',index=False)
        paired=n2.pivot(index=['station','model'],columns='variant',values='RMSE').reset_index()
        if 'filtrado' in paired and 'original' in paired:
            paired['delta_RMSE_filtrado_menos_original']=paired.filtrado-paired.original
            paired['ganho_percentual']=100*(paired.original-paired.filtrado)/paired.original
            paired.to_csv(tables/'efeito_filtragem_rmse.csv',index=False)
            readable_table(paired[['station','model','original','filtrado','ganho_percentual']],tables/'efeito_filtragem_rmse')
        best=n2.loc[n2.groupby('station').RMSE.idxmin()]
        best.to_csv(tables/'menor_rmse_retrospectivo.csv',index=False)
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for station,ax in zip(STATIONS,axes.flat):
            d=n2.loc[n2.station.eq(station)].pivot(index='model',columns='variant',values='RMSE').reindex([f'M{i}' for i in range(1,13)])
            for index,variant in enumerate(COLORS):
                if variant in d: ax.bar(np.arange(12)+(index-.5)*.36,d[variant],width=.34,color=COLORS[variant],label=LABELS[variant])
            ax.set_xticks(np.arange(12),[f'M{i}' for i in range(1,13)],rotation=45);ax.set_title(station,loc='left',fontweight='bold');ax.set_ylabel('RMSE (TECU)')
            if ax.get_legend_handles_labels()[0]: ax.legend()
        fig.suptitle('18/02/2024 | Mesmo VTEC observado para original e filtrado',fontsize=15)
        export(fig,ARTICLE/'figuras'/'comparacao_rmse')
    if classifiers:
        fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
        for station,ax in zip(STATIONS,axes.flat):
            row=next((r for r in classifiers if r['station']==station),None)
            if row is None: ax.set_title(station+' — pendente');ax.axis('off');continue
            matrix=np.array([[row['TN'],row['FP']],[row['FN'],row['TP']]])
            normalized=matrix/np.maximum(matrix.sum(axis=1,keepdims=True),1)
            ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
            for i in range(2):
                for j in range(2): ax.text(j,i,f'{matrix[i,j]}\n{normalized[i,j]:.1%}',ha='center',va='center',color='white' if normalized[i,j]>.55 else '#222222',fontsize=13)
            ax.set_xticks([0,1],['Não severo','Severo']);ax.set_yticks([0,1],['Não severo','Severo']);ax.grid(False)
            ax.set_xlabel('Predito');ax.set_ylabel('Observado');ax.set_title(f"{station} | P={row['precision']:.1%} · R={row['recall']:.1%}",loc='left')
        fig.suptitle('Núcleo 1 — 18/02/2024 | Contagens e proporções por classe observada',fontsize=13)
        export(fig,ARTICLE/'figuras'/'matrizes_confusao_nucleo1')
    gather_documents()
    from figura_hiperparametros import main as generate_hyperparameter_figures
    generate_hyperparameter_figures()
    complete=len(metrics)==96 and len(classifiers)==4 and all(r['final_configurations']==24 for r in status)
    write(ARTICLE/'auditoria.json',dict(status='completo' if complete else 'parcial',test_date='2024-02-18',
          final_configurations=len(metrics),expected_configurations=96,predictions=48*len(metrics),
          classifiers=len(classifiers),sources_sha256=sources,updated_at=datetime.now(timezone.utc).isoformat()))
    text='# Material para artigo — 18/02/2024\n\n'
    text+=('Conjunto completo e verificado.' if complete else '**Material parcial: os treinamentos ainda não terminaram.**')+'\n\n'
    text+='Tabelas em `tabelas/`; figuras em PNG, PDF e SVG em `figuras/`; dados das figuras em `dados_dos_graficos/`; documentação e fontes históricas em `documentacao/`.\n\n'
    text+='[Figuras dos hiperparâmetros, escolha do limiar e detalhamento M1–M12](figuras/hiperparametros/README.md).\n\n'
    text+='Só os reajustes finais entram na tabela principal. Menor RMSE é um ranking retrospectivo neste dia, não seleção independente para implantação. Ganho positivo indica menor RMSE após filtragem. Consulte o protocolo para partições, interpolação e horizontes diferentes das redes piecewise.\n\n'
    text+='As matrizes de confusão precedem qualquer revisão manual. Recall é priorizado por F2 no contexto de detecção de cintilação; precisão baixa e falsos positivos permanecem reportados.\n'
    (ARTICLE/'README.md').write_text(text,encoding='utf-8')
    captions='''# Legendas e leitura das figuras

- **S4 e VTEC do dia:** pontos cinza são observações nativas de todos os satélites que passaram pelo controle de qualidade. As séries de 30 minutos resultam da agregação por satélite e depois da mediana entre satélites. A linha de S4 não é um classificador da estação: o Núcleo 1 classifica cada satélite/época. Exceder 0,5 em um ponto nativo não equivale necessariamente a exceder o limiar no alvo reamostrado. Verde pontilhado identifica o VTEC reamostrado após filtragem. Azul e laranja identificam as previsões dos ramos original e filtrado.
- **Comparação temporal M1–M12:** cada painel compara os dois ramos da mesma arquitetura nas mesmas 48 épocas observadas. Resultados ausentes aparecem como pendentes, nunca como erro zero.
- **Dispersão observado × predito:** cada ponto corresponde a uma época de meia hora; a diagonal representa igualdade. Eixos compartilhados permitem perceber viés e perda da amplitude diária.
- **RMSE:** erro contra a referência observada comum. Comparar os dois ramos dentro de cada arquitetura. Diferenças entre LSTM e ConvLSTM também incluem a definição do alvo de treinamento. Ganho percentual positivo na tabela indica redução de RMSE pela filtragem.
- **Matrizes do Núcleo 1:** números absolutos e proporções normalizadas por classe observada. As métricas precedem qualquer revisão manual de falsos positivos. Precisão e recall não são intercambiáveis.

O título que destaca o menor RMSE descreve o ranking retrospectivo neste dia, entre modelos já concluídos. Não representa seleção independente para implantação. O protocolo usa um único dia principal e não permite afirmar ganho geral em outras datas.
'''
    (ARTICLE/'LEGENDAS.md').write_text(captions,encoding='utf-8')
    print(json.dumps({'complete':complete,'final_configurations':len(metrics),'classifiers':len(classifiers)},ensure_ascii=False),flush=True)
    if args.require_complete and not complete: raise SystemExit('Material ainda incompleto')


if __name__=='__main__': main()
