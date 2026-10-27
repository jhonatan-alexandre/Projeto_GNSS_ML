"""Compara cronômetros persistidos; não infere duração a partir de mtime."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from core_convlstm import ROOT, OUT, sha

DEST = OUT/'artigo/figuras/tempos_treinamento'
STATIONS = ('FRTZ','PALM','POAL','PRU2')
MODELS = [f'M{i}' for i in range(1,13)]
PIECEWISE = {5,6,7,8,11,12}
BLUE, ORANGE, INK = '#176B96','#CF7029','#203448'


def log_flags(text):
    flags=[]
    if len(re.findall(r'^Epoch 1/\d+',text,re.M))>1: flags.append('epocas_reiniciadas')
    if text.count('reset_default_graph is deprecated')>1: flags.append('multiplas_inicializacoes')
    if 'Traceback (most recent call last)' in text: flags.append('erro_em_tentativa_anterior')
    if 'REUSO VERIFICADO' in text or 'Recuperando slot' in text: flags.append('reuso_de_slots')
    return flags


def save(fig,name):
    for extension in ('png','pdf','svg'):
        fig.savefig(DEST/f'{name}.{extension}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
        'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'axes.titlecolor':INK,
        'axes.spines.top':False,'axes.spines.right':False})
    sources={};rows=[];unmeasured=[];log_inventory=[]
    def read(path):
        sources[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def inspect_log(path):
        if not path.exists(): return ['log_nao_localizado']
        blob=path.read_bytes();text=blob.decode('utf-8',errors='replace')
        digest=hashlib.sha256(blob).hexdigest()
        snap=DEST/'logs_consultados'/f'{digest[:12]}_{path.name}'
        snap.parent.mkdir(exist_ok=True)
        if not snap.exists():snap.write_bytes(blob)
        log_inventory.append(dict(source=str(path.relative_to(ROOT)),snapshot=str(snap.relative_to(DEST)),sha256=digest))
        return log_flags(text)
    for station in STATIONS:
        prior=ROOT/'experimentos'/f'{station.lower()}_comum_18fev2024'/'resultados'/station/'nucleo2'
        stage_log=OUT/'logs'/f'{station}_10_train_lstm.log'
        stage_text=stage_log.read_text(encoding='utf-8',errors='replace') if stage_log.exists() else ''
        inspect_log(stage_log)
        for path in sorted(prior.glob('checkpoint_m*.json')):
            item=read(path);r=item['metricas'];model=r['config_id']
            start=re.search(r'^Processo \d+: '+re.escape(r['modelo'])+r'\s*$',stage_text,re.M)
            flags=[]
            if start:
                tail=stage_text[start.end():]
                stop=re.search(r'^Processo \d+: ',tail,re.M)
                flags=log_flags(tail[:stop.start()] if stop else tail)
            else:flags=['log_detalhado_nao_localizado']
            duration=float(r['duracao_segundos']);assert np.isfinite(duration) and duration>=0
            rows.append(dict(station=station,model=model,variant='filtrado' if r['filtrado'] else 'original',
                phase='desenvolvimento',candidate='selecao_epocas_lstm',duration_seconds=duration,
                networks=48 if int(model[1:]) in PIECEWISE else 1,epochs=r['epocas_executadas'],
                flags=';'.join(flags),source=str(path.relative_to(ROOT))))
        unmeasured.append(dict(station=station,scope='busca_taxa_compartilhada_M1_M8',
             reason='Sem cronômetro por candidata no JSON; não distribuído artificialmente entre os modelos.'))
        for number in range(1,13):
            model=f'M{number}';family='lstm_integral' if number<=8 else 'convlstm'
            directory=OUT/'estacoes'/station/family
            for variant in ('original','filtrado'):
                dest=directory/'finais'/model/variant
                if number>=9: dest=dest/'selecionado'
                path=dest/'resumo.json'
                if not path.exists():
                    unmeasured.append(dict(station=station,scope=f'{model}_{variant}_final',reason='Resumo final ainda ausente.'))
                    continue
                r=read(path);assert r['phase']=='final' and r['all_development_refit']
                name=f'{model}_{variant}'+('' if number<=8 else '_selecionado_final')
                task=directory/'tarefas'/(f'{model}_{variant}.json' if number<=8 else name+'.json')
                assert sha(task)==r['task_sha256']
                taskdata=read(task);assert taskdata['test_date']=='2024-02-18'
                flags=inspect_log(directory/'logs'/f'{name}.log')
                duration=float(r['duration_seconds']);assert np.isfinite(duration) and duration>=0
                networks=len(r['selected_epochs']);assert networks==(48 if number in PIECEWISE else 1)
                rows.append(dict(station=station,model=model,variant=variant,phase='final',candidate='selecionado',
                    duration_seconds=duration,networks=networks,epochs=sum(r['selected_epochs'].values()),
                    flags=';'.join(flags),source=str(path.relative_to(ROOT))))
            if number>=9:
                for variant in ('original','filtrado'):
                    for candidate in ('c0','c1'):
                        label=candidate+('_validacao_anterior' if station=='POAL' and number>=11 else '')
                        path=directory/'busca'/model/variant/label/'resumo.json'
                        if not path.exists():
                            unmeasured.append(dict(station=station,scope=f'{model}_{variant}_{label}_busca',reason='Candidata sem resumo concluído.'))
                            continue
                        r=read(path);assert r['phase']=='search'
                        task=directory/'tarefas'/f'{model}_{variant}_{label}_search.json'
                        assert sha(task)==r['task_sha256'];read(task)
                        flags=inspect_log(directory/'logs'/f'{model}_{variant}_{label}_search.log')
                        duration=float(r['duration_seconds']);assert np.isfinite(duration) and duration>=0
                        rows.append(dict(station=station,model=model,variant=variant,phase='desenvolvimento',candidate=label,
                            duration_seconds=duration,networks=len(r['selected_epochs']),epochs=None,
                            flags=';'.join(flags),source=str(path.relative_to(ROOT))))
        # Tentativa de POAL interrompida: há evidência, mas nenhum cronômetro final.
        if station=='POAL':
            failed=OUT/'estacoes/POAL/convlstm/logs/M11_original_c0_search.log'
            flags=inspect_log(failed)
            if 'erro_em_tentativa_anterior' in flags:
                unmeasured.append(dict(station=station,scope='M11_original_c0_anterior',
                    reason='Tentativa interrompida; 14 slots reutilizados, sem duração completa recuperável no log.'))
    data=pd.DataFrame(rows)
    assert not data.duplicated(['station','model','variant','phase','candidate']).any()
    data.to_csv(DEST/'tempos_por_etapa_ramo_candidata.csv',index=False)
    final=data.loc[data.phase.eq('final')].copy()
    grouped=final.groupby(['station','model']).agg(seconds=('duration_seconds','sum'),
        branches=('variant','nunique'),networks=('networks','sum'),epochs=('epochs','sum'),
        flagged=('flags',lambda s:any(bool(x) for x in s))).reset_index()
    grouped['minutes']=grouped.seconds/60
    grouped['seconds_per_network']=grouped.seconds/grouped.networks
    grouped['complete']=grouped.branches.eq(2)
    grouped.to_csv(DEST/'tempos_finais_por_modelo_estacao.csv',index=False)
    grid=grouped.set_index(['model','station']).reindex(pd.MultiIndex.from_product([MODELS,STATIONS],names=['model','station']))
    matrix=grid.minutes.to_numpy().reshape(12,4)
    average=grid.seconds_per_network.to_numpy().reshape(12,4)
    valid=grid.complete.fillna(False).to_numpy().reshape(12,4)
    flagged=grid.flagged.fillna(False).to_numpy().reshape(12,4)
    matrix=np.where(valid,matrix,np.nan);average=np.where(valid,average,np.nan)
    fig,axes=plt.subplots(1,2,figsize=(16.5,10.5),gridspec_kw={'wspace':.35})
    fig.subplots_adjust(left=.09,right=.94,bottom=.19,top=.84)
    for ax,values,title,unit in zip(axes,[matrix,average],
            ['A  |  Tempo total por modelo','B  |  Média de tempo por rede'],
            ['Minutos · original + filtrado','Segundos por rede · média dos dois ramos']):
        finite=values[np.isfinite(values)];assert len(finite)>0 and finite.min()>0
        cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#ECEFF2')
        norm=LogNorm(vmin=min(.1,float(finite.min())),vmax=float(finite.max())*1.05)
        if 'Média' in title:norm=LogNorm(vmin=float(finite.min())*.85,vmax=float(finite.max())*1.05)
        im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,norm=norm,aspect='auto')
        ax.set_xticks(range(4),STATIONS,fontsize=12);ax.xaxis.tick_top()
        ax.set_yticks(range(12),[f'M{i}  ·  '+('96 redes' if i in PIECEWISE else '2 redes') for i in range(1,13)],fontsize=10)
        ax.set_title(title,pad=40,fontsize=14,weight='bold')
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,12,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False)
        for y in range(12):
            for x in range(4):
                value=values[y,x]
                label='Pendente' if not np.isfinite(value) else f'{value:.2f}'.replace('.',',')+(' †' if flagged[y,x] else '')
                color='white' if np.isfinite(value) and norm(value)>.65 else INK
                ax.text(x,y,label,ha='center',va='center',fontsize=10.5,weight='bold',color=color)
        cb=fig.colorbar(im,ax=ax,fraction=.042,pad=.035);cb.set_label(unit+' (cor em escala log)',fontsize=9)
    fig.text(.055,.956,'TEMPO REGISTRADO DE TREINAMENTO FINAL',fontsize=22,weight='bold',color=INK)
    fig.text(.055,.919,'M1–M12 por estação • Soma dos ramos original e filtrado • Ajuste até 16/02/2024; teste em 18/02/2024',fontsize=11,color=INK)
    totals=final.groupby('station').duration_seconds.sum()/3600
    fig.text(.055,.139,'Somas dos tempos finais registrados:  '+'   |   '.join(f'{s}: {totals.get(s,0):.2f} h'.replace('.',',') for s in STATIONS),fontsize=11,weight='bold')
    fig.text(.055,.105,'† Retomada/interrupção detectada no log: o cronômetro salvo não cobre necessariamente todas as tentativas anteriores.',fontsize=10)
    fig.text(.055,.077,'O worker cronometra preparação da rede, ajuste, previsão, salvamento e verificação. Não é uma medição exclusiva das épocas de treino.',fontsize=10)
    fig.text(.055,.049,'Execuções concorrentes e números de épocas distintos: comparação descritiva desta execução, não benchmark isolado de arquiteturas.',fontsize=10)
    fig.text(.055,.023,f'{len(final)}/96 configurações finais com cronômetro • A média por rede divide o tempo total por 2 (convencional) ou 96 (piecewise).',fontsize=9,color='#526577')
    save(fig,'01_tempo_final_por_modelo_estacao')

    fig,axes=plt.subplots(2,2,figsize=(16,10.5),sharex=True,sharey=True)
    fig.subplots_adjust(left=.065,right=.975,top=.85,bottom=.19,hspace=.29,wspace=.14)
    allmax=0
    for station,ax in zip(STATIONS,axes.flat):
        subset=data.loc[data.station.eq(station)]
        pre=subset.loc[subset.phase.eq('desenvolvimento')].groupby('model').duration_seconds.sum().reindex(MODELS)
        fin=subset.loc[subset.phase.eq('final')].groupby('model').duration_seconds.sum().reindex(MODELS)
        counts=subset.loc[subset.phase.eq('final')].groupby('model').size().reindex(MODELS,fill_value=0)
        x=np.arange(12);a=pre.fillna(0).to_numpy()/60;b=fin.fillna(0).to_numpy()/60
        ax.bar(x,a,color=BLUE,label='Desenvolvimento / seleção registrados',width=.7)
        bars=ax.bar(x,b,bottom=a,color=ORANGE,label='Reajuste final registrado',width=.7)
        allmax=max(allmax,float(np.max(a+b)))
        for index,bar in enumerate(bars):
            if counts.iloc[index]<2:bar.set_hatch('//')
        ax.set_title(station,loc='left',weight='bold',fontsize=14)
        ax.set_xticks(x,MODELS);ax.tick_params(axis='x',labelbottom=True)
        ax.set_ylabel('Tempo registrado (minutos)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.legend(fontsize=8.5,loc='upper left')
    for ax in axes.flat:ax.set_ylim(0,allmax*1.18)
    fig.text(.055,.955,'TEMPO REGISTRADO POR ETAPA DO EXPERIMENTO',fontsize=21,weight='bold')
    fig.text(.055,.918,'Cada barra soma original + filtrado. Escala linear comum entre estações; fases mantidas separadas.',fontsize=11)
    fig.text(.055,.137,'M1–M8, azul: ajuste com validação e seleção de épocas. M9–M12, azul: quatro ajustes candidatos (2 configurações × 2 ramos).',fontsize=10)
    fig.text(.055,.106,'A busca da taxa compartilhada de M1–M8, tentativas sem cronômetro, preparação dos dados e Núcleo 1 não entram nestas somas.',fontsize=10)
    fig.text(.055,.075,'POAL M11/M12: utiliza a busca com a janela autorizada; slots reutilizados conservam o trabalho anterior não cronometrado integralmente.',fontsize=10)
    fig.text(.055,.044,'Os totais são o tempo medido nas execuções registradas, não o custo completo de todo o histórico nem a duração global da sessão paralela.',fontsize=10)
    save(fig,'02_tempo_desenvolvimento_e_final')

    pd.DataFrame(unmeasured).to_csv(DEST/'tempos_nao_recuperaveis.csv',index=False)
    sources[str(Path(__file__).resolve().relative_to(ROOT))]=sha(__file__)
    metadata=dict(created_at=datetime.now(timezone.utc).isoformat(),final_configurations=len(final),
        sources_sha256=sources,logs=log_inventory,unmeasured=unmeasured,
        measurement='tempo_monotonico_do_worker_na_execucao_que_produziu_o_resumo',
        total_historical_training_time_reconstructed=False)
    (DEST/'auditoria_tempos.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    guide='''# Tempo de treinamento por modelo e estação

- **Comparação principal — reajuste final:** [PNG](01_tempo_final_por_modelo_estacao.png) · [PDF](01_tempo_final_por_modelo_estacao.pdf) · [SVG](01_tempo_final_por_modelo_estacao.svg).
- **Desenvolvimento e reajuste separados:** [PNG](02_tempo_desenvolvimento_e_final.png) · [PDF](02_tempo_desenvolvimento_e_final.pdf) · [SVG](02_tempo_desenvolvimento_e_final.svg).
- [Valores finais por modelo/estação](tempos_finais_por_modelo_estacao.csv), [todos os registros por ramo e etapa](tempos_por_etapa_ramo_candidata.csv), [lacunas de medição](tempos_nao_recuperaveis.csv), [fontes e logs preservados](auditoria_tempos.json).

## Leitura correta

A primeira figura soma os dois ramos de cada arquitetura. Cada configuração
convencional tem uma rede, e cada configuração piecewise possui 48 redes, uma
por horário: são duas ou 96 redes ao somar original e filtrado. O painel da
direita divide o mesmo tempo pelo número de redes; não mede cada rede individualmente.

Os tempos vêm de `duration_seconds` dos resumos finais e dos candidatos ConvLSTM,
e de `duracao_segundos` nos checkpoints de desenvolvimento M1–M8. São cronômetros
do worker: incluem preparação de sequências/redes, ajuste, previsão e gravação;
nos workers finais incluem também recarga/verificação e limpeza entre redes.
Não incluem necessariamente importações, leitura inicial dos dados ou o
tempo de espera entre processos. Nenhuma duração foi inferida de datas de
modificação dos arquivos. As estações executaram em paralelo e disputaram
recursos; tempos somados não equivalem à duração global observada no relógio.

## Retomadas e dados ausentes

O cronômetro de um resumo corresponde à chamada que o produziu. Uma chamada
retomada pode carregar redes já prontas ou repetir uma tentativa interrompida.
O M9 original de FRTZ possui reinício explícito nos logs e está marcado com †.
O tempo da tentativa anterior não é recuperável integralmente; os segundos
arredondados impressos por época não foram somados como se fossem um cronômetro
equivalente ao do worker.

Para POAL M11/M12, usa-se apenas a busca da versão autorizada. O candidato
M11 original C0 reaproveita 14 slots da tentativa interrompida, cujo tempo
completo não foi registrado. Não há duplicação da busca antiga e da adaptada.
A busca da taxa de aprendizado compartilhada por M1–M8 não dispõe de duração
por candidata nos JSON e não foi artificialmente atribuída a um modelo.
Lacunas de medição permanecem explícitas; não representam custo zero.

## Legendas sugeridas

**Figura 1:** Tempo de parede registrado pelos workers de reajuste final de
M1–M12, por estação, somando os ramos original e filtrado. À esquerda, tempo
total em minutos; à direita, média de segundos por rede. A intensidade da cor
usa escala logarítmica. Modelos piecewise incluem 48 redes por ramo. O símbolo
† identifica retomada detectada, cuja medição não cobre necessariamente todas
as tentativas. Os tempos refletem épocas e condições de execução distintas.

**Figura 2:** Tempos registrados no desenvolvimento e no reajuste final,
separados por arquitetura e estação. Para M1–M8, desenvolvimento corresponde
ao ajuste com validação e seleção de épocas; para M9–M12, à busca entre duas
configurações em cada ramo. A figura não representa o custo computacional total
de todas as tentativas históricas, nem uma avaliação isolada de desempenho.

Reprodução: `.venv/Scripts/python.exe -X utf8 scripts/fevereiro/figura_tempos.py`.
'''
    (DEST/'README.md').write_text(guide,encoding='utf-8')
    print(json.dumps(dict(final_configurations=len(final),records=len(data),hours_final=totals.to_dict(),
                         output=str(DEST)),ensure_ascii=False))


if __name__=='__main__':main()
