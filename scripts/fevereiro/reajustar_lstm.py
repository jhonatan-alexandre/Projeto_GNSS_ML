"""Reajuste M1–M8 em todo desenvolvimento, após seleção no fluxo original."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
from core_convlstm import ROOT, OUT, sha, write, spec

CODE = Path(__file__).resolve().parent


def run(station, observed):
    base = ROOT/'experimentos'/f'{station.lower()}_comum_18fev2024'
    cfg = ROOT/'configs'/f'xie_{station.lower()}_comum_18fev2024.json'
    audit = subprocess.run([sys.executable, '-X','utf8',str(ROOT/'scripts/auditar_retomada_fev.py'),
                            '--config',str(cfg),'--output-root',str(base),'--station',station],
                           cwd=ROOT, capture_output=True, text=True, encoding='utf-8', check=True)
    checked = json.loads(audit.stdout)
    assert len(checked['checkpoints']) == 17
    assert all(not c['differences'] and c['artifacts_valid'] for c in checked['checkpoints'])
    assert not checked['loss_masking'], 'Reajuste deste protocolo não usa loss masking'
    prior=base/'resultados'/station/'nucleo2'
    dataset=base/'dados_processados'/station/f'{station}_dataset_regressao.csv'
    directory=OUT/'estacoes'/station/'lstm_integral'
    sources=[CODE/'core_convlstm.py',CODE/'worker_convlstm.py',Path(__file__).resolve(),cfg,dataset,observed,
             ROOT/'scripts/10_train_lstm.py',prior/'hiperparametros_lstm.json']
    selected={}
    for checkpoint in sorted(prior.glob('checkpoint_m*.json')):
        item=json.loads(checkpoint.read_text(encoding='utf-8'))
        meta=item['metricas']; model=meta['config_id']; name=meta['modelo']
        variant='filtrado' if meta['filtrado'] else 'original'
        historypath=prior/f'historico_{name}.json'
        history=json.loads(historypath.read_text(encoding='utf-8'))
        histories=history if meta['organizacao']=='piecewise' else {'all':history}
        epochs={str(slot):int(h['melhor_epoca']) for slot,h in histories.items()}
        assert all(1 <= value <= 50 for value in epochs.values())
        assert all(h['validacao_fim'][:10] <= '2024-02-16' for h in histories.values())
        selected[f'{model}_{variant}']={'spec':spec(model,float(meta['learning_rate'])), 'epochs':epochs,
                                      'history_sha256':sha(historypath), 'checkpoint_sha256':sha(checkpoint)}
        sources.extend([checkpoint,historypath])
    assert len(selected)==16
    hashes={str(p):sha(p) for p in sources}
    frozen=directory/'selecao_congelada.json'
    protocol={'test_date':'2024-02-18', 'train_end':'2024-02-16 23:30:00', 'selected':selected,
              'input_hashes':hashes, 'test_used_for_selection':False,
              'training_target':'próprio ramo; apenas épocas com observação real comum',
              'evaluation_target':'VTEC observado comum', 'full_development_refit':True}
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding='utf-8'))==protocol
    else:
        write(frozen,protocol)
    hashes={**hashes,str(frozen):sha(frozen)}
    metrics=[]
    for key,choice in selected.items():
        model,variant=key.split('_'); dest=directory/'finais'/model/variant
        task=dict(station=station,model=model,variant=variant,spec=choice['spec'],phase='final',
                  dataset=str(dataset),observed=str(observed),output=str(dest),hashes=hashes,
                  fixed_epochs=choice['epochs'],train_end='2024-02-16 23:30',test_date='2024-02-18')
        taskpath=directory/'tarefas'/f'{key}.json'
        if taskpath.exists():
            assert json.loads(taskpath.read_text(encoding='utf-8'))==task
        else:
            write(taskpath,task)
        summary=dest/'resumo.json'
        if not summary.exists():
            write(directory/'status.json',dict(status='reajustando',station=station,model=model,variant=variant))
            env=os.environ.copy()
            env.update(TF_ENABLE_ONEDNN_OPTS='0',TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            log=directory/'logs'/f'{key}.log'; log.parent.mkdir(exist_ok=True)
            print('REAJUSTE INTEGRAL',station,model,variant,flush=True)
            with log.open('a',encoding='utf-8') as stream:
                subprocess.run([sys.executable,'-X','utf8','-u',str(CODE/'worker_convlstm.py'),'--task',str(taskpath)],
                               cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        result=json.loads(summary.read_text(encoding='utf-8'))
        assert result['task_sha256']==sha(taskpath)
        for path,digest in result['artifacts'].items(): assert sha(path)==digest
        metrics.append(dict(station=station,model=model,variant=variant,**result['metrics_common']))
    pd.DataFrame(metrics).to_csv(directory/'metricas_comuns.csv',index=False)
    write(directory/'status.json',dict(status='concluido',station=station,configurations=16,predictions=768,networks=392))
