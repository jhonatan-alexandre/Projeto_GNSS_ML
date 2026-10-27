"""Treina as quatro arquiteturas ConvLSTM com seleção temporal compartilhada."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pandas as pd
from core_convlstm import ROOT, OUT, sha, write, spec
from reajustar_lstm import run as refit_lstm
from lock import acquire

CODE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stations', nargs='+', default=['FRTZ', 'PALM', 'POAL', 'PRU2'])
    args = parser.parse_args()
    scientific = {str(CODE/n): sha(CODE/n) for n in ['core_convlstm.py', 'worker_convlstm.py', 'treinar_convlstm.py']}
    for station in args.stations:
        station_lock = acquire(OUT/'estacoes'/station/'treinamento.lock')
        base = ROOT/'experimentos'/f'{station.lower()}_comum_18fev2024'
        processed = base/'dados_processados'/station
        dataset = processed/f'{station}_dataset_regressao.csv'
        features = processed/f'{station}_features.csv'
        cfg = ROOT/'configs'/f'xie_{station.lower()}_comum_18fev2024.json'
        config = json.loads(cfg.read_text(encoding='utf-8'))
        assert config['period']['train_end'] == '2024-02-16 23:59:59'
        assert config['period']['test_end'] == '2024-02-18 23:59:59'
        directory = OUT/'estacoes'/station/'convlstm'
        directory.mkdir(parents=True, exist_ok=True)
        # Referência anterior à filtragem e à interpolação, comum aos dois ramos.
        obs = directory/'vtec_observado_30min.csv'
        raw = pd.read_csv(features, usecols=['datetime','VTEC'], parse_dates=['datetime'])
        reference = raw.groupby('datetime').VTEC.median().rename('VTEC_observado').reset_index()
        test = reference.loc[reference.datetime.between('2024-02-18', '2024-02-18 23:30')]
        assert len(test) == 48 and test.VTEC_observado.notna().all(), 'Teste sem 48 épocas observadas'
        if obs.exists():
            pd.testing.assert_frame_equal(pd.read_csv(obs, parse_dates=['datetime']), reference, check_exact=False)
        else:
            reference.to_csv(obs, index=False)
        refit_lstm(station, obs)
        hashes = {**scientific, **{str(x): sha(x) for x in [dataset, features, obs, cfg]}}
        protocol = {'test_date':'2024-02-18', 'final_train_end':'2024-02-16 23:30:00',
                    'all_observed_development_refit':True, 'input_hashes':hashes,
                    'architectures':{'M9':'multivariado convencional', 'M10':'univariado convencional',
                                     'M11':'univariado piecewise', 'M12':'multivariado piecewise'},
                    'selection':'early stopping 03–09/02; seleção 10–16/02; média do RMSE comum dos ramos',
                    'target':'VTEC observado comum; ajuste final com todos os alvos históricos observados elegíveis',
                    'limitations':['entradas interpoladas pelo fluxo existente', 'comparação exploratória',
                                   'previsões sequenciais: entradas anteriores ao alvo, inclusive do próprio dia de teste',
                                   'piecewise usa o mesmo horário dos 15 dias anteriores']}
        manifest = directory/'manifesto.json'
        if manifest.exists():
            assert json.loads(manifest.read_text(encoding='utf-8'))['protocol'] == protocol
        else:
            write(manifest, {'created_at':now(), 'protocol':protocol})
            for name in ['core_convlstm.py','worker_convlstm.py','treinar_convlstm.py']:
                path=directory/'codigo'/name; path.parent.mkdir(exist_ok=True); shutil.copy2(CODE/name,path)
        def execute(model, variant, settings, phase, label, fixed_epochs=None):
            dest=directory/('busca' if phase=='search' else 'finais')/model/variant/label
            task=dict(station=station, model=model, variant=variant, spec=settings, phase=phase,
                      dataset=str(dataset), observed=str(obs), output=str(dest), hashes=hashes,
                      train_end='2024-02-16 23:30', test_date='2024-02-18')
            if phase=='final':
                task['fixed_epochs']=fixed_epochs
                task['hashes']={**hashes, str(directory/'selecao_congelada.json'):sha(directory/'selecao_congelada.json')}
            taskpath=directory/'tarefas'/f'{model}_{variant}_{label}_{phase}.json'
            if taskpath.exists():
                assert json.loads(taskpath.read_text(encoding='utf-8'))==task
            else:
                write(taskpath, task)
            summary=dest/'resumo.json'
            if summary.exists():
                result=json.loads(summary.read_text(encoding='utf-8'))
                assert result['task_sha256']==sha(taskpath)
                for path,digest in result['artifacts'].items(): assert sha(path)==digest
                return result
            write(directory/'status.json', dict(status='treinando', station=station, model=model,
                  variant=variant, phase=phase, candidate=label, updated_at=now()))
            env=os.environ.copy()
            env.update(TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            log=directory/'logs'/f'{model}_{variant}_{label}_{phase}.log'; log.parent.mkdir(exist_ok=True)
            print('INICIO CONVLSTM', station, model, variant, label, phase, flush=True)
            with log.open('a', encoding='utf-8') as stream:
                subprocess.run([sys.executable,'-X','utf8','-u',str(CODE/'worker_convlstm.py'),
                                '--task',str(taskpath)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            return json.loads(summary.read_text(encoding='utf-8'))
        selected={}
        for model in ['M9','M10','M11','M12']:
            candidates={}
            for label,rate,filters in [('c0',.001,16),('c1',.0003,32)]:
                candidates[label]={v:execute(model,v,spec(model,rate,filters),'search',label)
                                   for v in ['original','filtrado']}
            winner=min(candidates, key=lambda c:sum(r['metrics_common']['RMSE'] for r in candidates[c].values())/2)
            for variant,result in candidates[winner].items():
                selected[f'{model}_{variant}']={'spec':spec(model,.001 if winner=='c0' else .0003,16 if winner=='c0' else 32),
                    'epochs':result['selected_epochs'], 'selection_rmse':result['metrics_common']['RMSE'], 'candidate':winner}
        frozen=directory/'selecao_congelada.json'
        if frozen.exists():
            assert json.loads(frozen.read_text(encoding='utf-8'))['selected']==selected
        else:
            write(frozen, {'frozen_at':now(), 'test_used_for_selection':False, 'selected':selected})
        metrics=[]
        for key,item in selected.items():
            model,variant=key.split('_')
            result=execute(model,variant,item['spec'],'final','selecionado',item['epochs'])
            metrics.append(dict(station=station,model=model,variant=variant,**result['metrics_common']))
        pd.DataFrame(metrics).to_csv(directory/'metricas_comuns.csv',index=False)
        assert len(metrics)==8
        write(directory/'status.json', dict(status='concluido',station=station,final_configurations=8,
                                           predictions=384,networks=196,updated_at=now()))
        subprocess.run([sys.executable,'-X','utf8',str(CODE/'relatorio.py')],cwd=ROOT,check=True)
        station_lock.close()


if __name__=='__main__':
    main()
