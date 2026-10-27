"""Retomada de POAL com a exceção temporal aprovada e reuso auditável."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
from core_convlstm import ROOT, OUT, sha, write, spec
from reajustar_lstm import run as refit_lstm
from lock import acquire

CODE = Path(__file__).resolve().parent
DIRECTORY = OUT/'estacoes/POAL/convlstm'


def now():
    return datetime.now(timezone.utc).isoformat()


def verified_summary(taskpath):
    task = json.loads(taskpath.read_text(encoding='utf-8'))
    for path, digest in task['hashes'].items():
        assert sha(path) == digest, path
    summary = Path(task['output'])/'resumo.json'
    if not summary.exists():
        return None
    result = json.loads(summary.read_text(encoding='utf-8'))
    assert result['task_sha256'] == sha(taskpath)
    for path, digest in result['artifacts'].items():
        assert sha(path) == digest, path
    return result


def main():
    station_lock = acquire(OUT/'estacoes/POAL/treinamento.lock')
    approval = OUT/'validacao_POAL_autorizada.json'
    decision = json.loads(approval.read_text(encoding='utf-8'))
    assert decision['status'] == 'autorizado_pelo_usuario'
    assert decision['affected_slots'] == list(range(14, 42))
    assert decision['early_stopping'] == ['2024-01-27', '2024-02-02']
    assert decision['selection'] == ['2024-02-03', '2024-02-09']
    assert decision['test_date'] == '2024-02-18'
    assert sha(decision['source']) == decision['source_sha256']
    base = ROOT/'experimentos/poal_comum_18fev2024'
    dataset = base/'dados_processados/POAL/POAL_dataset_regressao.csv'
    observed = DIRECTORY/'vtec_observado_30min.csv'
    refit_lstm('POAL', observed)
    sources = [CODE/name for name in ('core_convlstm.py', 'worker_convlstm.py',
                'poal_validation.py', 'worker_poal.py', 'treinar_poal.py')]
    sources += [approval, dataset, observed, Path(decision['source']),
                ROOT/'configs/xie_poal_comum_18fev2024.json']
    hashes = {str(path): sha(path) for path in sources}
    manifest = DIRECTORY/'manifesto_validacao_anterior.json'
    protocol = dict(station='POAL', affected_models=['M11','M12'], affected_slots=list(range(14,42)),
                    search_fit_end='2024-01-26 23:30', early_stopping=['2024-01-27','2024-02-02'],
                    selection=['2024-02-03','2024-02-09'], final_train_end='2024-02-16 23:30',
                    test_date='2024-02-18', test_used_for_selection=False, input_hashes=hashes,
                    note='Demais horários e modelos mantêm a seleção original. RMSE agrupado nas épocas observadas de cada horário.')
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding='utf-8'))['protocol'] == protocol
    else:
        write(manifest, dict(created_at=now(), protocol=protocol))

    def execute(model, variant, settings, phase, label, fixed_epochs=None):
        legacy = DIRECTORY/'tarefas'/f'{model}_{variant}_{label}_{phase}.json'
        if phase == 'search' and model in ('M9','M10'):
            # Estes oito candidatos já foram concluídos com as janelas originais.
            prior = json.loads(legacy.read_text(encoding='utf-8'))
            for key, value in dict(station='POAL',model=model,variant=variant,spec=settings,phase=phase,
                    dataset=str(dataset),observed=str(observed),train_end='2024-02-16 23:30',test_date='2024-02-18').items():
                assert prior[key] == value, key
            result = verified_summary(legacy)
            assert result is not None, f'Candidato original ausente: {legacy}'
            print('REUSO CANDIDATO', model, variant, label, flush=True)
            return result, legacy
        output_label = label+'_validacao_anterior' if phase == 'search' else label
        dest = DIRECTORY/('busca' if phase=='search' else 'finais')/model/variant/output_label
        task = dict(station='POAL', model=model, variant=variant, spec=settings, phase=phase,
                    dataset=str(dataset), observed=str(observed), output=str(dest), hashes=dict(hashes),
                    train_end='2024-02-16 23:30', test_date='2024-02-18')
        if phase == 'search' and legacy.exists():
            task['legacy_task'] = str(legacy)
            task['hashes'][str(legacy)] = sha(legacy)
        if phase == 'final':
            task['fixed_epochs'] = fixed_epochs
            frozen = DIRECTORY/'selecao_congelada.json'
            task['hashes'][str(frozen)] = sha(frozen)
        taskpath = DIRECTORY/'tarefas'/f'{model}_{variant}_{output_label}_{phase}.json'
        if taskpath.exists():
            assert json.loads(taskpath.read_text(encoding='utf-8')) == task
        else:
            write(taskpath, task)
        result = verified_summary(taskpath)
        if result is not None:
            return result, taskpath
        write(DIRECTORY/'status.json', dict(status='treinando',station='POAL',model=model,
              variant=variant,phase=phase,candidate=output_label,updated_at=now(),
              validation_exception=str(approval)))
        env = os.environ.copy()
        env.update(TF_ENABLE_ONEDNN_OPTS='0',TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        log = DIRECTORY/'logs'/f'{model}_{variant}_{output_label}_{phase}.log'
        print('INICIO POAL', model, variant, output_label, phase, now(), flush=True)
        with log.open('a', encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-X','utf8','-u',str(CODE/'worker_poal.py'),
                            '--task',str(taskpath)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        result = verified_summary(taskpath)
        assert result is not None
        return result, taskpath

    selected = {}; selection_sources = {}
    for model in ('M9','M10','M11','M12'):
        candidates = {}
        for label, rate, filters in [('c0',.001,16),('c1',.0003,32)]:
            candidates[label] = {}
            for variant in ('original','filtrado'):
                result, taskpath = execute(model,variant,spec(model,rate,filters),'search',label)
                candidates[label][variant] = result
                selection_sources[str(taskpath)] = sha(taskpath)
                summary = Path(json.loads(taskpath.read_text(encoding='utf-8'))['output'])/'resumo.json'
                selection_sources[str(summary)] = sha(summary)
        winner = min(candidates,key=lambda c:sum(r['metrics_common']['RMSE'] for r in candidates[c].values())/2)
        for variant, result in candidates[winner].items():
            selected[f'{model}_{variant}'] = dict(spec=spec(model,.001 if winner=='c0' else .0003,16 if winner=='c0' else 32),
                epochs=result['selected_epochs'],selection_rmse=result['metrics_common']['RMSE'],candidate=winner)
    frozen = DIRECTORY/'selecao_congelada.json'
    if frozen.exists():
        existing = json.loads(frozen.read_text(encoding='utf-8'))
        assert existing['selected'] == selected and existing['selection_sources'] == selection_sources
    else:
        write(frozen,dict(frozen_at=now(),test_used_for_selection=False,selected=selected,
                         selection_sources=selection_sources,validation_exception=str(approval)))
    metrics = []
    for key, item in selected.items():
        model, variant = key.split('_')
        result, _ = execute(model,variant,item['spec'],'final','selecionado',item['epochs'])
        metrics.append(dict(station='POAL',model=model,variant=variant,**result['metrics_common']))
    pd.DataFrame(metrics).to_csv(DIRECTORY/'metricas_comuns.csv',index=False)
    assert len(metrics) == 8
    write(DIRECTORY/'status.json',dict(status='concluido',station='POAL',final_configurations=8,
          predictions=384,networks=196,updated_at=now(),validation_exception=str(approval)))
    subprocess.run([sys.executable,'-X','utf8',str(CODE/'relatorio.py')],cwd=ROOT,check=True)
    station_lock.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(DIRECTORY/'erro_validacao_anterior.json',dict(status='erro',error=str(exc),updated_at=now()))
        raise
