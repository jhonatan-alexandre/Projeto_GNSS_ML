"""Exceção de POAL autorizada em 23/09/2026; fontes originais preservadas."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from core_convlstm import prepare, sha


def uses_earlier_window(task, slot):
    return (task['station'] == 'POAL' and task['phase'] == 'search'
            and task['model'] in ('M11', 'M12') and slot is not None and 14 <= slot <= 41)


def prepare_for_task(frame, observed, task, slot=None):
    function = _prepare_earlier if uses_earlier_window(task, slot) else prepare
    return function(frame, observed, task['model'], task['variant'], task['phase'], slot)


def legacy_slot(task, slot):
    """Reutiliza somente um slot não afetado, com proveniência original íntegra."""
    if not task.get('legacy_task') or uses_earlier_window(task, slot) or task['phase'] != 'search':
        return None
    path = Path(task['legacy_task'])
    assert sha(path) == task['hashes'][str(path)]
    old = json.loads(path.read_text(encoding='utf-8'))
    for key in ('station', 'model', 'variant', 'spec', 'phase', 'dataset', 'observed', 'train_end', 'test_date'):
        assert old[key] == task[key], key
    for filename, digest in old['hashes'].items():
        assert sha(filename) == digest, filename
    prefix = Path(old['output']) / str(slot)
    history = prefix / 'historico.json'
    if not history.exists():
        return None
    record = json.loads(history.read_text(encoding='utf-8'))
    assert record['task_sha256'] == sha(path)
    for filename, digest in record['artifacts'].items():
        assert sha(filename) == digest, filename
    return prefix, record


def _prepare_earlier(frame,observed,model,variant,phase,slot=None):
    if phase not in ('search','final'):raise ValueError('Fase inválida')
    number=int(model[1:]);multi=number in (2,4,6,8,9,12);piecewise=number in (5,6,7,8,11,12)
    f=frame.copy();f['datetime']=pd.to_datetime(f.datetime)
    assert f.datetime.is_monotonic_increasing and not f.datetime.duplicated().any()
    end=pd.Timestamp('2024-02-09 23:30')
    test=pd.Timestamp('2024-02-18')
    test_end=test + pd.Timedelta(hours=23, minutes=30)
    if phase=='search':f=f.loc[f.datetime<=end].copy()
    if piecewise:
        if slot is None:raise ValueError('Slot obrigatório')
        f=f.loc[f.datetime.dt.hour*2+f.datetime.dt.minute//30==slot].copy()
    common=observed.reindex(f.datetime).to_numpy(float)
    target_column='VTEC_original' if variant=='original' else 'VTEC_filtered'
    cols=[target_column]+(['Kp','Dst','F107','Ap','SSN'] if multi else [])
    raw_x=f[cols].to_numpy(float)
    raw_y=common.copy() if number>=9 else f[target_column].to_numpy(float)
    times=f.datetime.to_numpy();lookback=15 if piecewise else 48
    cutoff=end-pd.Timedelta(days=14) if phase=='search' else end
    train_rows=(times<=np.datetime64(cutoff))&np.isfinite(common)&np.isfinite(raw_y)
    if train_rows.sum()<2:raise ValueError('Histórico observado insuficiente')
    sx,sy=MinMaxScaler(),MinMaxScaler()
    sx.fit(raw_x[(times<=np.datetime64(cutoff))&np.isfinite(raw_x).all(axis=1)])
    sy.fit(raw_y[train_rows,None])
    sx.scale_[0],sx.min_[0]=sy.scale_[0],sy.min_[0]
    sx.data_min_[0],sx.data_max_[0],sx.data_range_[0]=sy.data_min_[0],sy.data_max_[0],sy.data_range_[0]
    xx=sx.transform(raw_x).astype('float32');yy=sy.transform(raw_y[:,None]).ravel().astype('float32')
    step=np.timedelta64(1,'D') if piecewise else np.timedelta64(30,'m')
    idx=np.array([i for i in range(lookback,len(f)) if times[i]-times[i-lookback]==step*lookback and np.isfinite(xx[i-lookback:i]).all()],int)
    t=times[idx];valid=np.isfinite(common[idx])&np.isfinite(raw_y[idx])
    masks={'fit':(t<=np.datetime64(cutoff))&valid,
           'early':(t>np.datetime64(cutoff))&(t<=np.datetime64(end-pd.Timedelta(days=7)))&valid if phase=='search' else np.zeros(len(t),bool),
           'selection':(t>np.datetime64(end-pd.Timedelta(days=7)))&(t<=np.datetime64(end))&valid if phase=='search' else np.zeros(len(t),bool),
           'test':(t>=np.datetime64(test))&(t<=np.datetime64(test_end))}
    if not masks['fit'].any():raise ValueError('Ajuste vazio')
    if phase=='search' and (not masks['early'].any() or not masks['selection'].any()):raise ValueError('Validação sem alvos observados')
    if phase=='search':assert not masks['test'].any()
    if phase=='final':
        assert masks['fit'].sum()==int(((t<=np.datetime64(end))&valid).sum())
        assert masks['test'].sum()==(1 if piecewise else 48)
        assert np.isfinite(common[idx][masks['test']]).all(),'Teste incompleto: recuperar dados antes de avaliar.'
    partition={k:dict(n=int(m.sum()),start=str(pd.Timestamp(t[m][0])) if m.any() else None,end=str(pd.Timestamp(t[m][-1])) if m.any() else None) for k,m in masks.items()}
    return dict(X=np.asarray([xx[i-lookback:i] for i in idx]),y=yy[idx],raw_y=raw_y[idx],common=common[idx],times=t,
                persistence=f.VTEC_original.to_numpy()[idx-1],sx=sx,sy=sy,cols=cols,partition=partition,**masks)

