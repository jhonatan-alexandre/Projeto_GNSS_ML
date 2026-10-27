"""Sequential resumable regression worker. Runtime roots never enter identity."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import joblib
import numpy as np
import pandas as pd

from scripts.duplo_nucleo.protocolo import Protocol
from scripts.duplo_nucleo import regressao as r

SCHEMA_VERSION = 1
MIN_FREE_BYTES = 5 * 1024**3
CODE_FILES = ['scripts/duplo_nucleo/worker.py','scripts/duplo_nucleo/regressao.py',
              'scripts/duplo_nucleo/protocolo.py','scripts/fevereiro/core_convlstm.py']
PROJECT = Path(__file__).resolve().parents[2]


if os.name=='nt':
    from ctypes import wintypes

    class _MemoryCounters(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
            ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
            ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]

    # Private DLL handles and one stable pointer type; no per-call metadata writes.
    _KERNEL32 = ctypes.WinDLL('kernel32',use_last_error=True)
    _PSAPI = ctypes.WinDLL('psapi',use_last_error=True)
    _CURRENT_PROCESS = _KERNEL32.GetCurrentProcess
    _CURRENT_PROCESS.argtypes = []; _CURRENT_PROCESS.restype = wintypes.HANDLE
    _QUERY_MEMORY = _PSAPI.GetProcessMemoryInfo
    _QUERY_MEMORY.argtypes = [wintypes.HANDLE,ctypes.POINTER(_MemoryCounters),wintypes.DWORD]
    _QUERY_MEMORY.restype = wintypes.BOOL


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def relative_path(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Artifact paths must be relative to output_root and stay inside it')
    result = (Path(root)/path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact path escapes output_root')
    return result


def code_hashes():
    return {p:sha256(PROJECT/p) for p in CODE_FILES}


def validate_task(task, output_root):
    required = {'schema_version','station','model','branch','phase','candidate','protocol',
                'dataset','observed_reference','output_dir','hashes'}
    if not required.issubset(task) or task['schema_version']!=SCHEMA_VERSION:
        raise ValueError('Invalid task schema')
    protocol = Protocol.from_dict(task['protocol'])
    if task['protocol'] != protocol.to_dict():
        raise ValueError('Task requires fully resolved Protocol.to_dict()')
    r.identity(task['model'])
    if task['station'] not in ('FRTZ','PALM','PRU2','POAL') or task['branch'] not in r.BRANCHES or task['phase'] not in ('search','final'):
        raise ValueError('Invalid station, branch or phase')
    if task['candidate'] not in r.candidates(task['model']):
        raise ValueError('Candidate must match approved finite scientific budget')
    for key in ('dataset','observed_reference','output_dir'):
        relative_path(output_root,task[key])
    if not {task['dataset'],task['observed_reference']}.issubset(task['hashes']):
        raise ValueError('Task requires dataset and observed reference hashes')
    for path,digest in task['hashes'].items():
        actual = relative_path(output_root,path)
        if not actual.is_file() or sha256(actual)!=digest:
            raise ValueError(f'Input hash mismatch: {path}')
    slots = task.get('slots',list(range(protocol.slot_count)) if r.identity(task['model'])[2] else ['all'])
    if not slots or len(slots)!=len(set(slots)):
        raise ValueError('Slots must be unique and nonempty')
    if r.identity(task['model'])[2]:
        if any(type(s) is not int or not 0<=s<96 for s in slots): raise ValueError('Invalid piecewise slot')
    elif slots!=['all']:
        raise ValueError('Conventional worker has one all-slot network')
    if task['phase']=='final':
        frozen = task.get('epochs_by_slot',{})
        maximum = r.model_spec(task['model'],task['candidate'])['epochs']
        if any(str(s) not in frozen or type(frozen[str(s)]) is not int or not 1<=frozen[str(s)]<=maximum for s in slots):
            raise ValueError('Final task requires frozen epochs for every requested network')
    scientific = {**task,'code_hashes':code_hashes()}
    # output_dir is a logical relative artifact name; no absolute staging root.
    return hashlib.sha256(json.dumps(scientific,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def disk_check(root):
    free = shutil.disk_usage(root).free
    if free<MIN_FREE_BYTES: raise RuntimeError(f'At least5GiB free required, available={free}bytes')
    return free


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    tmp.replace(path)


def record(path, root):
    return {'path':Path(path).resolve().relative_to(Path(root).resolve()).as_posix(),
            'sha256':sha256(path),'bytes':Path(path).stat().st_size}


def write_summary(path, data):
    write_json(path,data)
    write_json(Path(path).with_suffix('.integrity.json'),{'sha256':sha256(path)})


def verify_artifacts(row, root):
    for artifact in row.get('artifacts',{}).values():
        path = relative_path(root,artifact['path'])
        if not path.is_file() or sha256(path)!=artifact['sha256']:
            raise ValueError(f'Completed checkpoint artifact mismatch: {artifact["path"]}')


class Resources:
    def __enter__(self):
        self.wall = time.perf_counter(); self.cpu = time.process_time()
        self.initial,self.peak = process_memory()
        self.sampled_peak_rss = self.initial
        self.stop = threading.Event(); self.lock = threading.Lock(); self.error = None
        def sample():
            try:
                while not self.stop.wait(.05):
                    rss,peak = process_memory()
                    with self.lock:
                        self.peak = max(self.peak,rss,peak)
                        self.sampled_peak_rss = max(self.sampled_peak_rss,rss)
            except Exception as exc:
                with self.lock: self.error = exc
                self.stop.set()
        self.thread = threading.Thread(target=sample,daemon=True); self.thread.start()
        return self

    def values(self):
        with self.lock:
            if self.error is not None: raise self.error
        rss,peak = process_memory()
        with self.lock:
            if self.error is not None: raise self.error
            self.peak = max(self.peak,rss,peak)
            self.sampled_peak_rss = max(self.sampled_peak_rss,rss)
            measured_peak = self.peak; sampled_peak_rss = self.sampled_peak_rss
        return {'wall_seconds':time.perf_counter()-self.wall,'cpu_seconds':time.process_time()-self.cpu,
                'peak_process_memory_bytes':measured_peak,'initial_process_memory_bytes':self.initial,
                'peak_memory_above_initial_bytes':max(0,measured_peak-self.initial),
                'memory_peak_scope':'process_lifetime',
                'memory_above_initial_scope':'process_lifetime_peak_minus_initial_rss_not_interval_increment',
                'sampled_interval_peak_rss_bytes':sampled_peak_rss,
                'memory_measurement':'native process lifetime peak working set plus interval RSS samples'}

    def __exit__(self,kind,value,traceback):
        self.stop.set(); self.thread.join()
        with self.lock: error = self.error
        if error is not None:
            if value is not None and value is not error: raise error from value
            raise error


def process_memory():
    """Native per-process RSS/peak, requiring no optional psutil dependency."""
    if os.name=='nt':
        counters = _MemoryCounters(); counters.cb = ctypes.sizeof(counters)
        if not _QUERY_MEMORY(_CURRENT_PROCESS(),ctypes.byref(counters),counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize),int(counters.PeakWorkingSetSize)
    import resource
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak *= 1 if os.sys.platform=='darwin' else 1024
    statm = Path('/proc/self/statm')
    rss = int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if statm.exists() else peak
    return rss,peak


def infer(model, x, batch):
    return np.concatenate([np.asarray(model(x[i:i+batch],training=False)).reshape(-1)
                           for i in range(0,len(x),batch)]) if len(x) else np.empty(0)


def train(tf, model, prepared, spec, phase, epochs):
    """Explicit batches avoid tf.data background pools for thousands of small nets."""
    fit = prepared['fit']; x,y = prepared['X'][fit],prepared['y'][fit]
    early_x,early_y = prepared['X'][prepared['early']],prepared['y'][prepared['early']]
    rng = np.random.default_rng(spec['seed'])
    history = []; best = float('inf'); best_epoch = 0; best_weights = None; wait = 0
    batch = min(spec['batch_size'],len(x))
    for epoch in range(1,epochs+1):
        order = rng.permutation(len(x)); model.reset_metrics()
        for first in range(0,len(x),batch):
            ids = order[first:first+batch]; model.train_on_batch(x[ids],y[ids,None])
        loss = float(np.mean((infer(model,x,batch)-y)**2))
        row = {'epoch':epoch,'loss':loss}
        if phase=='search':
            validation = float(np.mean((infer(model,early_x,batch)-early_y)**2)); row['val_loss'] = validation
            if np.isfinite(validation) and validation<best:
                best = validation; best_epoch = epoch; best_weights = model.get_weights(); wait = 0
            else: wait += 1
        history.append(row)
        if phase=='search' and wait>=spec['patience']: break
    if phase=='search':
        if best_weights is None: raise RuntimeError('No finite early stopping loss')
        model.set_weights(best_weights)
    else: best_epoch = epochs
    return history,best_epoch


def _execute_slot(task,root,protocol,frame,observed,key,tf):
    slot = None if key=='all' else int(key)
    output = relative_path(root,task['output_dir'])/f'slot_{key}'
    output.mkdir(parents=True,exist_ok=True)
    with Resources() as resources:
        preparation_start = time.perf_counter()
        plan = r.paired_windows(frame,observed,task['model'],protocol,slot)
        if not plan['eligible']:
            row = {'status':'blocked','reason':plan['reason'],'coverage':plan,
                   'resources':resources.values(),'artifacts':{}}
            write_json(output/'blocked.json',row)
            row['artifacts']['blocked'] = record(output/'blocked.json',root)
            return row
        supplied = task.get('windows_by_slot',{}).get(key)
        if supplied is not None and supplied!=plan['windows']:
            raise ValueError('Frozen coverage windows differ from observed development coverage')
        prepared = r.prepare_sequences(frame,observed,task['model'],task['branch'],protocol,
                                       task['phase'],slot,plan['windows'])
        if prepared['partition']['fit']['n']<10: raise ValueError('Final fit below10-sequence minimum')
        preparation_seconds = time.perf_counter()-preparation_start
        spec = r.model_spec(task['model'],task['candidate'])
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(spec['seed'])
        disk_before = disk_check(root)
        model = r.build_model(tf,prepared['X'].shape[1:],spec)
        training_start = time.perf_counter(); training_cpu = time.process_time()
        epochs = spec['epochs'] if task['phase']=='search' else task['epochs_by_slot'][key]
        history,selected = train(tf,model,prepared,spec,task['phase'],epochs)
        training_wall = time.perf_counter()-training_start; training_cpu = time.process_time()-training_cpu
        mask_name = 'selection' if task['phase']=='search' else 'test'
        mask = prepared[mask_name]; x = prepared['X'][mask]
        inference_start = time.perf_counter(); inference_cpu = time.process_time()
        scaled = infer(model,x,spec['batch_size'])
        predictions = prepared['sy'].inverse_transform(scaled[:,None]).ravel() if len(scaled) else scaled
        inference_wall = time.perf_counter()-inference_start; inference_cpu = time.process_time()-inference_cpu
        t = pd.to_datetime(prepared['times'][mask])
        table = pd.DataFrame({'datetime':t,'station':task['station'],'model':task['model'],'branch':task['branch'],
                'candidate':task['candidate']['id'],'slot':key,'phase':task['phase'], 'emission_datetime':t,
                'target_available_at':t+pd.Timedelta(protocol.time_resolution),
                'input_available_at':pd.to_datetime(prepared['input_available_at'][mask]),
                'VTEC_observed_reference':prepared['common'][mask],'observed_reference_available':np.isfinite(prepared['common'][mask]),
                'VTEC_branch_target':prepared['raw_y'][mask],'prediction':predictions,
                'persistence':prepared['persistence'][mask],'retrospective_interpolation':True})
        table.to_csv(output/'predictions.csv',index=False)
        pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        write_json(output/'partition.json',{'partition':prepared['partition'],'coverage':plan,'columns':prepared['cols']})
        # Measure real serialization, retain binary models only for final phase.
        disk_before_save = disk_check(root)
        model_path = output/('model.keras' if task['phase']=='final' else 'candidate_incomplete.keras')
        model.save(model_path); model_bytes = model_path.stat().st_size
        artifacts = {name:record(output/file,root) for name,file in
                     [('predictions','predictions.csv'),('history','history.csv'),('partition','partition.json')]}
        reload_seconds = None
        if task['phase']=='final':
            joblib.dump({'sx':prepared['sx'],'sy':prepared['sy'],'columns':prepared['cols']},output/'scalers.joblib')
            start = time.perf_counter(); loaded = tf.keras.models.load_model(model_path)
            restored = infer(loaded,x,spec['batch_size'])
            if not np.allclose(scaled,restored,rtol=1e-5,atol=1e-6,equal_nan=False):
                raise RuntimeError('Reloaded final model predictions mismatch')
            restored_scalers = joblib.load(output/'scalers.joblib')
            if not np.array_equal(prepared['sy'].scale_,restored_scalers['sy'].scale_):
                raise RuntimeError('Reloaded final scaler mismatch')
            reload_seconds = time.perf_counter()-start
            artifacts['model'] = record(model_path,root); artifacts['scalers'] = record(output/'scalers.joblib',root)
            del loaded
        else: model_path.unlink()
        selection = {'times':[str(x) for x in t],'common':[float(x) for x in prepared['common'][mask]],
                     'prediction':[float(x) for x in predictions]} if task['phase']=='search' else None
        result = {'status':'complete','candidate':task['candidate'],'spec':spec,'coverage':plan,'partition':prepared['partition'],
                  'epochs_executed':len(history),'epochs_selected':selected,'metrics':r.historical.scores(prepared['common'][mask],predictions),
                  'selection':selection,'artifacts':artifacts,'resources':{**resources.values(),
                    'preparation_wall_seconds':preparation_seconds,'training_wall_seconds':training_wall,
                    'training_cpu_seconds':training_cpu,'inference_wall_seconds':inference_wall,
                    'inference_cpu_seconds':inference_cpu,'inference_n':len(predictions),
                    'inference_seconds_per_prediction':inference_wall/len(predictions) if len(predictions) else None,
                    'model_bytes':model_bytes,'model_retained':task['phase']=='final','parameters':model.count_params(),
                    'trainable_parameters':int(sum(np.prod(v.shape) for v in model.trainable_weights)),
                    'reload_verify_seconds':reload_seconds,'free_bytes_before_training':disk_before,'free_bytes_before_model_save':disk_before_save}}
        if not np.isfinite(predictions).all(): raise RuntimeError('Nonfinite model predictions')
        write_json(output/'slot_summary.json',result)
        result['artifacts']['slot_summary'] = record(output/'slot_summary.json',root)
        del model; tf.keras.backend.clear_session()
        return result


def execute_task(task, output_root, resume=False):
    root = Path(output_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    task_hash = validate_task(task,root); protocol = Protocol.from_dict(task['protocol'])
    output = relative_path(root,task['output_dir']); summary_path = output/'summary.json'
    if summary_path.exists():
        if not resume: raise ValueError('Existing task requires resume or a new output_dir')
        integrity = summary_path.with_suffix('.integrity.json')
        if not integrity.is_file() or json.loads(integrity.read_text(encoding='utf8')).get('sha256')!=sha256(summary_path):
            raise ValueError('Completed/partial checkpoint summary integrity mismatch')
        summary = json.loads(summary_path.read_text(encoding='utf8'))
        if summary['task_sha256']!=task_hash: raise ValueError('Resume scientific task/code/data hash mismatch')
        for row in summary['slots'].values(): verify_artifacts(row,root)
    else:
        summary = {'schema_version':SCHEMA_VERSION,'status':'running','task_sha256':task_hash,
                   'station':task['station'],'model':task['model'],'branch':task['branch'],'phase':task['phase'],
                   'candidate':task['candidate'],'protocol':protocol.to_dict(),'input_hashes':task['hashes'],
                   'code_hashes':code_hashes(),'slots':{},'retrospective_interpolation':True,
                   'resume_boundary':'completed slot; interrupted slot retrains from scratch',
                   'environment':{'python':os.sys.version,'numpy':np.__version__,'pandas':pd.__version__}}
    slots = task.get('slots',list(range(protocol.slot_count)) if r.identity(task['model'])[2] else ['all'])
    pending = [str(s) for s in slots if str(s) not in summary['slots']]
    if not pending:
        if summary['status']!='complete':
            summary['status'] = 'complete'; write_summary(summary_path,summary)
        return summary
    disk_check(root)
    import tensorflow as tf
    summary['environment']['tensorflow'] = tf.__version__
    frame = pd.read_csv(relative_path(root,task['dataset']),parse_dates=['datetime'])
    ref = pd.read_csv(relative_path(root,task['observed_reference']),parse_dates=['datetime'])
    if ref.datetime.duplicated().any(): raise ValueError('Duplicate reference timestamps')
    observed = ref.set_index('datetime').VTEC_observed_reference.copy()
    if 'observed_reference_available' in ref:
        unavailable = ~ref.observed_reference_available.astype(str).str.lower().isin(['true','1'])
        observed.loc[ref.datetime[unavailable]] = np.nan
    output.mkdir(parents=True,exist_ok=True)
    wall = time.perf_counter(); cpu = time.process_time()
    write_json(output/'task.json',task)
    for key in pending:
        with Resources() as failed_resources:
            try: summary['slots'][key] = _execute_slot(task,root,protocol,frame,observed,key,tf)
            except Exception as exc:
                summary.setdefault('failed_attempts',[]).append({'slot':key,'error':type(exc).__name__,
                     'message':str(exc),'resources':failed_resources.values()})
                summary['status'] = 'failed'; write_summary(summary_path,summary); raise
        summary['status'] = 'running'; write_summary(summary_path,summary)
    summary.setdefault('execution_segments',[]).append({'wall_seconds':time.perf_counter()-wall,
                                                      'cpu_seconds':time.process_time()-cpu,'slots':pending})
    summary['status'] = 'complete'; write_summary(summary_path,summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True); parser.add_argument('--output-root',required=True)
    parser.add_argument('--resume',action='store_true'); args = parser.parse_args()
    task = json.loads(Path(args.task).read_text(encoding='utf8'))
    summary = execute_task(task,args.output_root,args.resume)
    print(json.dumps({'status':summary['status'],'task_sha256':summary['task_sha256'],
                      'networks_complete':sum(x['status']=='complete' for x in summary['slots'].values()),
                      'networks_blocked':sum(x['status']=='blocked' for x in summary['slots'].values())}))


if __name__ == '__main__': main()
