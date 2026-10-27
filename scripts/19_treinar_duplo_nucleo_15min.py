"""Run the frozen 15-minute experiment with one monitored child at a time.

Outputs belong to the explicit runtime root. Completed workers always validate
their own checkpoints on resume; runner status is an activity journal only.
"""
import argparse
import copy
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.duplo_nucleo.protocolo import Protocol
from scripts.duplo_nucleo import regressao as r, worker as w

MIN_FREE_BYTES = 5 * 1024**3
SCIENTIFIC_CODE = [
    'scripts/19_treinar_duplo_nucleo_15min.py', 'scripts/duplo_nucleo/protocolo.py',
    'scripts/duplo_nucleo/preparacao.py', 'scripts/duplo_nucleo/regressao.py',
    'scripts/duplo_nucleo/worker.py', 'scripts/fevereiro/core_convlstm.py',
    'scripts/05_integrar_clima_espacial.py', 'scripts/06_feature_engineering.py',
    'scripts/07_dataset_classificacao.py', 'scripts/08_train_xgboost.py',
    'scripts/interpolacao_vtec.py',
]
CHILD_ENV = {'TF_ENABLE_ONEDNN_OPTS':'0', 'TF_CPP_MIN_LOG_LEVEL':'2',
             'TF_NUM_INTRAOP_THREADS':'2', 'TF_NUM_INTEROP_THREADS':'2',
             'OMP_NUM_THREADS':'2', 'LOKY_MAX_CPU_COUNT':'2'}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def require_disk(root):
    path = Path(root).resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f'At least 5 GiB free required; available {free} bytes')
    return free


def scientific_fingerprint(config, input_hashes, code_files=None):
    files = code_files if code_files is not None else [PROJECT/p for p in SCIENTIFIC_CODE]
    # Stable project labels; output location and reporting code are absent.
    code = {p.relative_to(PROJECT).as_posix() if p.is_relative_to(PROJECT) else p.name:w.sha256(p)
            for p in map(Path, files)}
    value = {'config':config, 'input_hashes':input_hashes, 'code_hashes':code}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()


if os.name == 'nt':
    _kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _open_process = _kernel.OpenProcess
    _open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    _open_process.restype = ctypes.c_void_p
    _exit_code = _kernel.GetExitCodeProcess
    _exit_code.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    _exit_code.restype = ctypes.c_int
    _close_handle = _kernel.CloseHandle
    _close_handle.argtypes = [ctypes.c_void_p]
    _close_handle.restype = ctypes.c_int


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == 'nt':
        handle = _open_process(0x1000, False, pid)
        if not handle:
            # Access denied is evidence of a process, not permission to duplicate it.
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_ulong()
            return bool(_exit_code(handle, ctypes.byref(code))) and code.value == 259
        finally:
            _close_handle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def new_state(fingerprint, protocol, stations, root):
    return {'schema_version':1, 'fingerprint':fingerprint, 'protocol':protocol.to_dict(),
            'stations':stations, 'status':'initialized', 'stage':'initialized',
            'pid':None, 'child_pid':None, 'activity_verified':False,
            'started_at':now(), 'updated_at':now(), 'runtime_output_root':str(Path(root).resolve()),
            'durable_output_root':str(PROJECT/'experimentos/duplo_nucleo_15min_20261013'),
            'resume_path':'runner_status.json', 'tasks':{}, 'selections':{}, 'failures':[],
            'matrix':matrix_counts(stations), 'retrospective_interpolation':True,
            'resume_boundary':'completed worker slot; partial preparation restarts',
            'reporting':{'status':'pending'}}


def reconcile_activity(state, alive=process_alive):
    state = copy.deepcopy(state)
    parent_alive = alive(state.get('pid'))
    child_alive = alive(state.get('child_pid'))
    state['activity_verified'] = bool(parent_alive or child_alive) and state['status']=='running'
    state['activity_checked_at'] = now()
    if state['status']=='running' and not parent_alive and not child_alive:
        state['status'] = 'interrupted'
        state['pid'] = state['child_pid'] = None
        for task in state['tasks'].values():
            if task['status']=='running':
                task.update(status='interrupted', activity_verified=False, interrupted_at=now())
    return state


def load_state(root, fingerprint, protocol, stations, resume=False):
    path = Path(root)/'runner_status.json'
    if not path.exists():
        return new_state(fingerprint, protocol, stations, root)
    if not resume:
        raise FileExistsError('Existing runner status requires --resume or a new output root')
    state = json.loads(path.read_text(encoding='utf8'))
    if state.get('schema_version') != 1:
        raise ValueError('Invalid runner state schema')
    previous = Protocol.from_dict(state['protocol'])
    if previous.to_dict()!=protocol.to_dict() or state.get('fingerprint')!=fingerprint:
        raise ValueError('Resume fingerprint/protocol/code/data mismatch')
    state = reconcile_activity(state)
    if state['activity_verified']:
        raise RuntimeError('Another runner or child is still active; refusing concurrent training')
    state['stations'] = stations
    state['matrix'] = matrix_counts(stations)
    state['runtime_output_root'] = str(Path(root).resolve())
    return state


def save_state(root, state):
    state['updated_at'] = now()
    atomic_json(Path(root)/'runner_status.json', state)


@contextmanager
def runner_lock(root):
    """OS lock released on process exit; persistent file avoids stale PID races."""
    path = Path(root)/'.runner.lock'
    with path.open('a+b') as stream:
        stream.seek(0,os.SEEK_END)
        if stream.tell()==0:
            stream.write(b'\0'); stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another runner is active for this output root') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name=='nt':
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def execute_child(command, root, state, key, log_path, save=None, poll_seconds=1):
    """Wait for each child, capture PID/poll evidence, and reap even on failure."""
    if state.get('child_pid') and process_alive(state['child_pid']):
        raise RuntimeError('A child is already active')
    require_disk(root)
    save = save or (lambda s:save_state(root,s))
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    row = {'status':'running','stage':state['stage'],'started_at':now(),
           'log':log_path.resolve().relative_to(Path(root).resolve()).as_posix(),
           'command':command,'pid':None,'activity_verified':False}
    state['tasks'][key] = row
    state.update(status='running',pid=os.getpid(),current_task=key,child_pid=None,activity_verified=True)
    save(state)
    start = time.perf_counter()
    child = None
    try:
        with log_path.open('a',encoding='utf8') as log:
            child = subprocess.Popen(command,cwd=PROJECT,env={**os.environ,**CHILD_ENV},
                                     stdout=log,stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            state['child_pid'] = child.pid
            while True:
                exit_code = child.poll()
                row['activity_verified'] = exit_code is None
                row['activity_checked_at'] = now()
                state['activity_verified'] = exit_code is None
                save(state)
                if exit_code is not None:
                    break
                time.sleep(poll_seconds)
        row.update(status='complete' if exit_code==0 else 'failed',exit_code=exit_code,
                   finished_at=now(),wall_seconds=time.perf_counter()-start,activity_verified=False)
        state.update(child_pid=None,activity_verified=False,current_task=None)
        if exit_code:
            raise RuntimeError(f'{key} failed with exit code {exit_code}; see {row["log"]}')
        save(state)
        return row
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        row.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                   finished_at=now(),wall_seconds=time.perf_counter()-start,activity_verified=False,
                   error=f'{type(exc).__name__}: {exc}')
        if child is not None:
            row['exit_code'] = child.returncode
        state.update(status=row['status'],child_pid=None,activity_verified=False,current_task=None)
        state['failures'].append({'task':key,'at':now(),'error':row['error'],'log':row['log']})
        save(state)
        raise


def matrix_counts(stations):
    final = search = 0
    for model in (f'M{i}' for i in range(1,13)):
        networks = 96 if r.identity(model)[2] else 1
        final += networks*2
        search += networks*2*len(r.candidates(model))
    return {'configurations':len(stations)*12*2,'max_final_networks':len(stations)*final,
            'max_search_networks':len(stations)*search,
            'final_networks_complete':0,'final_networks_blocked':0,'final_networks_pending':len(stations)*final,
            'search_networks_complete':0,'search_networks_blocked':0}


def make_task(root, station, model, branch, phase, candidate, protocol, prepared, pilot=False):
    root = Path(root).resolve()
    dataset = Path(prepared['regression']).resolve().relative_to(root).as_posix()
    reference = Path(prepared['observed_reference']).resolve().relative_to(root).as_posix()
    piecewise = r.identity(model)[2]
    prefix = 'piloto_runner' if pilot else 'resultados'
    return {'schema_version':1,'station':station,'model':model,'branch':branch,'phase':phase,
            'candidate':candidate,'protocol':protocol.to_dict(),'dataset':dataset,'observed_reference':reference,
            'output_dir':f'{prefix}/{station}/nucleo2/{model}/{branch}/{phase}/{candidate["id"]}',
            'slots':[0] if pilot and piecewise else list(range(protocol.slot_count)) if piecewise else ['all'],
            'hashes':{dataset:w.sha256(root/dataset), reference:w.sha256(root/reference)}}


def execute_worker(task, root, state, key, resume=False):
    task_hash = w.validate_task(task,root)
    path = Path(root)/'tasks'/f'{key}.json'
    atomic_json(path,task)
    command = [sys.executable,'-m','scripts.duplo_nucleo.worker','--task',str(path), '--output-root',str(root)]
    if resume:
        command.append('--resume')
    # No inline runner cache is authoritative. The child verifies summary
    # integrity, task identity and all artifacts even if previously complete.
    execute_child(command,root,state,key,Path(root)/'logs'/f'{key}.log')
    summary_path = w.relative_path(root,task['output_dir'])/'summary.json'
    integrity = summary_path.with_suffix('.integrity.json')
    if not integrity.is_file() or json.loads(integrity.read_text(encoding='utf8')).get('sha256')!=w.sha256(summary_path):
        raise ValueError('Worker summary integrity mismatch after child completion')
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    if summary['status']!='complete' or summary['task_sha256']!=task_hash:
        raise ValueError('Worker did not produce a scientifically matching complete summary')
    expected = {str(s) for s in task['slots']}
    if set(summary['slots'])!=expected or any(row['status'] not in ('complete','blocked') for row in summary['slots'].values()):
        raise ValueError('Worker summary slot set/status mismatch')
    for row in summary['slots'].values():
        w.verify_artifacts(row,root)
    row = state['tasks'].setdefault(key,{})
    row.update(status='complete',task_sha256=task_hash,phase=task['phase'],station=task['station'],
               model=task['model'],branch=task['branch'],candidate=task['candidate'],
               summary=summary_path.relative_to(root).as_posix(),summary_sha256=w.sha256(summary_path),
               slots={slot:{'status':v['status'],'reason':v.get('reason')} for slot,v in summary['slots'].items()})
    save_state(root,state)
    return summary


def final_task(search_task, selected, winner_summary):
    task = copy.deepcopy(search_task)
    epochs = selected['epochs_by_branch'][task['branch']]
    if not epochs:
        raise ValueError('No eligible frozen networks for final task')
    task.update(phase='final',candidate=selected['candidate'],epochs_by_slot=epochs,
                slots=[int(s) if s!='all' else s for s in epochs],
                windows_by_slot={s:winner_summary['slots'][s]['coverage']['windows'] for s in epochs})
    task['output_dir'] = f'resultados/{task["station"]}/nucleo2/{task["model"]}/{task["branch"]}/final/{selected["candidate"]["id"]}'
    return task


def resolve_config(path):
    config = json.loads(Path(path).read_text(encoding='utf8'))
    protocol = Protocol.from_dict({**config.get('period',{}),**config.get('protocol',{}),
        'piecewise_look_back_days':config['regression']['piecewise_look_back_days'],
        's4_threshold':config['classification']['target_threshold_s4'],
        'temporal_gap':config['classification']['temporal_gap']})
    expected = {'time_resolution':'15min','forecast_minutes':15,'forecast_horizon_steps':1,
                'rolling_window_steps':12,'daily_lag_steps':96}
    if any(config['preprocessing'].get(k)!=v for k,v in expected.items()):
        raise ValueError('Configuration does not match frozen 15-minute preparation')
    regression = config['regression']
    expected_regression = {'look_back':96,'lstm_units':31,'lstm_layers':3,'dropout':.2,'epochs':50,
        'batch_size':400,'early_stopping_patience':3,'convlstm_epochs':60,'convlstm_batch_size':64,
        'convlstm_early_stopping_patience':8,'loss_masking':False,
        'learning_rate_candidates':[c['learning_rate'] for c in r.candidates('M1')],
        'convlstm_candidates':[{k:v for k,v in c.items() if k!='id'} for c in r.candidates('M9')]}
    if any(regression.get(k)!=v for k,v in expected_regression.items()):
        raise ValueError('Configuration changes approved candidate or architecture budgets')
    if [m['id'] for m in regression['architectures']]!=[f'M{i}' for i in range(1,13)]:
        raise ValueError('Configuration must retain M1 through M12')
    if config['resources']!={'minimum_free_gib':5,'training_workers':1,'threads':2}:
        raise ValueError('Configuration requires sequential workers and two threads')
    return config,protocol


def source_hashes(config, stations):
    result = {}
    def resolve(value, base=PROJECT):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (base/path).resolve()
    for station in stations:
        for label,value in config['inputs']['clean_sources'][station].items():
            result[f'inputs/{station}/{label}'] = w.sha256(resolve(value))
    data_root = resolve(config['inputs']['data_root'])
    for i,value in enumerate(config['inputs']['space_weather_files']):
        result[f'inputs/weather/{i}'] = w.sha256(resolve(value,data_root))
    return result


def update_counts(state):
    matrix = matrix_counts(state['stations'])
    for row in state['tasks'].values():
        if row.get('station') not in state['stations'] or row.get('phase') not in ('search','final'):
            continue
        if row.get('pilot'):
            continue
        for slot in row.get('slots',{}).values():
            matrix[f'{row["phase"]}_networks_{slot["status"]}'] += 1
    # Search-blocked slots absent from frozen final requests still count.
    for selection in state['selections'].values():
        if selection['station'] in state['stations']:
            matrix['final_networks_blocked'] += len(selection.get('blocked_slots',{}))*2
    matrix['final_networks_pending'] = matrix['max_final_networks']-matrix['final_networks_complete']-matrix['final_networks_blocked']
    state['matrix'] = matrix


def report(root, state):
    module = PROJECT/'scripts/duplo_nucleo/relatorio.py'
    if not module.is_file():
        state['reporting'] = {'status':'pending_module'}
        save_state(root,state)
        return False
    state['stage'] = 'reporting'
    state['reporting'] = {'status':'running','code_sha256':w.sha256(module)}
    save_state(root,state)
    execute_child([sys.executable,'-m','scripts.duplo_nucleo.relatorio','--output-root',str(root)],
                  root,state,'report',Path(root)/'logs/report.log')
    state['reporting'].update(status='complete',finished_at=now())
    save_state(root,state)
    return True


def _run(args):
    segment_start = time.perf_counter()
    segment_started_at = now()
    config_path = Path(args.config).resolve()
    config,protocol = resolve_config(config_path)
    stations = args.stations or config['stations']
    if len(stations)!=len(set(stations)) or any(s not in config['stations'] for s in stations):
        raise ValueError('Stations must be a unique subset of configured stations')
    root = Path(args.output_root or PROJECT/config['outputs']['experiment_root']).resolve()
    require_disk(root)
    # Fingerprint all configured inputs, so an incremental station subset can
    # extend the same experiment without weakening code/data rejection.
    inputs = source_hashes(config,config['stations'])
    fingerprint = scientific_fingerprint(config,inputs)
    root.mkdir(parents=True,exist_ok=True)
    state = load_state(root,fingerprint,protocol,stations,args.resume)
    state.update(input_hashes=inputs,code_hashes={p:w.sha256(PROJECT/p) for p in SCIENTIFIC_CODE},
                 scientific_config=config,free_bytes_preflight=require_disk(root))
    atomic_json(root/'resolved_config.json',{**config,'protocol':protocol.to_dict()})
    save_state(root,state)
    if args.preflight_only:
        state.update(status='preflight_complete',stage='preflight',pid=None,activity_verified=False)
        state.setdefault('execution_segments',[]).append({'started_at':segment_started_at,'finished_at':now(),
            'wall_seconds':time.perf_counter()-segment_start,'mode':'preflight','status':state['status']})
        save_state(root,state)
        return state
    state.update(status='running',stage='preparation',pid=os.getpid(),activity_verified=True)
    save_state(root,state)
    try:
        prepared = {}
        for station in stations:
            command = [sys.executable,str(Path(__file__).resolve()),'--config',str(config_path),
                       '--output-root',str(root),'--prepare-station',station]
            if args.resume:
                command.append('--resume')
            execute_child(command,root,state,f'prepare_{station}',root/f'logs/prepare_{station}.log')
            manifest_path = root/f'dados_processados/{station}/preparation_manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf8'))
            prepared[station] = {k:str(w.relative_path(root,v['path'])) for k,v in manifest['artifacts'].items()}
            prepared[station]['metadata'] = str(manifest_path)
            state['tasks'][f'prepare_{station}'].update(manifest=manifest_path.relative_to(root).as_posix(),
                                                        manifest_sha256=w.sha256(manifest_path))
            save_state(root,state)
        if args.prepare_only:
            state.update(status='preparation_complete',stage='preparation')
        elif args.pilot:
            state['stage'] = 'pilot_search'
            measured = []
            for station in stations:
                for model in ('M1','M5','M9','M11'):
                    for branch in r.BRANCHES:
                        candidate = r.candidates(model)[0]
                        task = make_task(root,station,model,branch,'search',candidate,protocol,prepared[station],pilot=True)
                        key = f'pilot_{station}_{model}_{branch}_{candidate["id"]}'
                        summary = execute_worker(task,root,state,key,args.resume)
                        state['tasks'][key]['pilot'] = True
                        measured.append({'task':key,'slots':{s:row['resources'] for s,row in summary['slots'].items()},
                                         'complete':sum(v['status']=='complete' for v in summary['slots'].values()),
                                         'blocked':sum(v['status']=='blocked' for v in summary['slots'].values())})
            state['pilot'] = {'status':'complete','measured_tasks':measured,
                'planned_networks':len(stations)*8,'complete_networks':sum(x['complete'] for x in measured),
                'blocked_networks':sum(x['blocked'] for x in measured),
                'assumptions':'first approved candidate; original/filtered M1/M5/M9/M11; piecewise slot0; full epoch ceilings/patience; isolated pilot paths',
                'full_run_estimate':None,'estimate_reason':'No measured duration is assigned to unexecuted matrix tasks.'}
            state.update(status='pilot_complete',stage='pilot_search')
        else:
            for station in stations:
                for model in (f'M{i}' for i in range(1,13)):
                    state['stage'] = 'search'
                    summaries = []; tasks = {}
                    for candidate in r.candidates(model):
                        for branch in r.BRANCHES:
                            task = make_task(root,station,model,branch,'search',candidate,protocol,prepared[station])
                            key = f'{station}_{model}_{branch}_search_{candidate["id"]}'
                            summaries.append(execute_worker(task,root,state,key,args.resume))
                            tasks[(candidate['id'],branch)] = task
                    denominator = [str(s) for s in tasks[next(iter(tasks))]['slots']]
                    blocked = {slot:[{'candidate':s['candidate'],'branch':s['branch'],
                                      'reason':s['slots'][slot].get('reason'),'coverage':s['slots'][slot].get('coverage')}
                                     for s in summaries if s['slots'][slot]['status']=='blocked'] for slot in denominator}
                    blocked = {slot:rows for slot,rows in blocked.items() if rows}
                    # Coverage is candidate-independent; inconsistent blocked
                    # masks are a scientific failure, never a smaller comparison.
                    if any(len(rows)!=len(summaries) for rows in blocked.values()):
                        raise ValueError('Candidate/branch coverage impediments do not coincide')
                    selection = {'station':station,'model':model,'slot_denominator':len(denominator),
                                 'blocked_slots':blocked,'summary_paths':[t['output_dir']+'/summary.json' for t in tasks.values()]}
                    if len(blocked)==len(denominator):
                        selection.update(status='blocked',reason='No paired development support for this architecture')
                    else:
                        selected = r.select_candidate(summaries)
                        selection.update(status='selected',**selected)
                    selection_path = root/f'resultados/{station}/nucleo2/{model}/selection.json'
                    atomic_json(selection_path,selection)
                    state['selections'][f'{station}_{model}'] = selection
                    update_counts(state); save_state(root,state)
                    if selection['status']=='blocked':
                        continue
                    state['stage'] = 'final'
                    for branch in r.BRANCHES:
                        winning = next(s for s in summaries if s['candidate']==selected['candidate'] and s['branch']==branch)
                        task = final_task(tasks[(selected['candidate']['id'],branch)],selected,winning)
                        execute_worker(task,root,state,f'{station}_{model}_{branch}_final_{selected["candidate"]["id"]}',args.resume)
                        update_counts(state); save_state(root,state)
            state.update(status='training_complete',stage='final')
        update_counts(state)
        completion = state['status']
        state['scientific_status'] = completion
        save_state(root,state)
        has_report = report(root,state)
        state.update(status=completion if args.pilot or args.prepare_only else
                     ('complete_with_blocks' if state['matrix']['final_networks_blocked'] else 'complete') if has_report else 'awaiting_report',
                     stage='finished',pid=None,child_pid=None,activity_verified=False,finished_at=now())
        state.setdefault('execution_segments',[]).append({'started_at':segment_started_at,'finished_at':now(),
            'wall_seconds':time.perf_counter()-segment_start,
            'mode':'pilot' if args.pilot else 'preparation' if args.prepare_only else 'full',
            'status':state['status']})
        save_state(root,state)
        return state
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                     pid=None,child_pid=None,activity_verified=False,error=f'{type(exc).__name__}: {exc}')
        state['failures'].append({'stage':state['stage'],'at':now(),'error':state['error']})
        state.setdefault('execution_segments',[]).append({'started_at':segment_started_at,'finished_at':now(),
            'wall_seconds':time.perf_counter()-segment_start,'mode':'failed_attempt','status':state['status']})
        update_counts(state); save_state(root,state)
        raise


def run(args):
    config, _ = resolve_config(Path(args.config).resolve())
    root = Path(args.output_root or PROJECT/config['outputs']['experiment_root']).resolve()
    require_disk(root)
    root.mkdir(parents=True,exist_ok=True)
    with runner_lock(root):
        return _run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(PROJECT/'configs/duplo_nucleo_15min.json'))
    parser.add_argument('--output-root')
    parser.add_argument('--stations',nargs='+')
    parser.add_argument('--resume',action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight-only',action='store_true')
    modes.add_argument('--prepare-only',action='store_true')
    modes.add_argument('--pilot',action='store_true')
    parser.add_argument('--prepare-station',choices=('FRTZ','PALM','PRU2','POAL'),help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare_station:
        from scripts.duplo_nucleo.preparacao import prepare_station
        if not args.output_root:
            parser.error('--prepare-station requires explicit --output-root')
        result = prepare_station(args.config,args.prepare_station,args.output_root,resume=args.resume)
        print(json.dumps(result,ensure_ascii=False))
    else:
        state = run(args)
        print(json.dumps({'status':state['status'],'stage':state['stage'],
                          'runtime_output_root':state['runtime_output_root'],'matrix':state['matrix']},ensure_ascii=False))


if __name__ == '__main__':
    main()
