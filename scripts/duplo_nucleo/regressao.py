"""15-minute regression contracts, paired coverage and finite model selection.

Inputs retain retrospectively interpolated historical values. Availability checks
describe nominal aggregates, and are not a claim of online causal interpolation.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from scripts.duplo_nucleo.protocolo import Protocol
from scripts.fevereiro import core_convlstm as historical

WEATHER = ['Kp', 'Dst', 'F107', 'Ap', 'SSN']
BRANCHES = ('original', 'filtered')


class InsufficientDevelopmentSupport(ValueError):
    """Recoverable missing finite development support for fitting scalers."""


def identity(model):
    if model not in {f'M{i}' for i in range(1, 13)}:
        raise ValueError('Unknown model identity')
    n = int(model[1:])
    return n, n in (2,4,6,8,9,12), n in (5,6,7,8,11,12)


def candidates(model):
    n, _, _ = identity(model)
    if n < 9:
        return [{'id': f'lr_{rate:g}', 'learning_rate': rate}
                for rate in [.05, .005, .0005, .00005, .000005, .0000005]]
    return [{'id':'f16_lr0.001','filters':16,'learning_rate':.001},
            {'id':'f32_lr0.0003','filters':32,'learning_rate':.0003}]


def model_spec(model, candidate):
    return historical.spec(model, rate=candidate['learning_rate'], filters=candidate.get('filters',16))


def build_model(tf, shape, spec):
    if spec['architecture'] != 'convlstm' or shape[0] == 15:
        return historical.build(tf, shape, spec)
    if shape[0] != 96:
        raise ValueError('Conventional ConvLSTM requires96 past15min bins')
    # Preserve historical ConvLSTM residual identity; explicit8x12 replaces8x6.
    L = tf.keras.layers
    inputs = L.Input(shape=shape)
    h = L.Reshape((8,12,shape[1]))(inputs)
    h = L.ConvLSTM1D(spec['filters'],3,padding='same',activation='tanh',recurrent_activation='sigmoid')(h)
    h = L.Dense(32,activation='relu')(L.Flatten()(h))
    h = L.Dropout(spec['dropout'])(h)
    delta = L.Dense(1,kernel_initializer='zeros',bias_initializer='zeros')(h)
    model = tf.keras.Model(inputs,L.Add()([inputs[:,-1,:1],delta]))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=spec['learning_rate'],clipnorm=spec['clipnorm']),loss='mse')
    return model


def default_windows(protocol):
    return {k:getattr(protocol,k) for k in ('fit_end','early_stopping_start','early_stopping_end','selection_start','selection_end')}


def prepare_sequences(frame, observed, model, branch, protocol, phase='search', slot=None, windows=None):
    n, multi, piece = identity(model)
    if branch not in BRANCHES or phase not in ('search','final'):
        raise ValueError('Invalid branch or phase')
    if not isinstance(protocol, Protocol):
        raise TypeError('Fully resolved Protocol required')
    f = frame.copy(); f['datetime'] = pd.to_datetime(f.datetime)
    if f.datetime.duplicated().any() or not f.datetime.is_monotonic_increasing:
        raise ValueError('Unique sorted regular15min grid required')
    if len(f)>1 and not (f.datetime.diff().dropna() == pd.Timedelta(protocol.time_resolution)).all():
        raise ValueError('Regular15min grid required, do not compress missing bins')
    f = f.loc[(f.datetime>=pd.Timestamp(protocol.study_start)) &
              (f.datetime<=pd.Timestamp(protocol.train_end if phase=='search' else protocol.test_end))].copy()
    if piece:
        if slot is None or not 0 <= slot < protocol.slot_count:
            raise ValueError('Piecewise slot0..95 required')
        f = f.loc[f.datetime.dt.hour*4+f.datetime.dt.minute//15 == slot].copy()
    common = observed.reindex(pd.DatetimeIndex(f.datetime)).to_numpy(float)
    col = f'VTEC_{branch}'
    cols = [col] + (WEATHER if multi else [])
    raw_x = f[cols].to_numpy(float)
    raw_y = common.copy() if n>=9 else f[col].to_numpy(float)
    times = f.datetime.to_numpy(dtype='datetime64[ns]')
    lb = protocol.piecewise_look_back_days if piece else protocol.look_back
    step = np.timedelta64(1,'D') if piece else np.timedelta64(15,'m')
    available = times + np.timedelta64(15,'m')
    if multi and 'weather_available_at' in f:
        weather = pd.to_datetime(f.weather_available_at).to_numpy(dtype='datetime64[ns]')
        available = np.maximum(available,weather)
    idx = np.asarray([i for i in range(lb,len(f))
                      if times[i]-times[i-lb] == lb*step and np.isfinite(raw_x[i-lb:i]).all()
                      and (available[i-lb:i]<=times[i]).all()],dtype=int)
    w = default_windows(protocol) if windows is None else windows
    dates = [pd.Timestamp(w[k]) for k in ('fit_end','early_stopping_start','early_stopping_end','selection_start','selection_end')]
    if any(a>=b for a,b in zip(dates,dates[1:])) or dates[-1]>pd.Timestamp(protocol.train_end):
        raise ValueError('Search windows must be ordered inside development')
    cutoff = pd.Timestamp(protocol.train_end if phase=='final' else w['fit_end'])
    fit_rows = (times<=np.datetime64(cutoff)) & np.isfinite(raw_y) & np.isfinite(common)
    x_rows = (times<=np.datetime64(cutoff)) & np.isfinite(raw_x).all(axis=1)
    if not fit_rows.any() or not x_rows.any():
        raise InsufficientDevelopmentSupport('No finite development data to fit scalers')
    sx, sy = MinMaxScaler(), MinMaxScaler()
    sx.fit(raw_x[x_rows]); sy.fit(raw_y[fit_rows,None])
    # Residual ConvLSTM must see VTEC in the target scaler's coordinate system.
    for attribute in ('scale_','min_','data_min_','data_max_','data_range_'):
        getattr(sx,attribute)[0] = getattr(sy,attribute)[0]
    xx = sx.transform(raw_x).astype('float32')
    yy = sy.transform(raw_y[:,None]).ravel().astype('float32')
    t = times[idx]; valid = np.isfinite(common[idx]) & np.isfinite(raw_y[idx])
    zero = np.zeros(len(idx),bool)
    masks = {'fit':(t<=np.datetime64(cutoff)) & valid,
             'early':(t>=np.datetime64(w['early_stopping_start'])) & (t<=np.datetime64(w['early_stopping_end'])) & valid if phase=='search' else zero.copy(),
             'selection':(t>=np.datetime64(w['selection_start'])) & (t<=np.datetime64(w['selection_end'])) & valid if phase=='search' else zero.copy(),
             'test':(t>=np.datetime64(protocol.test_start)) & (t<=np.datetime64(protocol.test_end)) if phase=='final' else zero.copy()}
    partition = {k:{'n':int(m.sum()),'start':str(pd.Timestamp(t[m][0])) if m.any() else None,
                    'end':str(pd.Timestamp(t[m][-1])) if m.any() else None} for k,m in masks.items()}
    return {'X':np.asarray([xx[i-lb:i] for i in idx],dtype='float32').reshape((-1,lb,len(cols))),
            'y':yy[idx], 'raw_y':raw_y[idx], 'common':common[idx], 'times':t,
            'input_times':np.asarray([times[i-lb:i] for i in idx]).reshape((-1,lb)),
            'input_available_at':np.asarray([available[i-lb:i].max() for i in idx]),
            'persistence':raw_x[idx-1,0], 'sx':sx,'sy':sy,'cols':cols,'partition':partition,**masks}


def paired_windows(frame, observed, model, protocol, slot=None):
    """Move paired weekly validation windows backwards by coverage only."""
    start = pd.Timestamp(protocol.selection_start)
    stop = pd.Timestamp(protocol.selection_end)
    attempts = []
    while start>=pd.Timestamp(protocol.study_start)+pd.Timedelta('14d'):
        early_start = start-pd.Timedelta('7d')
        w = {'fit_end':str(early_start-pd.Timedelta('1s')), 'early_stopping_start':str(early_start),
             'early_stopping_end':str(start-pd.Timedelta('1s')), 'selection_start':str(start), 'selection_end':str(stop)}
        try:
            pair = [prepare_sequences(frame,observed,model,b,protocol,slot=slot,windows=w) for b in BRANCHES]
            support = {k:len(set(pair[0]['times'][pair[0][k]]) & set(pair[1]['times'][pair[1][k]]))
                       for k in ('fit','early','selection')}
        except InsufficientDevelopmentSupport as exc:
            support = {'fit':0,'early':0,'selection':0}
            attempts.append({'windows':w,'support':support,'reason':str(exc)})
        else:
            attempts.append({'windows':w,'support':support})
            if support['fit']>=10 and support['early']>=2 and support['selection']>=2:
                return {'eligible':True,'reason':None,'windows':w,'support':support,'attempts':attempts}
        start -= pd.Timedelta('7d'); stop -= pd.Timedelta('7d')
    return {'eligible':False,'reason':'No paired development coverage meeting10/2/2 minima',
            'windows':None,'support':attempts[-1]['support'] if attempts else {},'attempts':attempts}


def select_candidate(summaries):
    """Use the same observed target support for every candidate and both branches."""
    grouped = {}
    for s in summaries:
        key = s['candidate']['id']
        if s['branch'] in grouped.setdefault(key,{}):
            raise ValueError('Duplicate candidate/branch')
        grouped[key][s['branch']] = s
    support = None; common_reference = None; scores = []; freezes = {}
    for cid, branches in grouped.items():
        if set(branches)!=set(BRANCHES):
            raise ValueError('Each candidate requires both branches')
        by_branch = {}; frozen = {}
        for branch,s in branches.items():
            predictions = {}; epochs = {}
            for slot,row in s['slots'].items():
                if row['status']!='complete': continue
                selection = row['selection']
                for t,y,p in zip(selection['times'],selection['common'],selection['prediction']):
                    if np.isfinite(y) and np.isfinite(p): predictions[(slot,t)] = (y,p)
                epochs[slot] = row['epochs_selected']
            by_branch[branch] = predictions; frozen[branch] = epochs
        shared = set(by_branch['original']) & set(by_branch['filtered'])
        if not shared: raise ValueError('No paired selection support')
        if support is None: support = shared
        elif support != shared: raise ValueError('Candidate selection support mismatch')
        reference = {k:by_branch['original'][k][0] for k in shared}
        if common_reference is None: common_reference = reference
        elif common_reference != reference: raise ValueError('Common reference differs between candidates')
        rmse = {}
        for branch,preds in by_branch.items():
            rmse[branch] = float(np.sqrt(np.mean([(preds[k][1]-preds[k][0])**2 for k in shared])))
        for k in shared:
            if by_branch['original'][k][0] != by_branch['filtered'][k][0]:
                raise ValueError('Common reference differs between branches')
        scores.append({'candidate':branches['original']['candidate'],'paired_n':len(shared),
                       'rmse_by_branch':rmse,'mean_paired_rmse':float(np.mean(list(rmse.values())))})
        freezes[cid] = frozen
    if not scores: raise ValueError('No candidates')
    winner = min(scores,key=lambda x:(x['mean_paired_rmse'],x['candidate']['id']))
    return {**winner,'epochs_by_branch':freezes[winner['candidate']['id']], 'scores':scores}
