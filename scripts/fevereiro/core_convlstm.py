"""ConvLSTM: desenvolvimento até 16/02 e teste comum em 18/02/2024."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'estudo_20240218'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)


def scores(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float);mask=np.isfinite(y)&np.isfinite(p)
    if not mask.any():return {'n':0,'RMSE':None,'MAE':None,'R2':None}
    y,p=y[mask],p[mask];e=p-y;den=np.sum((y-y.mean())**2)
    return dict(n=len(y),RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(abs(e))),R2=float(1-np.sum(e**2)/den) if den else None)


def prepare(frame,observed,model,variant,phase,slot=None):
    if phase not in ('search','final'):raise ValueError('Fase inválida')
    number=int(model[1:]);multi=number in (2,4,6,8,9,12);piecewise=number in (5,6,7,8,11,12)
    f=frame.copy();f['datetime']=pd.to_datetime(f.datetime)
    assert f.datetime.is_monotonic_increasing and not f.datetime.duplicated().any()
    end=pd.Timestamp('2024-02-16 23:30')
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


def spec(model,rate=.001,filters=16):
    number=int(model[1:])
    if number>=9:return dict(layers=1,units=filters,filters=filters,learning_rate=rate,batch_size=64,epochs=60,patience=8,clipnorm=1.,dropout=.2,seed=42,architecture='convlstm')
    return dict(layers=3,units=31,learning_rate=rate,batch_size=400,epochs=50,patience=3,clipnorm=None,dropout=.2,seed=42,architecture='attention' if number in (3,4,7,8) else 'lstm')


def build(tf,shape,s):
    L=tf.keras.layers;inputs=L.Input(shape=shape);h=inputs
    if s['architecture']=='convlstm':
        blocks,width=(5,3) if shape[0]==15 else (8,6)
        assert shape[0]==blocks*width
        h=L.Reshape((blocks,width,shape[1]))(h)
        h=L.ConvLSTM1D(s['filters'],3,padding='same',activation='tanh',recurrent_activation='sigmoid')(h)
        h=L.Dense(32,activation='relu')(L.Flatten()(h));h=L.Dropout(.2)(h)
        delta=L.Dense(1,kernel_initializer='zeros',bias_initializer='zeros')(h)
        output=L.Add()([inputs[:,-1,:1],delta])
    else:
        attention=s['architecture'] in ('attention','residual')
        for i in range(s['layers']):h=L.LSTM(s['units'],return_sequences=attention or i<s['layers']-1)(h)
        if attention:
            logits=L.Dense(1,activation='linear' if s['architecture']=='residual' else 'tanh')(h)
            context=L.Flatten()(L.Dot(axes=1)([L.Softmax(axis=1)(logits),h]))
            h=L.Concatenate()([context,h[:,-1,:]]) if s['architecture']=='residual' else context
        h=L.Dropout(s['dropout'])(h)
        if s['architecture']=='residual':
            output=L.Add()([inputs[:,-1,:1],L.Dense(1,kernel_initializer='zeros',bias_initializer='zeros')(h)])
        else:output=L.LeakyReLU(negative_slope=.3)(L.Dense(1)(h))
    model=tf.keras.Model(inputs,output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=s['learning_rate'],clipnorm=s['clipnorm']),loss='mse')
    return model
