import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from core_convlstm import prepare,build,write,sha,scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);a=p.parse_args()
    task=json.loads(a.task.read_text(encoding='utf-8'));dest=Path(task['output']);dest.mkdir(parents=True,exist_ok=True)
    for path,digest in task['hashes'].items():assert sha(path)==digest,path
    frame=pd.read_csv(task['dataset'],parse_dates=['datetime'])
    obs=pd.read_csv(task['observed'],parse_dates=['datetime']).set_index('datetime').VTEC_observado
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    slots=range(48) if int(task['model'][1:]) in (5,6,7,8,11,12) else [None]
    records=[];predictions=[];artifacts={};started=time.monotonic()
    options=tf.data.Options();options.threading.private_threadpool_size=1;options.threading.max_intra_op_parallelism=1
    for slot in slots:
        key='all' if slot is None else str(slot);prefix=dest/key;prefix.mkdir(exist_ok=True)
        cached=prefix/'historico.json'
        if cached.exists():
            h=json.loads(cached.read_text(encoding='utf-8'));assert h['task_sha256']==sha(a.task)
            for path,digest in h['artifacts'].items():assert sha(path)==digest
            records.append(h);predictions.append(pd.read_csv(prefix/'predicoes.csv',parse_dates=['datetime']));artifacts.update(h['artifacts']);continue
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42)
        data=prepare(frame,obs,task['model'],task['variant'],task['phase'],slot)
        spec=task['spec'];model=build(tf,data['X'].shape[1:],spec)
        def ds(mask):return tf.data.Dataset.from_tensor_slices((data['X'][mask],data['y'][mask])).batch(min(spec['batch_size'],int(data['fit'].sum()))).with_options(options).prefetch(1)
        callbacks=[tf.keras.callbacks.TerminateOnNaN()];kwargs={}
        if task['phase']=='search':
            epochs=spec['epochs'];callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=spec['patience'],restore_best_weights=True))
            kwargs['validation_data']=ds(data['early'])
        else:epochs=task['fixed_epochs'][key]
        history=model.fit(ds(data['fit']),epochs=epochs,callbacks=callbacks,verbose=2 if slot is None else 0,**kwargs).history
        assert all(np.isfinite(v).all() for v in history.values())
        if task['phase']=='final':assert len(history['loss'])==epochs
        mask=data['selection'] if task['phase']=='search' else data['test']
        normalized=np.asarray(model(data['X'][mask],training=False)).reshape(-1,1)
        pred=data['sy'].inverse_transform(normalized).ravel()
        q=pd.DataFrame({'datetime':data['times'][mask],'observado':data['raw_y'][mask],'observado_comum':data['common'][mask],'predito':pred,'persistencia':data['persistence'][mask],'slot':key})
        q.to_csv(prefix/'predicoes.csv',index=False)
        model.save(prefix/'modelo.keras');joblib.dump({'entrada':data['sx'],'alvo':data['sy'],'colunas':data['cols']},prefix/'escalas.joblib')
        loaded=tf.keras.models.load_model(prefix/'modelo.keras',compile=False)
        for expected_weight,loaded_weight in zip(model.get_weights(),loaded.get_weights(),strict=True):
            np.testing.assert_array_equal(expected_weight,loaded_weight)
        actual=np.asarray(loaded(data['X'][mask],training=False)).reshape(-1,1)
        np.testing.assert_allclose(actual,normalized,rtol=1e-6,atol=1e-7)
        hashes={str(p.resolve()):sha(p) for p in [prefix/'modelo.keras',prefix/'escalas.joblib',prefix/'predicoes.csv']}
        record={'slot':key,'task_sha256':sha(a.task),'spec':spec,'history':history,'epochs_executed':len(history['loss']),
                'best_epoch':int(np.argmin(history['val_loss']))+1 if task['phase']=='search' else epochs,
                'partition':data['partition'],'metrics_common':scores(q.observado_comum,q.predito),'metrics_own':scores(q.observado,q.predito),
                'reload_max_difference':float(np.max(abs(actual-normalized))),'reload_same_batch':True,
                'reload_weights_identical':True,'executed_worker':str(Path(__file__).resolve()),'executed_worker_sha256':sha(__file__),'artifacts':hashes}
        write(cached,record);records.append(record);predictions.append(q);artifacts.update(hashes)
        print('SLOT',task['station'],task['model'],task['variant'],task['phase'],key,record['metrics_common'],flush=True)
        del model,loaded,data;tf.keras.backend.clear_session();gc.collect()
    q=pd.concat(predictions,ignore_index=True).sort_values('datetime');q.to_csv(dest/'predicoes.csv',index=False)
    if task['phase']=='final':assert len(q)==48
    artifacts[str((dest/'predicoes.csv').resolve())]=sha(dest/'predicoes.csv')
    record={'station':task['station'],'model':task['model'],'variant':task['variant'],'phase':task['phase'],
            'task_sha256':sha(a.task),'metrics_common':scores(q.observado_comum,q.predito),'metrics_own':scores(q.observado,q.predito),
            'selected_epochs':{r['slot']:r['best_epoch'] for r in records},'duration_seconds':time.monotonic()-started,
            'all_development_refit':task['phase']=='final','executed_worker':str(Path(__file__).resolve()),'executed_worker_sha256':sha(__file__),'artifacts':artifacts}
    for path,digest in task['hashes'].items():assert sha(path)==digest,path
    write(dest/'resumo.json',record)
    print('CONCLUIDO',record['metrics_common'],flush=True)


if __name__=='__main__':
    main();sys.stdout.flush();sys.stderr.flush()
    if os.name=='nt':os._exit(0)
