"""Executa somente PRU2 com configuração explícita do relatório, em saída isolada."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'experimentos/pru2_relatorio_dez2023_mar2024_20260923'
CONFIG=DEST/'config_pru2_relatorio.json'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    c=json.loads(CONFIG.read_text(encoding='utf-8'))
    assert c['stations']==['PRU2'] and c['classification']['filter_probability_threshold']==.75
    assert not c['classification']['tune_hyperparameters']
    for k,v in dict(train_fraction=.8,lstm_units=64,lstm_layers=2,learning_rate=.001,epochs=50,batch_size=400,early_stopping_patience=5,output_activation='linear').items():assert c['regression'][k]==v
    assert len(c['regression']['architectures'])==8 and c['regression']['learning_rate_candidates']==[]
    assert c['period']['test_start']=='2024-03-28 00:00:00'
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('*.py')}
    sources[str(CONFIG.relative_to(ROOT))]=sha(CONFIG)
    sources[str(Path(__file__).resolve().relative_to(ROOT))]=sha(Path(__file__).resolve())
    env={k:v for k,v in os.environ.items() if not k.upper().startswith('GNSS_')}
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',GNSS_MIN_FREE_SPACE_GB='5',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    command=[sys.executable,'-X','utf8','-u',str(ROOT/'scripts/12_multistation_pipeline.py'),
        '--config',str(CONFIG),'--output-root',str(DEST),'--stations','PRU2']
    if args.preflight_only:command.append('--preflight-only')
    else:
        marker=DEST/'execucao_iniciada.json'
        if marker.exists():raise RuntimeError('Execução já iniciada: não sobrescrever artefatos. Conferir estado antes de retomar.')
        marker.write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'sources_sha256':sources,'command':command},ensure_ascii=False,indent=2),encoding='utf-8')
    result=subprocess.run(command,cwd=ROOT,env=env)
    if result.returncode:raise SystemExit(result.returncode)
    if args.preflight_only:return
    for p,h in sources.items():
        if sha(ROOT/p)!=h:raise RuntimeError(f'Fonte mudou durante execução: {p}')
    spec=importlib.util.spec_from_file_location('existing_checks',ROOT/'scripts/14_retreinar_xie.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    errors=module.completeness_errors(DEST,'PRU2',c)
    artifact_hashes={}
    for checkpoint in (DEST/'resultados/PRU2/nucleo2').glob('checkpoint_m*.json'):
        item=json.loads(checkpoint.read_text(encoding='utf-8'))
        # O formato de checkpoint já é validado pelo treinador; registra também os arquivos finais.
        artifact_hashes[str(checkpoint.relative_to(DEST))]=sha(checkpoint)
    for p in (DEST/'modelos/PRU2').rglob('*'):
        if p.is_file():artifact_hashes[str(p.relative_to(DEST))]=sha(p)
    networks=len(list((DEST/'modelos/PRU2/nucleo2').rglob('*.keras')))
    if networks!=392:errors.append(f'Esperadas 392 redes, encontradas {networks}')
    (DEST/'auditoria_final.json').write_text(json.dumps({'status':'completo' if not errors else 'erro','errors':errors,'networks':networks,'sources_sha256':sources,'artifacts_sha256':artifact_hashes,'finished_utc':datetime.now(timezone.utc).isoformat()},ensure_ascii=False,indent=2),encoding='utf-8')
    if errors:raise RuntimeError('; '.join(errors))
    print('PRU2 concluída e auditada: 16 configurações, 392 redes.',flush=True)


if __name__=='__main__':main()
