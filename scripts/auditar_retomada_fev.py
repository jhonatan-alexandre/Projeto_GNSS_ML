"""Diagnóstico somente de leitura dos checkpoints do fluxo de fevereiro."""
import importlib.util
import json
from pathlib import Path
import pandas as pd

script = Path(__file__).with_name('10_train_lstm.py')
spec = importlib.util.spec_from_file_location('lstm_audit', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
folder = Path(module.PASTA_RESULTADOS)
rate_path = folder / 'hiperparametros_lstm.json'
report = {'station': module.STATION, 'dataset': module.DATASET, 'loss_masking': module.LOSS_MASKING}
if Path(module.DATASET).exists():
    frame = pd.read_csv(module.DATASET, parse_dates=['datetime'])
    report['coverage'] = {'first': str(frame.datetime.min()), 'last': str(frame.datetime.max()),
                          'test_18_rows': int(frame.datetime.dt.strftime('%Y-%m-%d').eq('2024-02-18').sum())}
if rate_path.exists():
    rate = json.loads(rate_path.read_text(encoding='utf-8'))['hiperparametros']['learning_rate']
    expected = module._proveniencia(rate)
    records = []
    for path in folder.glob('checkpoint_*.json'):
        item = json.loads(path.read_text(encoding='utf-8'))
        actual = item['proveniencia']
        differences = {key: {'expected': value, 'actual': actual.get(key)} for key,value in expected.items() if actual.get(key) != value}
        records.append({'file': path.name, 'differences': differences,
                        'artifacts_valid': all(module._artefato_aprovado(p,item['artefatos']) for p in item['artefatos'])})
    report['checkpoints'] = records
print(json.dumps(report, ensure_ascii=False, indent=2))
