"""Continua etapas existentes, com logs separados e retomada verificável."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'estudo_20240218'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stations', nargs='+', default=['FRTZ', 'PALM', 'PRU2', 'POAL'])
    parser.add_argument('--include-convlstm', action='store_true')
    args = parser.parse_args()
    status_path = OUT / 'status_retomada.json'
    state = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'stations': {}}
    state.update(status='executando', test_date='2024-02-18', updated_at=now())
    save(status_path, state)
    env = os.environ.copy()
    env.update(GNSS_LSTM_RESUME='1', PYTHONUTF8='1')
    for station in args.stations:
        assert station in ('FRTZ', 'PALM', 'POAL', 'PRU2')
        cfg = ROOT / 'configs' / f'xie_{station.lower()}_comum_18fev2024.json'
        base = ROOT / 'experimentos' / f'{station.lower()}_comum_18fev2024'
        prepared = base / 'dados_processados' / station / f'{station}_dataset_regressao.csv'
        # A preparação existente é mantida. O treinador verifica dataset,
        # código científico, protocolo e hashes de cada checkpoint reutilizado.
        stages = ['10_train_lstm.py'] if prepared.exists() else [
            '01_validar_insumos.py', '05_integrar_clima_espacial.py',
            '02_localizar_arquivos_ismr.py', '03_extrair_ismr.py',
            '04_controle_qualidade_eda.py', '06_feature_engineering.py',
            '07_dataset_classificacao.py', '08_train_xgboost.py',
            '09_filtrar_vtec.py', '10_train_lstm.py']
        try:
            for stage in stages:
                record = {'status': 'executando', 'stage': stage, 'updated_at': now()}
                state['stations'][station] = record
                save(status_path, state)
                log = OUT / 'logs' / f'{station}_{Path(stage).stem}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                record['log'] = str(log)
                command = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'scripts' / stage),
                           '--config', str(cfg), '--output-root', str(base), '--station', station]
                print('INICIO', station, stage, now(), flush=True)
                with log.open('a', encoding='utf-8') as stream:
                    stream.write('\nRETOMADA ' + now() + '\n'); stream.flush()
                    subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                print('FIM', station, stage, now(), flush=True)
            state['stations'][station] = {'status': 'lstm_concluido', 'updated_at': now()}
        except Exception as exc:
            state['stations'][station].update(status='erro', error=str(exc), updated_at=now())
            save(status_path, state)
            print('ERRO', station, str(exc), flush=True)
            # Uma estação com problema não impede o avanço das demais.
        save(status_path, state)
    if args.include_convlstm:
        for station in args.stations:
            if state['stations'][station]['status'] != 'lstm_concluido':
                continue
            state['stations'][station].update(status='convlstm_executando', updated_at=now())
            save(status_path, state)
            try:
                subprocess.run([sys.executable, '-X', 'utf8', '-u',
                                str(ROOT / 'scripts' / 'fevereiro' / 'treinar_convlstm.py'),
                                '--stations', station], cwd=ROOT, check=True)
                state['stations'][station].update(status='modelos_concluidos', updated_at=now())
            except Exception as exc:
                state['stations'][station].update(status='erro_convlstm', error=str(exc), updated_at=now())
            save(status_path, state)
    has_errors = any(v['status'].startswith('erro') for v in state['stations'].values())
    state.update(status='erro_parcial' if has_errors else 'concluido', updated_at=now())
    save(status_path, state)
    if has_errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
