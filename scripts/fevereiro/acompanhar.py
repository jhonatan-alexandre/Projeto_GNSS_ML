"""Estado legível e auditoria automática ao concluir as quatro estações."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from core_convlstm import ROOT,OUT,write
from lock import acquire, held


def read(path):
    if not path.exists(): return {}
    try: return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError: return {}


def state():
    rows=[];coordinator=read(OUT/'status_retomada.json')
    for station in ('FRTZ','PALM','POAL','PRU2'):
        base=ROOT/'experimentos'/f'{station.lower()}_comum_18fev2024'
        n1=(base/'resultados'/station/'nucleo1'/'predicoes_xgboost.csv').is_file()
        selection=len(list((base/'resultados'/station/'nucleo2').glob('checkpoint_m*.json')))
        final=OUT/'estacoes'/station
        active=held(final/'treinamento.lock')
        lstm=len(list((final/'lstm_integral'/'finais').glob('*/*/resumo.json')))
        conv=len(list((final/'convlstm'/'finais').glob('*/*/*/resumo.json')))
        networks=len(list((final/'lstm_integral'/'finais').glob('*/*/*/historico.json')))
        networks+=len(list((final/'convlstm'/'finais').glob('*/*/*/*/historico.json')))
        phase='na fila' if n1 else 'preparação'
        detail=coordinator.get('stations',{}).get(station,{})
        if detail: phase=detail.get('status','')+' '+detail.get('stage','')
        refit=read(final/'lstm_integral'/'status.json');search=read(final/'convlstm'/'status.json')
        if refit: phase='reajuste M1–M8: '+refit.get('status','')+' '+refit.get('model','')+' '+refit.get('variant','')
        if search: phase='ConvLSTM: '+search.get('status','')+' '+search.get('model','')+' '+search.get('variant','')+' '+search.get('phase','')
        if lstm+conv==24: phase='modelos concluídos; conferir auditoria'
        elif not active: phase='interrompido; última etapa: '+phase.strip()
        rows.append(dict(station=station,nucleo1=n1,selection_completed=selection,selection_expected=16,
                         lstm_final=lstm,convlstm_final=conv,configurations_final=lstm+conv,
                         configurations_expected=24,networks_final=networks,networks_expected=588,
                         process_active=active,phase=phase.strip()))
    complete=all(r['configurations_final']==24 and r['networks_final']==588 and r['nucleo1'] for r in rows)
    status='pronto_para_auditoria' if complete else 'executando' if any(r['process_active'] for r in rows) else 'interrompido'
    return dict(updated_at=datetime.now(timezone.utc).isoformat(),test_date='2024-02-18',
                status=status,stations=rows,coordinator_status=coordinator.get('status'))


def publish(current):
    write(OUT/'estado_execucao.json',current)
    lines=['# Estado da execução — 18/02/2024','',f"Atualização UTC: {current['updated_at']}",'',
           f"Estado: **{current['status']}**.",'',
           '| Estação | Núcleo 1 | Seleção M1–M8 | Configurações finais | Redes finais | Etapa |',
           '|---|---|---:|---:|---:|---|']
    for r in current['stations']:
        lines.append(f"| {r['station']} | {'Concluído' if r['nucleo1'] else 'Pendente'} | {r['selection_completed']}/16 | {r['configurations_final']}/24 | {r['networks_final']}/588 | {r['phase']} |")
    lines.extend(['','Processos ativos são verificados pelas travas do sistema; uma execução encerrada não é anunciada como ativa.',
                  'As contagens acompanham os arquivos concluídos. A conclusão científica exige também a auditoria de hashes, datas, partições e métricas em `artigo/auditoria.json`.'])
    (OUT/'STATUS_ATUAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
    if args.once:
        print(json.dumps(state(),ensure_ascii=False,indent=2));return
    monitor_lock=acquire(OUT/'monitor.lock');previous=None
    while True:
        current=state();signature=json.dumps(current['stations'],sort_keys=True)
        publish(current)
        if signature!=previous:
            print(current['updated_at'],' | '.join(f"{r['station']}: {r['configurations_final']}/24 finais; {r['phase']}" for r in current['stations']),flush=True)
            previous=signature
        if all(r['configurations_final']==24 and r['networks_final']==588 and r['nucleo1'] for r in current['stations']):
            log=OUT/'logs'/'auditoria_final.log';log.parent.mkdir(exist_ok=True)
            with log.open('a',encoding='utf-8') as stream:
                result=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('relatorio.py')),'--require-complete'],
                                      cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            current.update(status='concluido_verificado' if result.returncode==0 else 'erro_na_auditoria_final')
            publish(current)
            if result.returncode==0:
                doc=OUT/'README.md'
                text=doc.read_text(encoding='utf-8').replace('**A execução está em andamento.**','**Execução concluída e auditada; consulte `artigo/auditoria.json`.**')
                doc.write_text(text,encoding='utf-8')
            print(current['status'],flush=True);return
        if current['status']=='interrompido':
            publish(current)
            print(current['status'],flush=True);return
        time.sleep(60)


if __name__=='__main__': main()
