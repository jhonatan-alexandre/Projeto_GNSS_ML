"""Figuras auditáveis dos hiperparâmetros e escolhas efetivamente registradas."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from core_convlstm import ROOT, OUT, sha

STATIONS = ('FRTZ', 'PALM', 'POAL', 'PRU2')
DEST = OUT/'artigo/figuras/hiperparametros'
BLUE, GREEN, ORANGE, INK = '#176B96', '#14806A', '#CF7029', '#203448'
LABELS = ['LSTM · uni · conv.', 'LSTM · multi · conv.', 'Attention · uni · conv.',
          'Attention · multi · conv.', 'LSTM · uni · piecewise', 'LSTM · multi · piecewise',
          'Attention · uni · piecewise', 'Attention · multi · piecewise',
          'ConvLSTM · multi · conv.', 'ConvLSTM · uni · conv.',
          'ConvLSTM · uni · piecewise', 'ConvLSTM · multi · piecewise']


def save(fig, name):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(DEST/f'{name}.{extension}', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def number(value):
    if value is None or pd.isna(value): return '—'
    return f'{value:g}'.replace('.', ',')


def epoch_range(epochs):
    low, high = min(epochs.values()), max(epochs.values())
    return str(low) if low == high else f'{low}–{high}'


def panel_title(fig, y, title, subtitle):
    fig.text(.055, y, title, fontsize=15, weight='bold', color=INK)
    fig.text(.055, y-.022, subtitle, fontsize=10, color='#526577')


def card(ax, title, body, color):
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((.01,.02),.98,.95,boxstyle='round,pad=0.01,rounding_size=.025',
                              transform=ax.transAxes,facecolor=color,edgecolor='#D4DEE5'))
    ax.text(.055,.87,title,transform=ax.transAxes,va='top',fontsize=12,weight='bold',color=INK)
    ax.text(.055,.71,body,transform=ax.transAxes,va='top',fontsize=10.5,linespacing=1.6,color=INK)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.titlecolor':INK,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.grid':True,'grid.alpha':.17})
    sources = {}; classifiers = {}; lstm = {}; curves = {}; rows = []; summary_rows = []
    snapshot = datetime.now(timezone.utc).isoformat()
    def read_json(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))
    for station in STATIONS:
        base = ROOT/'experimentos'/f'{station.lower()}_comum_18fev2024'/'resultados'/station
        c = read_json(base/'nucleo1/hiperparametros_xgboost.json')
        assert not c['teste_usado_na_selecao']
        candidate_path = base/'nucleo1/busca_hiperparametros.csv'
        curve_path = base/'nucleo1/curvas_limiar_validacao.csv'
        for path in (candidate_path,curve_path): sources[str(path.relative_to(ROOT))] = sha(path)
        candidates = pd.read_csv(candidate_path)
        winner = candidates.sort_values(['f2','precision','candidato'],ascending=[False,False,True]).iloc[0]
        assert np.isclose(winner.limiar,c['limiar_probabilidade'],rtol=0,atol=1e-10)
        for key in ('n_estimators','max_depth','learning_rate'):
            assert np.isclose(winner[key],c['parametros'][key])
        frame = pd.read_csv(curve_path)
        curve = frame.loc[frame.candidato.eq(winner.candidato)].sort_values('limiar')
        top = curve.sort_values(['f2','limiar'],ascending=[False,False]).iloc[0]
        assert np.isclose(top.limiar,winner.limiar,rtol=0,atol=1e-10)
        curves[station] = curve
        classifiers[station] = (c,winner)
        curve.assign(station=station).to_csv(DEST/f'dados_curva_limiar_{station}.csv',index=False)
        l = read_json(base/'nucleo2/hiperparametros_lstm.json')
        candidates_l = l['selecao_taxa_aprendizado']['candidatas']
        assert not l['selecao_taxa_aprendizado']['dados_teste_utilizados']
        assert min(candidates_l,key=lambda x:x['melhor_val_loss'])['learning_rate'] == l['hiperparametros']['learning_rate']
        lstm[station] = l
        for model in range(1,13):
            family = 'lstm_integral' if model <= 8 else 'convlstm'
            directory = OUT/'estacoes'/station/family
            frozen_path = directory/'selecao_congelada.json'
            frozen = read_json(frozen_path)['selected'] if frozen_path.exists() else {}
            for variant in ('original','filtrado'):
                item = frozen.get(f'M{model}_{variant}')
                row = dict(station=station,model=f'M{model}',variant=variant,
                           selected=item is not None,final_completed=False)
                if item:
                    epochs = item['epochs']; row.update(item['spec'])
                    row.update(epochs_min=min(epochs.values()),epochs_max=max(epochs.values()),
                               epochs_label=epoch_range(epochs),networks=len(epochs))
                    dest = directory/'finais'/f'M{model}'/variant
                    if model >= 9: dest = dest/'selecionado'
                    result_path = dest/'resumo.json'
                    if result_path.exists():
                        result = read_json(result_path)
                        assert result['phase']=='final' and result['all_development_refit']
                        assert result['selected_epochs']==epochs
                        taskname = f'M{model}_{variant}.json' if model<=8 else f'M{model}_{variant}_selecionado_final.json'
                        taskpath = directory/'tarefas'/taskname
                        assert sha(taskpath)==result['task_sha256']
                        task = read_json(taskpath)
                        assert task['spec']==item['spec'] and task['fixed_epochs']==epochs
                        row['final_completed']=True
                rows.append(row)
        p = c['parametros']
        summary_rows.append([station,f"{c['limiar_probabilidade']:.4f}".replace('.',','),
            str(p['n_estimators']),str(p['max_depth']),number(p['learning_rate']),
            number(p.get('min_child_weight')),number(p.get('reg_lambda')),
            'SMOTE' if c['amostragem_treino']['balanceamento']=='smote' else 'Pesos de classe',
            number(p['scale_pos_weight'])])
    data = pd.DataFrame(rows)
    data.to_csv(DEST/'hiperparametros_redes_snapshot.csv',index=False)

    fig = plt.figure(figsize=(17,15))
    fig.text(.055,.973,'HIPERPARÂMETROS E CRITÉRIOS DE ESCOLHA',fontsize=22,weight='bold')
    fig.text(.055,.948,'Quatro estações GNSS  •  Desenvolvimento: 01/12/2023–16/02/2024  •  Teste preservado: 18/02/2024',fontsize=11)
    panel_title(fig,.912,'A  |  Limiar de probabilidade: máximo F₂ na validação temporal',
                'Curvas da candidata XGBoost vencedora • folds 1–4 para escolha; fold 5 reservado • empate em F₂: maior limiar')
    for i,station in enumerate(STATIONS):
        ax=fig.add_axes([.058+i*.237,.708,.203,.157])
        curve=curves[station]; c,w=classifiers[station]
        # Redução visual apenas: seleção e marcação usam a curva completa.
        idx=np.unique(np.r_[np.linspace(0,len(curve)-1,min(1200,len(curve)),dtype=int),np.argmin(abs(curve.limiar-w.limiar))])
        plotted=curve.iloc[idx]
        for column,label,color in [('precision','Precisão',BLUE),('recall','Recall',ORANGE),('f2','F₂',GREEN)]:
            ax.plot(plotted.limiar,plotted[column],color=color,lw=1.6 if column=='f2' else 1,alpha=1 if column=='f2' else .7,label=label)
        ax.axvline(w.limiar,color=INK,lw=1,ls='--')
        ax.scatter([w.limiar],[w.f2],s=45,color=GREEN,edgecolor='white',zorder=5)
        ax.set(title=f'{station}  |  τ = {w.limiar:.4f}'.replace('.',','),xlim=(0,1),ylim=(0,1.04),xlabel='Limiar de probabilidade τ')
        if i==0: ax.set_ylabel('Precisão / recall / F₂')
        ax.set_xticks([0,.25,.5,.75,1]);ax.legend(loc='lower left',fontsize=8,framealpha=.9)
    fig.text(.058,.669,'F₂ = 5PR / (4P + R)  •  Alerta se P(S4 futuro ≥ 0,5) ≥ τ.  O limiar de S4 (0,5) é distinto do limiar probabilístico τ.',fontsize=10)

    panel_title(fig,.63,'B  |  XGBoost: hiperparâmetros efetivamente selecionados',
                'Quatro candidatas; vence o maior F₂ (desempate por precisão). Subsample = 0,8; colsample_bytree = 0,8; semente = 42.')
    ax=fig.add_axes([.055,.48,.89,.116]);ax.axis('off')
    columns=['Estação','τ escolhido','Árvores','Profundidade','Taxa η','Min. peso\nfilho','λ (L2)','Balanceamento','Peso positivo']
    table=ax.table(cellText=summary_rows,colLabels=columns,cellLoc='center',bbox=[0,0,1,1],
                   colWidths=[.09,.105,.085,.115,.09,.10,.08,.16,.115])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#DCE4E9');cell.set_facecolor('#EAF1F6' if r==0 else ('#F7F9FB' if r%2 else 'white'))
        if r==0:cell.set_text_props(weight='bold')
    fig.text(.055,.463,'— = parâmetro não explicitado no registro; utiliza o padrão da biblioteca. A escolha utiliza validação, não métricas do teste.',fontsize=9,color='#526577')

    panel_title(fig,.423,'C  |  LSTM M1–M8: escolha da taxa de aprendizado por estação',
                'Menor MSE normalizado de validação na rede convencional multivariada ORIGINAL; taxa compartilhada por M1–M8 e pelos dois ramos.')
    for i,station in enumerate(STATIONS):
        ax=fig.add_axes([.058+i*.237,.239,.203,.142])
        l=lstm[station];candidates=l['selecao_taxa_aprendizado']['candidatas']
        x=np.array([r['learning_rate'] for r in candidates]);y=np.array([r['melhor_val_loss'] for r in candidates])
        order=np.argsort(x);best=np.argmin(y)
        ax.plot(x[order],y[order],'-o',color=BLUE,ms=4,lw=1.5)
        ax.scatter(x[best],y[best],s=90,color=ORANGE,edgecolor='white',zorder=5)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks([5e-7,5e-5,5e-3,5e-2],['5×10⁻⁷','5×10⁻⁵','5×10⁻³','5×10⁻²'])
        ax.set_title(f'{station}  |  η = {number(l["hiperparametros"]["learning_rate"])}',fontsize=11)
        ax.set_xlabel('Taxa de aprendizado η')
        if i==0:ax.set_ylabel('MSE de validação (log)')
        ax.tick_params(axis='x',labelsize=8)

    card(fig.add_axes([.055,.064,.284,.13]),'LSTM / Attention · M1–M8',
         '3 camadas × 31 unidades; dropout 0,2\nAdam; MSE; batch até 400\nSaída: LeakyReLU (α = 0,3)\nBusca: até 50 épocas; patience 3\nConvencional: 48 épocas anteriores\nPiecewise: mesmo horário em 15 dias', '#EEF5F9')
    card(fig.add_axes([.358,.064,.284,.13]),'ConvLSTM · M9–M12',
         '1 ConvLSTM1D; kernel 3; Dense 32\nC0: 16 filtros / η = 0,001\nC1: 32 filtros / η = 0,0003\nAdam; MSE; batch 64; dropout 0,2\nAté 60 épocas; patience 8; clipnorm 1', '#EDF6F1')
    card(fig.add_axes([.661,.064,.284,.13]),'Escolha → congelamento → ajuste final',
         'ConvLSTM: menor RMSE médio dos ramos\nEarly stopping: 03–09/02; seleção: 10–16/02\nPOAL M11/M12, slots 14–41: janela anterior*\nÉpocas escolhidas por rede/horário\nReajuste até 16/02; teste só em 18/02', '#FFF5E9')
    fig.text(.055,.041,'* POAL autorizado: ajuste da busca até 26/01; early stopping 27/01–02/02; seleção 03–09/02. Reajuste final e teste permanecem iguais.',fontsize=9)
    fig.text(.055,.022,f'Registros locais verificados • Snapshot UTC: {snapshot[:19].replace("T"," ")} • Estado de cada rede na figura de detalhamento.',fontsize=8.5,color='#526577')
    save(fig,'01_escolha_limiar_e_hiperparametros')

    fig=plt.figure(figsize=(17,12))
    fig.text(.055,.955,'REDES NEURAIS | PARÂMETROS SELECIONADOS E AJUSTE FINAL',fontsize=20,weight='bold')
    fig.text(.055,.922,'η = taxa de aprendizado; O/F = épocas dos ramos original/filtrado. Intervalos representam os 48 horários das redes piecewise.',fontsize=11)
    contents=[]; colors=[]
    for model in range(1,13):
        row=[f'M{model}\n{LABELS[model-1]}']; rowcolors=['#EDF2F7']
        for station in STATIONS:
            group=data.loc[data.station.eq(station)&data.model.eq(f'M{model}')]
            if not group.selected.all():
                row.append('Seleção não congelada\nParâmetros finais pendentes');rowcolors.append('#F1F3F5');continue
            o=group.loc[group.variant.eq('original')].iloc[0];f=group.loc[group.variant.eq('filtrado')].iloc[0]
            assert o.learning_rate==f.learning_rate
            header=f'η = {number(o.learning_rate)}'
            if model>=9:header+=f'  |  {int(o.filters)} filtros'
            count=int(group.final_completed.sum())
            row.append(header+f'\nO: {o.epochs_label}  |  F: {f.epochs_label}\nAjustes finais: {count}/2')
            rowcolors.append('#E8F3EB' if count==2 else '#FFF1DC')
        contents.append(row);colors.append(rowcolors)
    ax=fig.add_axes([.055,.118,.89,.77]);ax.axis('off')
    table=ax.table(cellText=contents,colLabels=['Modelo / organização',*STATIONS],cellColours=colors,
                   cellLoc='center',bbox=[0,0,1,1],colWidths=[.205,.19875,.19875,.19875,.19875])
    table.auto_set_font_size(False);table.set_fontsize(10.3)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('white');cell.set_linewidth(2)
        if r==0:cell.set_facecolor(INK);cell.set_text_props(color='white',weight='bold',fontsize=12)
        elif c==0:cell.set_text_props(weight='bold',fontsize=10)
    fig.text(.055,.087,'VERDE: dois ramos finais concluídos     |     AMARELO: seleção congelada, ajuste final incompleto     |     CINZA: seleção ainda não congelada',fontsize=10)
    complete=int(data.final_completed.sum())
    fig.text(.055,.06,f'Snapshot: {complete}/96 configurações finais verificadas. Seleção congelada não significa treinamento final concluído.',fontsize=11,weight='bold')
    fig.text(.055,.035,'M1–M8: 3 × 31 unidades, Adam, MSE, dropout 0,2, batch ≤400. M9–M12: ConvLSTM1D, Adam, MSE, batch 64, dropout 0,2.',fontsize=10)
    fig.text(.055,.016,f'Atualização UTC: {snapshot[:19].replace("T"," ")} • Fonte: seleções congeladas, tarefas e resumos finais; nenhum hiperparâmetro inferido de resultados do teste.',fontsize=8.5,color='#526577')
    save(fig,'02_hiperparametros_redes_por_estacao')
    sources[str(Path(__file__).resolve().relative_to(ROOT))]=sha(__file__)
    (DEST/'fontes_e_verificacao.json').write_text(json.dumps(dict(snapshot_utc=snapshot,
        final_configurations=complete,expected_configurations=96,test_used_for_selection=False,
        sources_sha256=sources,thresholds={s:classifiers[s][0]['limiar_probabilidade'] for s in STATIONS},
        lstm_learning_rates={s:lstm[s]['hiperparametros']['learning_rate'] for s in STATIONS}),ensure_ascii=False,indent=2),encoding='utf-8')
    guide = '''# Figuras dos hiperparâmetros e da escolha do limiar

## Figura principal

[PNG](01_escolha_limiar_e_hiperparametros.png) · [PDF](01_escolha_limiar_e_hiperparametros.pdf) · [SVG editável](01_escolha_limiar_e_hiperparametros.svg)

**Legenda sugerida:** Seleção de hiperparâmetros e do limiar de probabilidade no
desenvolvimento, mantendo o teste de 18/02/2024 separado. (A) Precisão, recall e
F₂ em função do limiar para a candidata XGBoost vencedora de cada estação; o
marcador identifica o máximo de F₂. (B) Parâmetros selecionados do classificador.
(C) Comparação das seis taxas de aprendizado por meio do menor MSE normalizado
de validação da LSTM convencional multivariada no ramo original. Os quadros
inferiores descrevem as configurações e os critérios das redes. Os escores dos
painéis A e C são de validação, não de teste, e o MSE normalizado não deve ser
interpretado como erro em TECU. A exceção temporal autorizada de POAL é explícita.

## Detalhamento das redes

[PNG](02_hiperparametros_redes_por_estacao.png) · [PDF](02_hiperparametros_redes_por_estacao.pdf) · [SVG editável](02_hiperparametros_redes_por_estacao.svg)

**Legenda sugerida:** Taxas de aprendizado e épocas congeladas por modelo,
estação e ramo. Para redes piecewise, apresenta-se o mínimo e máximo das épocas
dos 48 horários. A cor distingue dois ajustes finais concluídos, seleção pronta
com ajuste ainda incompleto e seleção ainda não congelada. As épocas são as
escolhidas para o reajuste final, não o teto da busca com early stopping.
Uma célula amarela não afirma que seus dois ramos já foram treinados integralmente.

## Reprodução e fontes

Execute `.venv/Scripts/python.exe -X utf8 scripts/fevereiro/figura_hiperparametros.py`
a partir da raiz do projeto. A consolidação geral também atualiza estas figuras.
Os parâmetros não são alterados e nenhum treinamento é disparado por esse script.
As curvas completas estão nos CSV desta pasta; a figura reduz apenas a quantidade
de pontos desenhados, preservando o ponto vencedor exato. Os registros de origem
e seus SHA-256 estão em [fontes_e_verificacao.json](fontes_e_verificacao.json).
O [CSV das redes](hiperparametros_redes_snapshot.csv) contém as 96 configurações
esperadas, incluindo estados pendentes. Todos os números representam o instante
da geração e são atualizados quando novas redes são consolidadas.
'''
    (DEST/'README.md').write_text(guide,encoding='utf-8')
    print(json.dumps(dict(output=str(DEST),final_configurations=complete,figures=2,formats=['png','pdf','svg']),ensure_ascii=False))


if __name__=='__main__':main()
