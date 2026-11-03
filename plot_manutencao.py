"""Resolve cp-maintenance.py e gera figuras/6_manutencao.png (cronograma + MW fora)."""
import runpy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

SAIDA = Path('figuras')
TINTA, TINTA_2, GRADE = '#0b0b0b', '#52514e', '#e1e0d9'
COR = {'Oil': '#2a78d6', 'Hydro': '#eb6834', 'Coal': '#1baf7a', 'Nuclear': '#eda100'}
ROTULO = {'Oil': 'Óleo', 'Hydro': 'Hidro', 'Coal': 'Carvão', 'Nuclear': 'Nuclear'}

ns = runpy.run_path('cp-maintenance.py')
msol, unit_vars = ns['msol'], ns['unit_vars']
units, load = ns['df_units'], ns['df_load']
cap_tot, margin = ns['cap_tot'], ns['margin']

peak = msol.get_objective_value()
print(f"status={msol.get_solve_status()} peak_offline={peak} "
      f"bound={msol.get_objective_bound()} tempo={msol.get_solve_time():.1f}s "
      f"branches={msol.get_infos().get('NumberOfBranches')}")

semanas = np.arange(52)
fora, equipes = np.zeros(52), np.zeros(52)
linhas = []
for _, u in units.iterrows():
    v = msol.get_var_solution(unit_vars[u['unit_name']])
    ini, fim = v.get_start(), v.get_end()
    fora[ini:fim] += u['cap_mw']
    equipes[ini:fim] += u['crew_req']
    linhas.append((u['unit_name'], u['fuel'], ini, fim))
limite = np.floor(cap_tot - (1 + margin) * load['peak_mw'].to_numpy()[:52])
semana_critica = int(np.argmin(limite - fora))
print(f"fora max={fora.max():.0f} MW  equipes max={equipes.max():.0f}  "
      f"folga minima da reserva={np.min(limite - fora):.0f} MW (t={semana_critica})")
print(f"semanas com equipes=max: {int((equipes == equipes.max()).sum())}")

plt.rcParams.update({'font.size': 9, 'text.color': TINTA, 'axes.labelcolor': TINTA_2,
                     'xtick.color': TINTA_2, 'ytick.color': TINTA_2})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True,
                             gridspec_kw={'height_ratios': [2.1, 1]})

for k, (nome, fuel, ini, fim) in enumerate(linhas):
    a1.barh(k, fim - ini, left=ini, height=0.72, color=COR[fuel], edgecolor='white', linewidth=0.8)
a1.set_yticks(range(len(linhas)), [l[0] for l in linhas], fontsize=5.5)
a1.invert_yaxis()
a1.set_ylabel('unidade')
a1.set_title('Cronograma de manutenção', loc='left', fontsize=10, color=TINTA)
a1.legend(handles=[Patch(color=COR[f], label=ROTULO[f]) for f in COR], ncol=2, fontsize=8,
          frameon=False, loc='upper right')

a2.fill_between(np.append(semanas, 52), np.append(fora, fora[-1]), step='post',
                color=COR['Oil'], alpha=0.30, linewidth=0)
a2.step(np.append(semanas, 52), np.append(fora, fora[-1]), where='post', color=COR['Oil'], lw=2)
a2.step(np.append(semanas, 52), np.append(limite, limite[-1]), where='post', color=TINTA_2, lw=1.2, ls='--')
a2.axhline(peak, color=TINTA, lw=1, ls=':')
a2.annotate(f'pico minimizado: {peak:.0f} MW', (14, peak), xytext=(0, -11), textcoords='offset points',
            ha='center', fontsize=8)
a2.text(43.6, 800, 'limite da reserva', fontsize=8, color=TINTA_2)
a2.set_ylim(0, 1.15 * limite.max())
a2.set_xlim(0, 52)
a2.set_xlabel('semana (índice t do modelo)')
a2.set_ylabel('MW fora de operação')
a2.set_title('Capacidade em manutenção', loc='left', fontsize=10, color=TINTA)

for a in (a1, a2):
    a.grid(axis='x', color=GRADE, lw=0.6)
    a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    a.spines['left'].set_color(GRADE)
    a.spines['bottom'].set_color(GRADE)

fig.tight_layout()
SAIDA.mkdir(exist_ok=True)
fig.savefig(SAIDA / '6_manutencao.png', dpi=180, bbox_inches='tight')
print('figuras/6_manutencao.png gerado')
