"""Compara L(lam^k) do subgradiente e dos relax-and-cut: figuras/7_relax_and_cut.png.

    python plot_relax_and_cut.py [arquivo_delayed.npz]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DELAYED = sys.argv[1] if len(sys.argv) > 1 else 'lagrange_delayed_t504.npz'
BOUND_MIP = 13_712_469.02   # ObjBound do MIP completo em 120 s (log de lagrange_delayed.py)
TINTA, TINTA_2, GRADE = '#0b0b0b', '#52514e', '#e1e0d9'
SERIES = [('Subgradiente', 'lagrange.npz', '#2a78d6'),
          ('Non-delayed', 'lagrange_nondelayed.npz', '#eb6834'),
          ('Delayed', DELAYED, '#1baf7a')]

base = np.load('lagrange.npz')
T = base['lam_rup'].shape[1]
ub = float(base['ub'])

plt.rcParams.update({'font.size': 10, 'text.color': TINTA, 'axes.labelcolor': TINTA_2,
                     'xtick.color': TINTA_2, 'ytick.color': TINTA_2})
fig, ax = plt.subplots(figsize=(6.4, 4.3))
for nome, arq, cor in SERIES:
    d = np.load(arq)
    if d['lam_rup'].shape[1] != T:
        raise SystemExit(f'{arq}: horizonte diferente de lagrange.npz')
    print(f"{nome}: it={len(d['L'])} melhor={d['L'].max():,.2f} (k={int(d['L'].argmax())})")
    ax.plot(np.arange(len(d['L'])), d['L'], color=cor, lw=2, label=nome)

ax.axhline(ub, color=TINTA, lw=1.2, ls='--')
ax.axhline(BOUND_MIP, color=TINTA_2, lw=1.2, ls=':')
ax.annotate('UB', (1, ub), xytext=(0, 3), textcoords='offset points', fontsize=9)
ax.annotate('bound do MIP (120 s)', (1, BOUND_MIP), xytext=(0, -12), textcoords='offset points',
            fontsize=9, color=TINTA_2)
ax.set_ylim(13_700_000, 13_715_500)
ax.yaxis.set_major_formatter(lambda v, _: f'{v / 1e6:.3f}'.replace('.', ','))
ax.set_xlabel('iteração $k$ (acumulada entre rodadas)')
ax.set_ylabel(r'$L(\lambda^k)$ [milhões de R\$]')
ax.legend(frameon=False, fontsize=9, loc='center right', ncol=1)
ax.grid(axis='y', color=GRADE, lw=0.6)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(GRADE)

fig.tight_layout()
Path('figuras').mkdir(exist_ok=True)
fig.savefig('figuras/7_relax_and_cut.png', dpi=180, bbox_inches='tight')
print('figuras/7_relax_and_cut.png gerado')
