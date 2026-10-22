"""Curva de convergencia do subgradiente a partir de lagrange.npz."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARQ = Path('lagrange.npz')
SAIDA = Path('figuras')

plt.rcParams.update({
    'figure.dpi': 110,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.axisbelow': True,
    'font.size': 9,
})


def main():
    if not ARQ.exists():
        raise SystemExit("lagrange.npz nao encontrado. Rode 'python lagrange.py' antes.")
    d = np.load(ARQ, allow_pickle=False)
    it, L = d['iteracao'], d['L']
    melhor, norma, rho = d['melhor_lb'], d['norma_subgradiente'], d['rho']
    z_lp, z_ld, ub = float(d['z_lp']), float(d['z_ld']), float(d['ub'])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6),
                                  gridspec_kw={'width_ratios': [1.35, 1]})

    ax.axhline(ub, color='crimson', ls='--', lw=1.2, label=r'$UB$ (incumbente MIP)')
    ax.axhline(z_lp, color='0.35', ls=':', lw=1.4, label=r'$z_{LP}$ (relaxação linear)')
    ax.plot(it, L, color='tab:blue', lw=1.4, marker='o', ms=3,
            label=r'$L(\lambda^k)$')
    ax.plot(it, melhor, color='seagreen', lw=2.0,
            label=r'melhor LB $= z_{LD}$')
    ax.fill_between(it, melhor, ub, color='crimson', alpha=0.07)
    ax.annotate(f'gap {100 * (ub - z_ld) / ub:.2f}%',
                xy=(it[-1] * 0.62, 0.5 * (z_ld + ub)), fontsize=8,
                color='crimson', ha='center')
    ax.set_xlabel('iteração $k$')
    ax.set_ylabel(r'limite [R\$]')
    ax.set_title(r'Limites: $z_{LP} \leq z_{LD} \leq z^* \leq UB$', fontsize=10)
    ax.ticklabel_format(axis='y', style='plain')
    ax.yaxis.set_major_formatter(lambda v, _: f'{v/1e6:.3f}M')
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)

    ax2.plot(it, norma, color='tab:purple', lw=1.4, marker='o', ms=3)
    ax2.set_xlabel('iteração $k$')
    ax2.set_ylabel(r'$\|s^k\|$', color='tab:purple')
    ax2.tick_params(axis='y', colors='tab:purple')
    ax2.set_ylim(0, 1.15 * norma.max())
    axr = ax2.twinx()
    axr.step(it, rho, where='post', color='darkorange', lw=1.6)
    axr.set_yscale('log', base=2)
    axr.set_ylabel(r'$\rho_k$', color='darkorange')
    axr.tick_params(axis='y', colors='darkorange')
    axr.grid(False)
    ax2.set_title(r'Subgradiente e passo: $\|s^k\|$ não decai, $\rho_k$ colapsa',
                  fontsize=10)

    fig.tight_layout()
    SAIDA.mkdir(exist_ok=True)
    fig.savefig(SAIDA / '5_lagrange_convergencia.png', bbox_inches='tight')
    plt.close(fig)
    print('figuras/5_lagrange_convergencia.png gerado')


if __name__ == '__main__':
    main()
