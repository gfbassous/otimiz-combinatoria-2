"""
Visualizações dos resultados do despacho hidrotérmico (IEEE 14 barras).

Lê 'resultados.npz' (gerado pelo main.py) e escreve em figuras/:
  1_geracao_demanda.png   - geração acumulada por usina + curva de demanda
  2_rede_fluxos.gif       - animação dos fluxos na rede, com geração/demanda por barra
  3_armazenamento.png     - energia armazenada nos reservatórios
  4_cmo.png               - custo marginal de operação

Uso:  python visualizacao.py [--passo N] [--fps N] [--sem-animacao]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

ARQ = Path('resultados.npz')
SAIDA = Path('figuras')
DIAS = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']

# Coordenadas do diagrama unifilar do IEEE 14 barras (apenas para desenho).
POS = {
    'B1': (1.0, 5.0), 'B2': (2.5, 4.2), 'B3': (4.2, 3.4), 'B4': (3.2, 2.2),
    'B5': (2.0, 3.0), 'B6': (1.0, 1.5), 'B7': (4.1, 1.2), 'B8': (5.4, 1.2),
    'B9': (3.4, 0.2), 'B10': (2.6, -0.7), 'B11': (1.6, 0.3), 'B12': (0.0, 0.6),
    'B13': (0.9, -0.6), 'B14': (2.4, -1.6),
}

plt.rcParams.update({
    'figure.dpi': 110,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.axisbelow': True,
    'font.size': 9,
})


def carregar():
    if not ARQ.exists():
        raise SystemExit("resultados.npz não encontrado. Rode 'python main.py' antes.")
    d = np.load(ARQ, allow_pickle=False)
    return {k: d[k] for k in d.files}


def rotulos_dia(ax, T, xlabel=True):
    """Eixo x em horas, com o nome do dia da semana marcado a cada 00h."""
    ticks = np.arange(0, T + 1, 12)
    rotulos = [f'{DIAS[(h // 24) % 7]}\n00h' if h % 24 == 0 and h < T else f'{h % 24:02d}h'
               for h in ticks]
    ax.set_xlim(0, T)
    ax.set_xticks(ticks)
    ax.set_xticklabels(rotulos, fontsize=7)
    for d in range(1, (T + 23) // 24):
        ax.axvline(24 * d, color='0.6', lw=0.8, ls=':')
    if xlabel:
        ax.set_xlabel('Hora da semana')


# --------------------------------------------------------------------------- #
# 1. Geração acumulada + demanda
# --------------------------------------------------------------------------- #
def fig_geracao(R):
    g_uhe, g_ute, defic = R['g_uhe'], R['g_ute'], R['deficit']
    uhe, ute = list(R['uhe']), list(R['ute'])
    T = g_uhe.shape[1]
    horas = np.arange(T)
    demanda_total = R['demanda'].sum(axis=0)

    series = [(u, g_uhe[i]) for i, u in enumerate(uhe)]
    series += [(u, g_ute[i]) for i, u in enumerate(ute)]
    series.append(('Déficit', defic.sum(axis=0)))

    cores = list(plt.get_cmap('Blues')(np.linspace(0.45, 0.9, len(uhe))))
    cores += list(plt.get_cmap('YlOrRd')(np.linspace(0.35, 0.85, len(ute))))
    cores.append((0.1, 0.1, 0.1, 1.0))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 1],
                                  sharex=True)
    base = np.zeros(T)
    for (nome, serie), cor in zip(series, cores):
        if nome == 'Déficit' and serie.max() < 1e-6:
            continue
        ax.bar(horas + 0.5, serie, bottom=base, width=1.0, color=cor, label=nome,
               edgecolor='none')
        base = base + serie
    ax.plot(horas + 0.5, demanda_total, color='k', lw=1.8, label='Demanda',
            drawstyle='steps-mid')
    ax.set_ylabel('Potência [MW]')
    ax.set_title('Geração acumulada por usina e demanda do sistema', fontweight='bold')
    ax.legend(ncols=5, fontsize=8, loc='upper left', framealpha=0.9)
    ax.set_ylim(0, max(base.max(), demanda_total.max()) * 1.28)

    energia_h = g_uhe.sum(axis=0)
    energia_t = g_ute.sum(axis=0)
    ax2.fill_between(horas + 0.5, 0, 100 * energia_h / np.maximum(demanda_total, 1e-9),
                     color=cores[1], alpha=0.8, step='mid', label='Hidráulica')
    ax2.fill_between(horas + 0.5, 100 * energia_h / np.maximum(demanda_total, 1e-9), 100,
                     color=cores[len(uhe) + 1], alpha=0.8, step='mid', label='Térmica')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Participação [%]')
    ax2.legend(ncols=2, fontsize=8, loc='lower left')
    rotulos_dia(ax2, T, xlabel=True)

    tot_h, tot_t = energia_h.sum() / 1000, energia_t.sum() / 1000
    fig.text(0.99, 0.965,
             f'Energia semanal: hidráulica {tot_h:.1f} GWh | térmica {tot_t:.1f} GWh | '
             f'custo total R$ {float(R["custo_total"]):,.0f}'.replace(',', '.'),
             ha='right', va='top', fontsize=8, color='0.3')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(SAIDA / '1_geracao_demanda.png', bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Animação dos fluxos na rede
# --------------------------------------------------------------------------- #
def fig_rede(R, passo=1, fps=8):
    barras = list(R['barras'])
    de, para = list(R['linha_de']), list(R['linha_para'])
    cap = R['capacidade_linhas']
    fluxo, demanda, cmo = R['fluxo'], R['demanda'], R['cmo']
    T = fluxo.shape[1]

    idx = {b: k for k, b in enumerate(barras)}
    ger = np.zeros_like(demanda)
    for u, b in enumerate(R['uhe_barra']):
        ger[idx[str(b)]] += R['g_uhe'][u]
    for g, b in enumerate(R['ute_barra']):
        ger[idx[str(b)]] += R['g_ute'][g]
    injecao = ger + R['deficit'] - demanda

    carreg = np.abs(fluxo) / cap[:, None]
    lim_inj = np.abs(injecao).max()
    norm_carga = Normalize(0, 1)
    norm_inj = TwoSlopeNorm(vmin=-lim_inj, vcenter=0.0, vmax=lim_inj)
    cmap_carga, cmap_inj = plt.get_cmap('viridis'), plt.get_cmap('coolwarm')

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_axis_off()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.13)
    cax1 = fig.add_axes((0.10, 0.06, 0.34, 0.022))
    cax2 = fig.add_axes((0.56, 0.06, 0.34, 0.022))
    fig.colorbar(ScalarMappable(norm_carga, cmap_carga), cax=cax1,
                 orientation='horizontal').set_label(
        'Carregamento da linha  |f| / capacidade', fontsize=8)
    fig.colorbar(ScalarMappable(norm_inj, cmap_inj), cax=cax2,
                 orientation='horizontal').set_label(
        'Injeção líquida na barra  (geração - demanda) [MW]', fontsize=8)
    cax1.tick_params(labelsize=7)
    cax2.tick_params(labelsize=7)

    xs = [POS[b][0] for b in barras]
    ys = [POS[b][1] for b in barras]

    def desenhar(t):
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
        ax.set_ylim(min(ys) - 0.9, max(ys) + 0.5)

        for l, (i, j) in enumerate(zip(de, para)):
            p0, p1 = np.array(POS[str(i)]), np.array(POS[str(j)])
            if fluxo[l, t] < 0:                      # seta aponta o sentido real
                p0, p1 = p1, p0
            v = p1 - p0
            d = v / np.linalg.norm(v)
            a, b = p0 + 0.16 * d, p1 - 0.16 * d
            c = cmap_carga(norm_carga(carreg[l, t]))
            ax.add_patch(FancyArrowPatch(a, b, color=c, lw=1.0 + 4.5 * carreg[l, t],
                                         arrowstyle='-|>', mutation_scale=13,
                                         shrinkA=0, shrinkB=0, zorder=1,
                                         alpha=0.95))
            mid = (a + b) / 2
            ax.text(*mid, f'{abs(fluxo[l, t]):.0f}', fontsize=6.2, ha='center',
                    va='center', color='0.15', zorder=3,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none',
                              alpha=0.75))

        ax.scatter(xs, ys, s=620, c=injecao[:, t], cmap=cmap_inj, norm=norm_inj,
                   edgecolors='k', linewidths=0.8, zorder=4)
        for k, b in enumerate(barras):
            x, y = POS[str(b)]
            ax.text(x, y, str(b), ha='center', va='center', fontsize=7.5,
                    fontweight='bold', zorder=5)
            ax.text(x, y - 0.33, f'G {ger[k, t]:.0f} | D {demanda[k, t]:.0f}',
                    ha='center', va='top', fontsize=6.4, color='0.25', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='0.8',
                              alpha=0.85))

        d_tot, g_tot = demanda[:, t].sum(), ger[:, t].sum()
        cmo_med = float(np.average(cmo[:, t], weights=np.maximum(demanda[:, t], 1e-9)))
        ax.set_title(
            f'Fluxos de potência na rede — {DIAS[(t // 24) % 7]} {t % 24:02d}h '
            f'(hora {t + 1} de {T})\n'
            f'Demanda {d_tot:.0f} MW | Geração {g_tot:.0f} MW | '
            f'CMO médio R$ {cmo_med:.0f}/MWh | '
            f'linha mais carregada {100 * carreg[:, t].max():.0f}%',
            fontsize=11, fontweight='bold')
        return ()

    quadros = range(0, T, passo)
    anim = FuncAnimation(fig, desenhar, frames=quadros, interval=1000 / fps, blit=False)
    anim.save(SAIDA / '2_rede_fluxos.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Energia armazenada nos reservatórios
# --------------------------------------------------------------------------- #
def fig_armazenamento(R):
    arm, cap = R['uhe_arm'], R['uhe_arm_cap']
    vert, turb = R['vert_uhe'], R['turb_uhe']
    uhe = list(R['uhe'])
    T = arm.shape[1]
    horas = np.arange(T)

    fig, axs = plt.subplots(3, 2, figsize=(13, 9))
    eixos = axs.flat
    cores = plt.get_cmap('Blues')(np.linspace(0.5, 0.9, len(uhe)))

    for u, nome in enumerate(uhe):
        ax = eixos[u]
        ax.fill_between(horas + 0.5, 0, arm[u], color=cores[u], alpha=0.55)
        ax.plot(horas + 0.5, arm[u], color=cores[u], lw=1.8, label='Armazenamento')
        ax.axhline(cap[u], color='0.35', ls='--', lw=1.0, label='Capacidade')
        ax.axhline(cap[u] / 2, color='crimson', ls=':', lw=1.2,
                   label='Inicial / meta final (50%)')
        ax.set_ylim(0, cap[u] * 1.18)
        ax.set_ylabel('Volume [MWh-eq]')
        ax.set_title(f'{nome} (barra {str(R["uhe_barra"][u])}) — '
                     f'final {100 * arm[u, -1] / cap[u]:.0f}% da capacidade',
                     fontsize=10, fontweight='bold')
        if vert[u].max() > 1e-6:
            axv = ax.twinx()
            axv.bar(horas + 0.5, vert[u], width=1.0, color='darkorange', alpha=0.6,
                    label='Vertimento')
            axv.set_ylim(0, vert[u].max() * 4)
            axv.set_ylabel('Vertimento [MW]', color='darkorange', fontsize=8)
            axv.tick_params(axis='y', colors='darkorange', labelsize=7)
            axv.grid(False)
        if u == 0:
            ax.legend(fontsize=7, loc='lower left', ncols=3)
        rotulos_dia(ax, T, xlabel=False)

    ax = eixos[4]
    for u, nome in enumerate(uhe):
        ax.plot(horas + 0.5, 100 * arm[u] / cap[u], lw=1.8, color=cores[u], label=nome)
    ax.axhline(50, color='crimson', ls=':', lw=1.2)
    ax.set_ylabel('Volume útil [%]')
    ax.set_title('Comparação entre reservatórios (% da capacidade)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, ncols=4, loc='lower left')
    ax.set_ylim(0, 118)
    rotulos_dia(ax, T)

    ax = eixos[5]
    base = np.zeros(T)
    for u, nome in enumerate(uhe):
        ax.bar(horas + 0.5, turb[u], bottom=base, width=1.0, color=cores[u], label=nome)
        base = base + turb[u]
    ax.set_ylabel('Turbinamento [MW]')
    ax.set_ylim(0, base.max() * 1.25)
    ax.set_title('Turbinamento total da cascata', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, ncols=4, loc='upper left')
    rotulos_dia(ax, T)

    fig.suptitle('Energia armazenada nos reservatórios da cascata',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(SAIDA / '3_armazenamento.png', bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Custo marginal de operação
# --------------------------------------------------------------------------- #
def fig_cmo(R):
    cmo, demanda = R['cmo'], R['demanda']
    barras = [str(b) for b in R['barras']]
    T = cmo.shape[1]
    horas = np.arange(T)
    peso = np.maximum(demanda, 1e-9)
    cmo_sistema = np.average(cmo, axis=0, weights=peso)

    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10),
                                       height_ratios=[2, 2, 1.4])

    ax.fill_between(horas + 0.5, cmo.min(axis=0), cmo.max(axis=0), color='steelblue',
                    alpha=0.25, label='Faixa entre as barras (mínimo–máximo)')
    ax.plot(horas + 0.5, cmo_sistema, color='navy', lw=1.8,
            label='CMO do sistema (média ponderada pela carga)')
    for k, (cvu, nome) in enumerate(zip(R['ute_cvu'], R['ute'])):
        ax.axhline(cvu, color='0.6', ls='--', lw=0.7,
                   label='CVU das UTEs' if k == 0 else None)
        ax.text(T - 1, cvu, f'{str(nome)}', va='bottom', ha='right',
                fontsize=7, color='0.45')
    ax.set_ylabel('CMO [R$/MWh]')
    ax.set_ylim(0, max(cmo.max(), R['ute_cvu'].max()) * 1.08)
    ax.set_title('Custo marginal de operação ao longo da semana', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    rotulos_dia(ax, T, xlabel=False)

    malha = ax2.pcolormesh(np.arange(T + 1), np.arange(len(barras) + 1), cmo,
                           cmap='inferno', shading='flat',
                           norm=Normalize(vmin=max(cmo.min(), 0.0),
                                          vmax=np.percentile(cmo, 99)))
    ax2.set_yticks(np.arange(len(barras)) + 0.5)
    ax2.set_yticklabels(barras, fontsize=7)
    ax2.set_ylabel('Barra')
    ax2.set_title('CMO por barra e por hora', fontweight='bold')
    ax2.grid(False)
    fig.colorbar(malha, ax=ax2, pad=0.01, fraction=0.03, extend='max',
                 label='R$/MWh')
    rotulos_dia(ax2, T, xlabel=False)

    medio = np.average(cmo, axis=1, weights=peso)
    ordem = np.argsort(medio)[::-1]
    ax3.bar(np.arange(len(barras)), medio[ordem], color='indianred')
    ax3.set_xticks(np.arange(len(barras)))
    ax3.set_xticklabels([barras[k] for k in ordem], fontsize=7)
    ax3.set_ylabel('CMO médio [R$/MWh]')
    ax3.set_xlabel('Barra')
    ax3.set_title('CMO médio semanal por barra (a diferença revela congestionamento)',
                  fontweight='bold')

    fig.tight_layout()
    fig.savefig(SAIDA / '4_cmo.png', bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Gráficos do despacho hidrotérmico')
    p.add_argument('--passo', type=int, default=1,
                   help='usar 1 quadro a cada N horas na animação')
    p.add_argument('--fps', type=int, default=8, help='quadros por segundo do GIF')
    p.add_argument('--animacao', action='store_true',
                   help='gera apenas as figuras estáticas')
    args = p.parse_args()

    R = carregar()
    SAIDA.mkdir(exist_ok=True)

    fig_geracao(R)
    print('1_geracao_demanda.png')
    fig_armazenamento(R)
    print('3_armazenamento.png')
    fig_cmo(R)
    print('4_cmo.png')
    if args.animacao:
        fig_rede(R, passo=args.passo, fps=args.fps)
        print('2_rede_fluxos.gif')
    print(f'Figuras em {SAIDA.resolve()}')


if __name__ == '__main__':
    main()
