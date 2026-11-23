"""
Visualiza a matriz de restricoes do despacho hidrotermico (modelo completo, igual a
branch&bound.py) como imagem, colorindo cada nao-zero pela familia da restricao.

Duas ordenacoes mostram estruturas diferentes:
  familia : linhas agrupadas por familia e colunas por tipo de variavel -> blocos
            (o que cada familia toca) e as bandas diagonais do acoplamento temporal
  tempo   : linhas e colunas ordenadas por etapa -> "escada": um bloco por hora,
            ligado a hora anterior por armazenamento, rampas e logica de estado, e
            a varias horas atras pelas janelas de tempo minimo

    python plot_matriz.py                  # 24 h, as duas ordenacoes
    python plot_matriz.py -T 72            # 72 h
    python plot_matriz.py -T 0             # horizonte completo de dados.xlsx
    python plot_matriz.py --ordem tempo    # so uma ordenacao

O modelo e montado por construir_modelo(True) de lagrange.py (requer gurobipy).
Imagens grandes sao agregadas em pixels: cada pixel recebe a cor da familia com
mais nao-zeros nele.
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# grupos de restricoes, na ordem das cores (paleta categorica, ordem fixa)
GRUPOS = [
    ('Rede (KCL, KVL, ref.)',      ('KCL', 'KVL', 'barra_referencia'), '#2a78d6'),
    ('Balanço hídrico',            ('balanco_hidrico', 'producao_hidro', 'balanco_armazenamento',
                                    'balanco_armazenamento_inicial', 'armazenamento_final'), '#eb6834'),
    ('Limites com binária',        ('capacidade_hidro', 'geracao_minima_hidro',
                                    'capacidade_ute', 'geracao_minima_ute'), '#1baf7a'),
    ('Lógica de estado',           ('logica_uct', 'logica_uct_inicial', 'partida_ou_desligamento'), '#eda100'),
    ('Rampa hidráulica',           ('rampa_up_hidro', 'rampa_down_hidro'), '#e87ba4'),
    ('Rampa térmica (dualizada)',  ('rampa_up_ute', 'rampa_down_ute'), '#008300'),
    ('Tempo mínimo (dualizada)',   ('tempo_min_ligado', 'tempo_min_desligado'), '#6250d6'),
]
ORDEM_VARS = ['g_uhe', 'turb_uhe', 'vert_uhe', 'defl_uhe', 'uhe_arm', 'uch',
              'g_ute', 'uct', 'part_ute', 'desl_ute', 'fluxo', 'theta', 'deficit']
TINTA, TINTA_2, GRADE = '#0b0b0b', '#52514e', '#c3c2b7'


def montar(etapas):
    """Executa lagrange.py ate a secao EXECUCAO, com CROP_ETAPAS = etapas, e monta o modelo."""
    src = Path('lagrange.py').read_text()
    marca = '####################### EXECUCAO'
    if marca not in src or not re.search(r'^CROP_ETAPAS\s*=', src, flags=re.M):
        raise SystemExit('lagrange.py mudou: nao achei CROP_ETAPAS ou a secao EXECUCAO')
    src = re.sub(r'^CROP_ETAPAS\s*=.*$', f'CROP_ETAPAS = {etapas}', src[:src.index(marca)], flags=re.M)
    ns = {}
    exec(compile(src, 'lagrange.py', 'exec'), ns)
    m, _ = ns['construir_modelo'](True, 'matriz')
    m.update()
    return m, ns['T']


def familia_restricao(nome):
    return re.sub(r'_\d+_\d+$', '', nome.split('[')[0])


def extrair(m):
    A = m.getA().tocoo()
    vars_ = m.getVars()
    fam_v = [x.VarName.split('[')[0] for x in vars_]
    t_v = np.array([int(x.VarName.rstrip(']').split(',')[-1]) for x in vars_])
    grupo_de = {f: k for k, (_, fams, _) in enumerate(GRUPOS) for f in fams}
    fam_r = [familia_restricao(c.ConstrName) for c in m.getConstrs()]
    desconhecidas = sorted(set(fam_r) - set(grupo_de))
    if desconhecidas:
        raise SystemExit(f'familias sem grupo em GRUPOS: {desconhecidas}')
    g_r = np.array([grupo_de[f] for f in fam_r])
    # etapa de uma restricao = maior etapa entre as variaveis que ela usa
    t_r = np.full(A.shape[0], -1)
    np.maximum.at(t_r, A.row, t_v[A.col])
    return A, fam_v, t_v, fam_r, g_r, t_r


def ordenar(modo, fam_v, t_v, fam_r, g_r, t_r):
    """Permutacoes (posicao nova de cada linha/coluna original)."""
    pv = np.array([ORDEM_VARS.index(f) for f in fam_v])
    pf = np.array([[f for _, fams, _ in GRUPOS for f in fams].index(f) for f in fam_r])
    idx_r, idx_c = np.arange(len(fam_r)), np.arange(len(fam_v))
    if modo == 'familia':
        ord_r = np.lexsort((idx_r, pf, g_r))
        ord_c = np.lexsort((idx_c, pv))
    else:
        ord_r = np.lexsort((idx_r, pf, g_r, t_r))
        ord_c = np.lexsort((idx_c, pv, t_v))
    pos_r, pos_c = np.empty_like(ord_r), np.empty_like(ord_c)
    pos_r[ord_r] = np.arange(len(ord_r))
    pos_c[ord_c] = np.arange(len(ord_c))
    return pos_r, pos_c, ord_r, ord_c


def imagem(A, g_r, pos_r, pos_c, pix):
    """Imagem RGB: cada pixel com a cor do grupo de restricao dominante."""
    nr, nc = A.shape
    br, bc = min(nr, pix), min(nc, pix)
    pr = pos_r[A.row] * br // nr
    pc = pos_c[A.col] * bc // nc
    cont = np.zeros((len(GRUPOS), br, bc), dtype=np.int64)
    np.add.at(cont, (g_r[A.row], pr, pc), 1)
    rgb = np.ones((br, bc, 3))
    cores = np.array([matplotlib.colors.to_rgb(c) for _, _, c in GRUPOS])
    tem = cont.sum(axis=0) > 0
    rgb[tem] = cores[cont.argmax(axis=0)[tem]]
    return rgb


def painel(ax, modo, A, fam_v, t_v, fam_r, g_r, t_r, T, pix):
    pos_r, pos_c, ord_r, ord_c = ordenar(modo, fam_v, t_v, fam_r, g_r, t_r)
    nr, nc = A.shape
    ax.imshow(imagem(A, g_r, pos_r, pos_c, pix), interpolation='nearest', aspect='auto',
              extent=(0, nc, nr, 0))
    if modo == 'familia':
        # separadores e rotulos dos blocos de colunas (tipo de variavel)
        fv = np.array(fam_v)[ord_c]
        ini = np.flatnonzero(np.r_[True, fv[1:] != fv[:-1]])
        for x in ini[1:]:
            ax.axvline(x, color=GRADE, lw=0.5)
        fim = np.r_[ini[1:], nc]
        ax.set_xticks((ini + fim) / 2, fv[ini], rotation=90, fontsize=7)
        # separadores e rotulos dos blocos de linhas (grupo de restricoes)
        gr = g_r[ord_r]
        ini_r = np.flatnonzero(np.r_[True, gr[1:] != gr[:-1]])
        for y in ini_r[1:]:
            ax.axhline(y, color=GRADE, lw=0.5)
        fim_r = np.r_[ini_r[1:], nr]
        ax.set_yticks((ini_r + fim_r) / 2, [GRUPOS[k][0] for k in gr[ini_r]], fontsize=7)
        ax.set_title('Ordenada por família', loc='left', fontsize=10, color=TINTA)
    else:
        passo = max(1, T // 12)
        tc, tr = t_v[ord_c], t_r[ord_r]
        xs = [np.searchsorted(tc, t) for t in range(0, T, passo)]
        ys = [np.searchsorted(tr, t) for t in range(0, T, passo)]
        ax.set_xticks(xs, [f't={t}' for t in range(0, T, passo)], rotation=90, fontsize=7)
        ax.set_yticks(ys, [f't={t}' for t in range(0, T, passo)], fontsize=7)
        ax.set_title('Ordenada por etapa', loc='left', fontsize=10, color=TINTA)
    ax.set_xlabel('variáveis (colunas)', color=TINTA_2, fontsize=8)
    ax.set_ylabel('restrições (linhas)', color=TINTA_2, fontsize=8)
    ax.tick_params(colors=TINTA_2, length=2)
    for s in ax.spines.values():
        s.set_color(GRADE)


def main():
    p = argparse.ArgumentParser(description='Imagem da matriz de restricoes do despacho')
    p.add_argument('-T', '--etapas', type=int, default=24,
                   help='horas do horizonte (padrao 24; 0 = horizonte completo)')
    p.add_argument('--ordem', choices=('ambas', 'familia', 'tempo'), default='ambas')
    p.add_argument('--pix', type=int, default=1400, help='resolucao maxima da imagem (pixels)')
    p.add_argument('--saida', help='arquivo de saida (padrao figuras/8_matriz_T<T>.png)')
    a = p.parse_args()

    m, T = montar(a.etapas)
    A, fam_v, t_v, fam_r, g_r, t_r = extrair(m)
    nr, nc = A.shape
    vals = np.abs(A.data)
    print(f'T={T}: {nr} restricoes x {nc} variaveis, {A.nnz} nao-zeros '
          f'(densidade {A.nnz / (nr * nc):.2e}), |a| em [{vals.min():.3g}, {vals.max():.3g}]')
    nnz_g = Counter(g_r[A.row])
    lin_g = Counter(g_r)
    for k, (nome, _, _) in enumerate(GRUPOS):
        print(f'  {nome:28s} {lin_g.get(k, 0):7d} linhas {nnz_g.get(k, 0):8d} nao-zeros')

    modos = ['familia', 'tempo'] if a.ordem == 'ambas' else [a.ordem]
    fig, axs = plt.subplots(1, len(modos), figsize=(7.2 * len(modos), 7.6), squeeze=False)
    for ax, modo in zip(axs[0], modos):
        painel(ax, modo, A, fam_v, t_v, fam_r, g_r, t_r, T, a.pix)
    fig.legend(handles=[Patch(color=c, label=f'{n} ({lin_g.get(k, 0)})')
                        for k, (n, _, c) in enumerate(GRUPOS)],
               loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f'Matriz de restrições, T = {T} h: {nr} × {nc}, {A.nnz} não-zeros',
                 x=0.01, ha='left', fontsize=11, color=TINTA)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    saida = Path(a.saida or f'figuras/8_matriz_T{T}.png')
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=200)
    print(f'{saida} gerado')


if __name__ == '__main__':
    main()
