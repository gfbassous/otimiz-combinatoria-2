"""
Modelo MIP completo do despacho hidrotermico (IEEE 14 barras) em PYOMO + utilitarios
de solver, compartilhados por branch_bound_pyomo.py e branch_cut_pyomo.py.

Mesma formulacao de branch&bound.py / branch&cut.py (rampas e tempo minimo
ficam no modelo; nada e dualizado).

Solver: 1o argumento de linha de comando de quem importa (ver escolher_solver()).
"""

import sys
import time
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition as TC

####################### DADOS #######################
CROP_ETAPAS = 0 

barras = [f'B{i}' for i in range(1, 15)]
linhas = [('B1', 'B2'), ('B1', 'B5'), ('B2', 'B3'), ('B2', 'B4'), ('B2', 'B5'),
          ('B3', 'B4'), ('B4', 'B5'), ('B4', 'B7'), ('B4', 'B9'), ('B5', 'B6'),
          ('B6', 'B11'), ('B6', 'B12'), ('B6', 'B13'), ('B7', 'B8'), ('B7', 'B9'),
          ('B9', 'B10'), ('B9', 'B14'), ('B10', 'B11'), ('B12', 'B13'), ('B13', 'B14')]
uhe = ['UHE1', 'UHE2', 'UHE3', 'UHE4']
ute = ['UTE1', 'UTE2', 'UTE3', 'UTE4', 'UTE5']


def ler_matriz(sheet, indices):
    df = pd.read_excel('dados.xlsx', sheet_name=sheet)
    df.columns = [str(c) for c in df.columns]
    if set(indices).issubset(df.columns):
        return df[indices].to_numpy(dtype=float).T
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    return df.loc[indices].to_numpy(dtype=float)


vaz_incr = ler_matriz('vazao', uhe)
demanda = ler_matriz('demanda', barras)
if vaz_incr.shape[1] != demanda.shape[1]:
    raise ValueError('vazao e demanda com horizontes diferentes em dados.xlsx')
if CROP_ETAPAS > 0:
    vaz_incr = vaz_incr[:, :CROP_ETAPAS]
    demanda = demanda[:, :CROP_ETAPAS]

T = demanda.shape[1]
N, L, H, G = len(barras), len(linhas), len(uhe), len(ute)

S_BASE = 100.
x_linhas = np.array([0.05917, 0.22304, 0.19797, 0.17632, 0.17388,
                     0.17103, 0.04211, 0.20912, 0.55618, 0.25202,
                     0.19890, 0.25581, 0.13027, 0.17615, 0.11001,
                     0.08450, 0.27038, 0.19207, 0.19988, 0.34802])
Bl = S_BASE / x_linhas
capacidade_linhas = np.array([190., 100., 120., 100., 90.,
                              80., 130., 90., 45., 110.,
                              55., 40., 48., 55., 100.,
                              55., 38., 50., 30., 32.])
c_deficit = 3000.
barra_ref = 0

uhe_pot = np.array([70., 95., 110., 120.])
uhe_g_min = 0.2 * uhe_pot
uhe_arm_cap = np.array([900., 460., 300., 170.])
uhe_barra = ['B1', 'B2', 'B6', 'B7']
prod_uhe = np.array([1., 1., 1., 1.])
defl_min = np.array([6., 10., 14., 18.])
uhe_montante = [None, 'UHE1', 'UHE2', 'UHE3']

ute_pot = np.array([110., 85., 70., 60., 45.])
ute_cvu = np.array([42., 85., 130., 210., 450.])
ute_g_min = np.array([22., 25., 22., 15., 5.])  # <= rampa (senao a unidade nao consegue partir/parar)
ute_rampa = np.array([22., 25., 30., 40., 45.])
ute_barra = ['B3', 'B6', 'B8', 'B2', 'B13']
ute_custo_partida = np.array([9000., 4500., 2600., 1300., 450.])
t_on = np.array([6, 4, 3, 2, 2])
t_off = np.array([5, 3, 2, 2, 2])
uct_inicial = np.array([1., 0., 0., 0., 0.])

idx_barra = {b: k for k, b in enumerate(barras)}
A = np.zeros((N, L))
for l, (i, j) in enumerate(linhas):
    A[idx_barra[i], l] = +1
    A[idx_barra[j], l] = -1
linhas_de = [[l for l in range(L) if A[n, l] != 0] for n in range(N)]
uhe_na_barra = [[u for u in range(H) if uhe_barra[u] == barras[n]] for n in range(N)]
ute_na_barra = [[g for g in range(G) if ute_barra[g] == barras[n]] for n in range(N)]


####################### MODELO #######################
def construir_modelo():
    m = pyo.ConcreteModel()
    Hs, Gs, Ls, Ns, Ts = range(H), range(G), range(L), range(N), range(T)

    m.g_uhe = pyo.Var(Hs, Ts, bounds=lambda m, u, t: (0, uhe_pot[u]))
    m.uch = pyo.Var(Hs, Ts, within=pyo.Binary)
    m.turb = pyo.Var(Hs, Ts, within=pyo.NonNegativeReals)
    m.vert = pyo.Var(Hs, Ts, within=pyo.NonNegativeReals)
    m.defl = pyo.Var(Hs, Ts, bounds=lambda m, u, t: (defl_min[u], None))
    m.arm = pyo.Var(Hs, Ts, bounds=lambda m, u, t: (0, uhe_arm_cap[u]))

    m.g_ute = pyo.Var(Gs, Ts, within=pyo.NonNegativeReals)
    m.uct = pyo.Var(Gs, Ts, within=pyo.Binary)
    m.part = pyo.Var(Gs, Ts, within=pyo.Binary)
    m.desl = pyo.Var(Gs, Ts, within=pyo.Binary)

    m.f = pyo.Var(Ls, Ts, bounds=lambda m, l, t: (-capacidade_linhas[l], capacidade_linhas[l]))
    m.theta = pyo.Var(Ns, Ts)
    m.deficit = pyo.Var(Ns, Ts, within=pyo.NonNegativeReals)

    m.KCL = pyo.Constraint(Ns, Ts, rule=lambda m, n, t: (
        sum(m.g_uhe[u, t] for u in uhe_na_barra[n])
        + sum(m.g_ute[g, t] for g in ute_na_barra[n])
        + m.deficit[n, t]
        - sum(A[n, l] * m.f[l, t] for l in linhas_de[n])) == demanda[n, t])
    m.KVL = pyo.Constraint(Ls, Ts, rule=lambda m, l, t: m.f[l, t] == Bl[l] * sum(
        A[n, l] * m.theta[n, t] for n in Ns if A[n, l] != 0))
    m.ref = pyo.Constraint(Ts, rule=lambda m, t: m.theta[barra_ref, t] == 0)

    m.bal_hid = pyo.Constraint(Hs, Ts, rule=lambda m, u, t: m.turb[u, t] + m.vert[u, t] == m.defl[u, t])
    m.prod = pyo.Constraint(Hs, Ts, rule=lambda m, u, t: m.g_uhe[u, t] == prod_uhe[u] * m.turb[u, t])

    def bal_arm(m, u, t):
        ant = uhe_arm_cap[u] / 2 if t == 0 else m.arm[u, t - 1]
        mont = 0 if uhe_montante[u] is None else m.defl[uhe.index(uhe_montante[u]), t]
        return m.arm[u, t] == ant + vaz_incr[u, t] - m.turb[u, t] - m.vert[u, t] + mont
    m.bal_arm = pyo.Constraint(Hs, Ts, rule=bal_arm)
    m.arm_final = pyo.Constraint(Hs, rule=lambda m, u: m.arm[u, T - 1] >= uhe_arm_cap[u] / 2)

    m.cap_h = pyo.Constraint(Hs, Ts, rule=lambda m, u, t: m.g_uhe[u, t] <= uhe_pot[u] * m.uch[u, t])
    m.min_h = pyo.Constraint(Hs, Ts, rule=lambda m, u, t: m.g_uhe[u, t] >= uhe_g_min[u] * m.uch[u, t])
    m.hup = pyo.Constraint(Hs, range(1, T), rule=lambda m, u, t: m.defl[u, t] - m.defl[u, t - 1] <= 0.1 * uhe_pot[u])
    m.hdn = pyo.Constraint(Hs, range(1, T), rule=lambda m, u, t: m.defl[u, t - 1] - m.defl[u, t] <= 0.1 * uhe_pot[u])

    m.cap_t = pyo.Constraint(Gs, Ts, rule=lambda m, g, t: m.g_ute[g, t] <= ute_pot[g] * m.uct[g, t])
    m.min_t = pyo.Constraint(Gs, Ts, rule=lambda m, g, t: m.g_ute[g, t] >= ute_g_min[g] * m.uct[g, t])
    m.rup = pyo.Constraint(Gs, range(1, T), rule=lambda m, g, t: m.g_ute[g, t] - m.g_ute[g, t - 1] <= ute_rampa[g])
    m.rdn = pyo.Constraint(Gs, range(1, T), rule=lambda m, g, t: m.g_ute[g, t - 1] - m.g_ute[g, t] <= ute_rampa[g])

    def logica(m, g, t):
        ant = uct_inicial[g] if t == 0 else m.uct[g, t - 1]
        return m.uct[g, t] - ant == m.part[g, t] - m.desl[g, t]
    m.logica = pyo.Constraint(Gs, Ts, rule=logica)
    m.part_ou_desl = pyo.Constraint(Gs, Ts, rule=lambda m, g, t: m.part[g, t] + m.desl[g, t] <= 1)

    jan_on = [(g, t) for g in Gs for t in range(t_on[g] - 1, T)]
    jan_off = [(g, t) for g in Gs for t in range(t_off[g] - 1, T)]
    m.on = pyo.Constraint(jan_on, rule=lambda m, g, t: sum(
        m.part[g, s] for s in range(t - t_on[g] + 1, t + 1)) <= m.uct[g, t])
    m.off = pyo.Constraint(jan_off, rule=lambda m, g, t: sum(
        m.desl[g, s] for s in range(t - t_off[g] + 1, t + 1)) <= 1 - m.uct[g, t])

    m.obj = pyo.Objective(expr=sum(ute_cvu[g] * m.g_ute[g, t] + ute_custo_partida[g] * m.part[g, t]
                                   for g in Gs for t in Ts)
                          + c_deficit * sum(m.deficit[n, t] for n in Ns for t in Ts),
                          sense=pyo.minimize)
    return m


def relaxar(m):
    """Copia do modelo com as binarias relaxadas para [0,1]."""
    r = m.clone()
    pyo.TransformationFactory('core.relax_integer_vars').apply_to(r)
    return r


####################### SOLVER #######################
def escolher_solver(padrao='appsi_highs'):
    return sys.argv[1] if len(sys.argv) > 1 else padrao


# tempo/gap/threads + chaves para DESLIGAR presolve, cortes e heuristicas do solver
# (para o B&B ficar "puro", como em branch&bound.py). O que nao tem chave aqui
# nao e desligado -- o aviso aparece na tela.
OPCOES = {
    'gurobi': dict(tempo='TimeLimit', gap='MIPGap', base={'Threads': 1},
                   puro={'Presolve': 0, 'Cuts': 0, 'Heuristics': 0}),
    'gurobi_direct': dict(tempo='TimeLimit', gap='MIPGap', base={'Threads': 1},
                          puro={'Presolve': 0, 'Cuts': 0, 'Heuristics': 0}),
    'gurobi_persistent': dict(tempo='TimeLimit', gap='MIPGap', base={'Threads': 1},
                              puro={'Presolve': 0, 'Cuts': 0, 'Heuristics': 0}),
    'appsi_highs': dict(tempo='time_limit', gap='mip_rel_gap', base={'threads': 1},
                        puro={'presolve': 'off', 'mip_heuristic_effort': 0.0}),
    'cbc': dict(tempo='seconds', gap='ratioGap', base={'threads': 1},
                puro={'preprocess': 'off', 'cuts': 'off', 'heuristics': 'off'}),
    'scip': dict(tempo='limits/time', gap='limits/gap', base={},
                 puro={'presolving/maxrounds': 0, 'separating/maxrounds': 0,
                       'separating/maxroundsroot': 0}),
    'cplex': dict(tempo='timelimit', gap='mipgap', base={'threads': 1}, puro={}),
    'glpk': dict(tempo='tmlim', gap='mipgap', base={}, puro={}),
}


def criar_solver(nome, tempo, gap=1e-3, puro=True):
    s = pyo.SolverFactory(nome)
    if not s.available(exception_flag=False):
        raise SystemExit(f"Solver '{nome}' indisponivel neste ambiente.")
    op = OPCOES.get(nome)
    if op is None:
        print(f"[aviso] sem mapa de opcoes para '{nome}' -- sem limites de tempo/gap")
        return s
    chaves = {op['tempo']: tempo, op['gap']: gap, **op['base']}
    if puro:
        chaves.update(op['puro'])
        if not op['puro']:
            print(f"[aviso] nao sei desligar presolve/cortes em '{nome}'")
    for k, v in chaves.items():
        s.options[k] = v
    return s


def resolver(solver, nome, modelo):
    """Resolve e devolve dict(tem, obj, bound, status, tempo). Nao levanta se nao ha solucao.

    Usa a interface legada de SolverFactory (vale tambem para appsi_highs).
    """
    t0 = time.time()
    out = dict(tem=False, obj=None, bound=None, status='erro', tempo=0.0)
    try:
        res = solver.solve(modelo, load_solutions=False)
        out['status'] = str(res.solver.termination_condition)
        b = res.problem.lower_bound
        out['bound'] = b if b is not None and np.isfinite(b) else None
        if len(res.solution) > 0:
            modelo.solutions.load_from(res)
            out['tem'] = True
    except Exception as e:
        print(f"[erro solver] {e}")
    out['tempo'] = time.time() - t0
    if out['tem']:
        out['obj'] = pyo.value(modelo.obj)
        if out['bound'] is None and out['status'].endswith('optimal'):
            out['bound'] = out['obj']  # solver nao reportou bound; otimo => bound ~ obj
    return out


def relatorio(titulo, r):
    print(f"\n[{titulo}] status={r['status']}  tempo={r['tempo']:.1f}s")
    if r['tem']:
        print(f"  objetivo (UB) = {r['obj']:,.2f}")
    if r['bound'] is not None:
        print(f"  bound    (LB) = {r['bound']:,.2f}")
    if r['tem'] and r['bound'] is not None:
        print(f"  gap           = {100 * (r['obj'] - r['bound']) / max(abs(r['obj']), 1e-9):.4f}%")


def arr(var, n):
    return np.array([[pyo.value(var[i, t]) for t in range(T)] for i in range(n)])
