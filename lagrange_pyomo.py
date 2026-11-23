"""
Relaxacao lagrangeana (metodo do subgradiente) do despacho hidrotermico em PYOMO,
para poder trocar de solver (gurobi, appsi_highs, cbc, glpk, scip, cplex, ...).

Mesma formulacao de lagrange.py (rampas e tempo minimo dualizados). Ver a
documentacao la; aqui so muda a camada de modelagem/solver.

    python lagrange_pyomo.py                 # solver padrao (SOLVER abaixo)
    python lagrange_pyomo.py appsi_highs     # escolhe o solver pela linha de comando
    python lagrange_pyomo.py cbc

Limite inferior valido: usa o melhor bound reportado pelo solver
(results.problem.lower_bound). Se o solver nao reportar bound, cai para o
valor do objetivo quando o subproblema foi resolvido ate a otimalidade
(aproximacao -- avisado na tela).
"""

import sys
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition as TC

###################### SOLVER ######################
SOLVER = sys.argv[1] if len(sys.argv) > 1 else 'gurobi'

# Nome da opcao de tempo/gap em cada solver (ajuste/adicione os que for testar).
OPCOES = {
    'gurobi':          dict(tempo='TimeLimit', gap='MIPGap', extra={'Threads': 1}),
    'gurobi_direct':   dict(tempo='TimeLimit', gap='MIPGap', extra={'Threads': 1}),
    'gurobi_persistent': dict(tempo='TimeLimit', gap='MIPGap', extra={'Threads': 1}),
    'appsi_highs':     dict(tempo='time_limit', gap='mip_rel_gap', extra={'threads': 1}),
    'highs':           dict(tempo='time_limit', gap='mip_rel_gap', extra={}),
    'cplex':           dict(tempo='timelimit', gap='mipgap', extra={'threads': 1}),
    'cbc':             dict(tempo='seconds', gap='ratioGap', extra={'threads': 1}),
    'glpk':            dict(tempo='tmlim', gap='mipgap', extra={}),
    'scip':            dict(tempo='limits/time', gap='limits/gap', extra={}),
    'xpress':          dict(tempo='maxtime', gap='miprelstop', extra={}),
}


def novo_solver(tempo, gap=None):
    s = pyo.SolverFactory(SOLVER)
    if not s.available(exception_flag=False):
        raise SystemExit(f"Solver '{SOLVER}' indisponivel neste ambiente.")
    op = OPCOES.get(SOLVER)
    if op is None:
        print(f"[aviso] sem mapa de opcoes para '{SOLVER}' -- rodando sem limites")
        return s
    s.options[op['tempo']] = tempo
    if gap is not None:
        s.options[op['gap']] = gap
    for k, val in op['extra'].items():
        s.options[k] = val
    return s


def resolver(solver, modelo):
    """Resolve e devolve (tem_solucao, objetivo, bound). Nao levanta se nao ha solucao."""
    try:
        res = solver.solve(modelo, load_solutions=False)
    except Exception as e:  # solver sem suporte a algum argumento etc.
        print(f"[erro solver] {e}")
        return False, None, None
    tc = res.solver.termination_condition
    tem = len(res.solution) > 0 if hasattr(res, 'solution') else False
    if tem:
        try:
            modelo.solutions.load_from(res)
        except Exception:
            tem = False
    obj = pyo.value(modelo.obj) if tem else None
    lb = None
    try:
        lb = res.problem.lower_bound
        if lb is None or not np.isfinite(lb):
            lb = None
    except Exception:
        lb = None
    if lb is None and tem and tc == TC.optimal:
        lb = obj  # aproximacao: sem bound reportado
        modelo._bound_aprox = True
    else:
        modelo._bound_aprox = False
    return tem, obj, lb


###################### PARAMETROS DO SUBGRADIENTE ######################
MAX_ITER = 40
RHO_0 = 2.0
RHO_MIN = 1e-3
PACIENCIA = 4
TOL_VIAB = 1e-6

GAP_SUB = 1e-3
TEMPO_SUB = 60
TEMPO_UB = 120
TEMPO_LP = 300

DUALIZAR_RAMPA_HIDRO = True
CROP_ETAPAS = 0

####################### DADOS #######################
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
Bl = S_BASE / x_linhas  # diagonal de B
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


def rep(v):
    return np.repeat(np.asarray(v, dtype=float)[:, None], T, axis=1)


idx_barra = {b: k for k, b in enumerate(barras)}
A = np.zeros((N, L))
for l, (i, j) in enumerate(linhas):
    A[idx_barra[i], l] = +1
    A[idx_barra[j], l] = -1

# listas de adjacencia (evita somas densas)
linhas_de = [[l for l in range(L) if A[n, l] != 0] for n in range(N)]
uhe_na_barra = [[u for u in range(H) if uhe_barra[u] == barras[n]] for n in range(N)]
ute_na_barra = [[g for g in range(G) if ute_barra[g] == barras[n]] for n in range(N)]
jusante_de = {u: [d for d in range(H) if uhe_montante[d] == uhe[u]] for u in range(H)}  # U[d,u]=1

R_UTE = rep(ute_rampa)[:, 1:]
R_HID = rep(0.1 * uhe_pot)[:, 1:]

jan_on = [(g, t, t - t_on[g] + 1) for g in range(G) for t in range(t_on[g] - 1, T)]
jan_off = [(g, t, t - t_off[g] + 1) for g in range(G) for t in range(t_off[g] - 1, T)]

FORMATOS = dict(rup=(G, T - 1), rdn=(G, T - 1),
                hup=(H, T - 1), hdn=(H, T - 1),
                on=(G, T), off=(G, T))
if not DUALIZAR_RAMPA_HIDRO:
    FORMATOS.pop('hup')
    FORMATOS.pop('hdn')


####################### MODELO #######################
def construir_modelo(incluir_dualizadas):
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

    # --- X ---
    def kcl(m, n, t):
        return (sum(m.g_uhe[u, t] for u in uhe_na_barra[n])
                + sum(m.g_ute[g, t] for g in ute_na_barra[n])
                + m.deficit[n, t]
                - sum(A[n, l] * m.f[l, t] for l in linhas_de[n])) == demanda[n, t]
    m.KCL = pyo.Constraint(Ns, Ts, rule=kcl)

    def kvl(m, l, t):
        return m.f[l, t] == Bl[l] * sum(A[n, l] * m.theta[n, t] for n in Ns if A[n, l] != 0)
    m.KVL = pyo.Constraint(Ls, Ts, rule=kvl)
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
    m.cap_t = pyo.Constraint(Gs, Ts, rule=lambda m, g, t: m.g_ute[g, t] <= ute_pot[g] * m.uct[g, t])
    m.min_t = pyo.Constraint(Gs, Ts, rule=lambda m, g, t: m.g_ute[g, t] >= ute_g_min[g] * m.uct[g, t])

    def logica(m, g, t):
        ant = uct_inicial[g] if t == 0 else m.uct[g, t - 1]
        return m.uct[g, t] - ant == m.part[g, t] - m.desl[g, t]
    m.logica = pyo.Constraint(Gs, Ts, rule=logica)
    m.part_ou_desl = pyo.Constraint(Gs, Ts, rule=lambda m, g, t: m.part[g, t] + m.desl[g, t] <= 1)

    # --- dualizadas (so no modelo completo) ---
    if incluir_dualizadas:
        m.rup = pyo.Constraint(Gs, range(1, T), rule=lambda m, g, t: m.g_ute[g, t] - m.g_ute[g, t - 1] <= ute_rampa[g])
        m.rdn = pyo.Constraint(Gs, range(1, T), rule=lambda m, g, t: m.g_ute[g, t - 1] - m.g_ute[g, t] <= ute_rampa[g])
        m.hup = pyo.Constraint(Hs, range(1, T), rule=lambda m, u, t: m.defl[u, t] - m.defl[u, t - 1] <= 0.1 * uhe_pot[u])
        m.hdn = pyo.Constraint(Hs, range(1, T), rule=lambda m, u, t: m.defl[u, t - 1] - m.defl[u, t] <= 0.1 * uhe_pot[u])
        m.on = pyo.Constraint(jan_on, rule=lambda m, g, t, ini: sum(m.part[g, s] for s in range(ini, t + 1)) <= m.uct[g, t])
        m.off = pyo.Constraint(jan_off, rule=lambda m, g, t, ini: sum(m.desl[g, s] for s in range(ini, t + 1)) <= 1 - m.uct[g, t])
    elif not DUALIZAR_RAMPA_HIDRO:
        m.hup = pyo.Constraint(Hs, range(1, T), rule=lambda m, u, t: m.defl[u, t] - m.defl[u, t - 1] <= 0.1 * uhe_pot[u])
        m.hdn = pyo.Constraint(Hs, range(1, T), rule=lambda m, u, t: m.defl[u, t - 1] - m.defl[u, t] <= 0.1 * uhe_pot[u])

    m.obj = pyo.Objective(expr=sum(ute_cvu[g] * m.g_ute[g, t] + ute_custo_partida[g] * m.part[g, t]
                                   for g in Gs for t in Ts)
                          + c_deficit * sum(m.deficit[n, t] for n in Ns for t in Ts),
                          sense=pyo.minimize)
    return m


def arr(var, n):
    return np.array([[pyo.value(var[i, t]) for t in range(T)] for i in range(n)])


################ LAGRANGEANO ################
def violacoes(gu, dfl, prt, dsl, u):
    s = dict(rup=gu[:, 1:] - gu[:, :-1] - R_UTE,
             rdn=gu[:, :-1] - gu[:, 1:] - R_UTE,
             on=np.zeros((G, T)), off=np.zeros((G, T)))
    if DUALIZAR_RAMPA_HIDRO:
        s['hup'] = dfl[:, 1:] - dfl[:, :-1] - R_HID
        s['hdn'] = dfl[:, :-1] - dfl[:, 1:] - R_HID
    for g, t, ini in jan_on:
        s['on'][g, t] = prt[g, ini:t + 1].sum() - u[g, t]
    for g, t, ini in jan_off:
        s['off'][g, t] = dsl[g, ini:t + 1].sum() + u[g, t] - 1.0
    return s


def coeficientes(lam):
    """Custos lineares perturbados (c + A'lam) e constante -lam'b."""
    d_ute = lam['rup'] - lam['rdn']
    c_gu = np.zeros((G, T)); c_gu[:, 1:] += d_ute; c_gu[:, :-1] -= d_ute
    const = float((-(lam['rup'] + lam['rdn']) * R_UTE).sum())

    c_df = np.zeros((H, T))
    if DUALIZAR_RAMPA_HIDRO:
        d_hid = lam['hup'] - lam['hdn']
        c_df[:, 1:] += d_hid; c_df[:, :-1] -= d_hid
        const += float((-(lam['hup'] + lam['hdn']) * R_HID).sum())

    c_prt = np.zeros((G, T)); c_dsl = np.zeros((G, T)); c_uct = np.zeros((G, T))
    for g, t, ini in jan_on:
        c_prt[g, ini:t + 1] += lam['on'][g, t]
        c_uct[g, t] -= lam['on'][g, t]
    for g, t, ini in jan_off:
        c_dsl[g, ini:t + 1] += lam['off'][g, t]
        c_uct[g, t] += lam['off'][g, t]
    const -= float(lam['off'].sum())
    return c_gu, c_df, c_prt, c_dsl, c_uct, const


def setar_objetivo_lagrangeano(m, lam):
    c_gu, c_df, c_prt, c_dsl, c_uct, const = coeficientes(lam)
    cg, cp = rep(ute_cvu) + c_gu, rep(ute_custo_partida) + c_prt
    expr = const
    expr += sum(cg[g, t] * m.g_ute[g, t] + cp[g, t] * m.part[g, t]
                + c_dsl[g, t] * m.desl[g, t] + c_uct[g, t] * m.uct[g, t]
                for g in range(G) for t in range(T))
    expr += c_deficit * sum(m.deficit[n, t] for n in range(N) for t in range(T))
    expr += sum(c_df[u, t] * m.defl[u, t] for u in range(H) for t in range(T))
    m.obj.expr = expr  # Objective existente: so troca a expressao


####################### EXECUCAO #######################
print(f"Solver: {SOLVER}")
print(f"Sistema: N={N} barras, L={L} linhas, H={H} UHE, G={G} UTE, T={T} etapas")
n_dual = sum(int(np.prod(f)) for f in FORMATOS.values())
print(f"Restricoes dualizadas: {n_dual}\n")

### 1) Referencias ###
m_full = construir_modelo(True)

m_lp = m_full.clone()
pyo.TransformationFactory('core.relax_integer_vars').apply_to(m_lp)
tem, z_lp, _ = resolver(novo_solver(TEMPO_LP), m_lp)
z_lp = z_lp if tem else float('nan')
print(f"[ref] relaxacao linear  z_LP  = {z_lp:,.2f}")

tem, UB, bnd = resolver(novo_solver(TEMPO_UB, GAP_SUB), m_full)
if tem:
    print(f"[ref] incumbente MIP    UB    = {UB:,.2f}  (bound {bnd if bnd is not None else float('nan'):,.2f})\n")
else:
    UB = None
    print("[ref] MIP completo sem incumbente no tempo dado -- passo usa alvo estimado\n")

### 2) Subproblema ###
m_rel = construir_modelo(False)
solver_sub = novo_solver(TEMPO_SUB, GAP_SUB)

### 3) Subgradiente ###
lam = {k: np.zeros(f) for k, f in FORMATOS.items()}
melhor_lb = -np.inf
melhor_lam = {k: a.copy() for k, a in lam.items()}
rho = RHO_0
sem_melhora = 0
hist = []
avisou_aprox = False

print(f"{'it':>3} {'L(lam)':>14} {'melhor LB':>14} {'||s||':>10} "
      f"{'max viol':>10} {'rho':>7} {'alpha':>10}")
print("-" * 74)

for it in range(MAX_ITER):
    setar_objetivo_lagrangeano(m_rel, lam)
    tem, obj, lb = resolver(solver_sub, m_rel)
    if not tem or lb is None:
        print(f"{it:>3}  subproblema sem solucao/bound -- parando")
        break
    if m_rel._bound_aprox and not avisou_aprox:
        print("[aviso] solver nao reportou bound; usando o objetivo como L(lam) (aproximado)")
        avisou_aprox = True

    L_val = lb
    s = violacoes(arr(m_rel.g_ute, G), arr(m_rel.defl, H), arr(m_rel.part, G),
                  arr(m_rel.desl, G), arr(m_rel.uct, G))
    norma2 = float(sum((a ** 2).sum() for a in s.values()))
    max_viol = float(max(a.max() for a in s.values()))

    if L_val > melhor_lb + 1e-9:
        melhor_lb = L_val
        melhor_lam = {k: a.copy() for k, a in lam.items()}
        sem_melhora = 0
    else:
        sem_melhora += 1
        if sem_melhora >= PACIENCIA:
            rho = max(rho / 2.0, RHO_MIN)
            sem_melhora = 0

    if norma2 <= TOL_VIAB:
        print(f"{it:>3} {L_val:>14,.2f} {melhor_lb:>14,.2f} {0.0:>10.4f} "
              f"{max_viol:>10.4f} {rho:>7.4f} {'--':>10}")
        print("\nSubgradiente nulo: solucao do subproblema viavel => L(lam) = z*.")
        break

    alvo = UB if UB is not None else melhor_lb + 0.1 * max(abs(melhor_lb), 1.0)
    alpha = max(rho * (alvo - L_val) / norma2, 0.0)

    print(f"{it:>3} {L_val:>14,.2f} {melhor_lb:>14,.2f} {np.sqrt(norma2):>10.4f} "
          f"{max_viol:>10.4f} {rho:>7.4f} {alpha:>10.3e}")
    hist.append((it, L_val, melhor_lb, np.sqrt(norma2), max_viol, rho, alpha))

    for k in lam:
        lam[k] = np.maximum(0.0, lam[k] + alpha * s[k])

    if rho <= RHO_MIN:
        print("\nrho no minimo -- parando.")
        break

### 4) Resumo ###
z_ld = melhor_lb
print("\n" + "=" * 62)
print(f"RESUMO  (solver: {SOLVER})")
print("=" * 62)
print(f"  z_LP  (relaxacao linear)      = {z_lp:>16,.2f}")
print(f"  z_LD  (dual lagrangeano)      = {z_ld:>16,.2f}")
if UB is not None:
    print(f"  UB    (incumbente MIP)        = {UB:>16,.2f}")
    print(f"  gap lagrangeano  (UB-z_LD)/UB = {100 * (UB - z_ld) / abs(UB):>15.2f}%")
    print(f"  gap linear       (UB-z_LP)/UB = {100 * (UB - z_lp) / abs(UB):>15.2f}%")

if hist:
    h = np.array(hist)
    np.savez(f'lagrange_pyomo_{SOLVER}.npz',
             iteracao=h[:, 0], L=h[:, 1], melhor_lb=h[:, 2],
             norma_subgradiente=h[:, 3], max_violacao=h[:, 4], rho=h[:, 5], alpha=h[:, 6],
             z_lp=z_lp, z_ld=z_ld, ub=UB if UB is not None else np.nan,
             **{f'lam_{k}': a for k, a in melhor_lam.items()})
    print(f"\nlagrange_pyomo_{SOLVER}.npz gerado.")
