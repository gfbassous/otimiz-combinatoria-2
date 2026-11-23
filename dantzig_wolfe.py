"""
Geracao de colunas (decomposicao de DANTZIG-WOLFE) do despacho hidrotermico,
mesmo problema de branch&bound.py / lagrange.py.

============================== A DECOMPOSICAO ==============================

MESTRE: toda a operacao hidraulica (g_uhe, uch, turb, vert, defl, arm e suas
restricoes, inclusive a rampa de defluencia), toda a transmissao (fluxo, theta,
KVL, barra de referencia) e o deficit.

SUBPROBLEMAS X^g, um por UTE g (independentes entre si):

    X^g = { (p, u, y, z) :  pmin_g u_t <= p_t <= pmax_g u_t
                            u_t - u_{t-1} = y_t - z_t,   y_t + z_t <= 1
                            sum_{s=t-ton+1}^{t}  y_s <= u_t        (tempo min. ligado)
                            sum_{s=t-toff+1}^{t} z_s <= 1 - u_t    (tempo min. desligado)
                            |p_t - p_{t-1}| <= R_g                 (rampa)
                            u, y, z binarias }

A UNICA restricao que liga as UTEs ao resto e a 1a lei de Kirchhoff na barra de
cada UTE. Cada x^g e escrito como combinacao convexa de pontos (colunas) de X^g:

    x^g = sum_k lam_gk x^gk,   sum_k lam_gk = 1,   lam >= 0

MESTRE RESTRITO (RMP), so com as colunas ja geradas:

    min  sum_g sum_k c_gk lam_gk + c_def sum deficit
    s.a. Mh g_uhe + sum_g e_b(g) sum_k p^gk lam_gk + deficit - A f = D    (duais pi)
         hidraulica + rede, iguais ao modelo original (uch relaxada em [0,1])
         sum_k lam_gk = 1                                                (duais mu_g)

    com c_gk = sum_t (cvu_g p^gk_t + cpart_g y^gk_t).

PRICING da UTE g (MIP pequeno: 4T variaveis):

    rc_g = min_{x in X^g}  sum_t (cvu_g - pi_{b(g),t}) p_t + cpart_g y_t  -  mu_g

    rc_g < 0  ->  a coluna entra no RMP.   Nenhuma entra  ->  RMP otimo = z_DW.

LIMITE DE LAGRANGE a cada iteracao (vale mesmo antes de convergir):

    z_RMP + sum_g min(0, rc_g)  <=  z_DW

(usa o ObjBound do pricing, entao continua valido com MIPGap > 0).

RELACOES ENTRE LIMITES

    z_LP  <=  z_DW  <=  z*

z_DW e o LP sobre conv(X^g) para as UTEs (e o LP comum para a hidraulica). E igual
ao dual lagrangeano que dualiza a 1a lei nas barras das UTEs mantendo X^g. Pode ser
> z_LP porque X^g com rampa, geracao minima e tempo minimo NAO tem a propriedade da
integralidade (as desigualdades de tempo minimo sozinhas ja descrevem a envoltoria,
mas combinadas com p e rampa nao).

ARTIFICIAIS. O RMP ganha variaveis com custo grande que so servem para mante-lo
viavel quando o branch-and-price restringe as colunas (branch_and_price.py):
    a_g      na convexidade   (sum_k lam_gk + a_g = 1)
    exc_n,t  na 1a lei        (sobregeracao)
Se, apos convergir, alguma ficar > 0, o no e inviavel. Na raiz elas ficam em 0.

    python dantzig_wolfe.py
"""

import time
from types import SimpleNamespace

import gurobipy as gp
import numpy as np
import pandas as pd

GRB = gp.GRB

###################### PARAMETROS ######################
CROP_ETAPAS = 24          # 0 = horizonte completo de dados.xlsx
MAX_ITER_CG = 500         # iteracoes de geracao de colunas
COLUNAS_POR_PRICING = 5   # solucoes do pool do pricing que podem entrar por iteracao
TOL_RC = 1e-6             # custo reduzido minimo (negativo) para uma coluna entrar
GAP_CG = 1e-7             # para quando (z_RMP - LB_lagrange) <= GAP_CG |z_RMP|
TEMPO_PRICING = 60        # s por pricing
TEMPO_REF = 300           # s do MIP compacto de referencia (z* / UB)
TEMPO_HEUR = 60           # s do mestre restrito inteiro (heuristica de UB)

####################### DADOS (iguais a lagrange.py) #######################
barras = [f'B{i}' for i in range(1, 15)]
linhas = [('B1', 'B2'), ('B1', 'B5'), ('B2', 'B3'), ('B2', 'B4'), ('B2', 'B5'),
          ('B3', 'B4'), ('B4', 'B5'), ('B4', 'B7'), ('B4', 'B9'), ('B5', 'B6'),
          ('B6', 'B11'), ('B6', 'B12'), ('B6', 'B13'), ('B7', 'B8'), ('B7', 'B9'),
          ('B9', 'B10'), ('B9', 'B14'), ('B10', 'B11'), ('B12', 'B13'), ('B13', 'B14')]
uhe = ['UHE1', 'UHE2', 'UHE3', 'UHE4']
ute = ['UTE1', 'UTE2', 'UTE3', 'UTE4', 'UTE5']


def ler_matriz(sheet, indices):
    """Le a planilha e devolve a matriz (len(indices), T), independente da orientacao."""
    df = pd.read_excel('dados.xlsx', sheet_name=sheet)
    df.columns = [str(c) for c in df.columns]
    if set(indices).issubset(df.columns):
        return df[indices].to_numpy(dtype=float).T
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    return df.loc[indices].to_numpy(dtype=float)


vaz_incr = ler_matriz('vazao', uhe)
demanda = ler_matriz('demanda', barras)
if CROP_ETAPAS > 0:
    vaz_incr, demanda = vaz_incr[:, :CROP_ETAPAS], demanda[:, :CROP_ETAPAS]
N, L, H, G, T = len(barras), len(linhas), len(uhe), len(ute), demanda.shape[1]

S_BASE = 100.
x_linhas = np.array([0.05917, 0.22304, 0.19797, 0.17632, 0.17388,
                     0.17103, 0.04211, 0.20912, 0.55618, 0.25202,
                     0.19890, 0.25581, 0.13027, 0.17615, 0.11001,
                     0.08450, 0.27038, 0.19207, 0.19988, 0.34802])
B = np.diag(S_BASE / x_linhas)
capacidade_linhas = np.array([190., 100., 120., 100., 90., 80., 130., 90., 45., 110.,
                              55., 40., 48., 55., 100., 55., 38., 50., 30., 32.])
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
ute_g_min = np.array([22., 25., 22., 15., 5.])
ute_rampa = np.array([22., 25., 30., 40., 45.])
ute_barra = ['B3', 'B6', 'B8', 'B2', 'B13']
ute_custo_partida = np.array([9000., 4500., 2600., 1300., 450.])
t_on = np.array([6, 4, 3, 2, 2])
t_off = np.array([5, 3, 2, 2, 2])
uct_inicial = np.array([1., 0., 0., 0., 0.])


def rep(v):
    """Replica um vetor (n,) em uma matriz (n, T)."""
    return np.repeat(np.asarray(v, dtype=float)[:, None], T, axis=1)


idx_barra = {b: k for k, b in enumerate(barras)}
A = np.zeros((N, L))
for l, (i, j) in enumerate(linhas):
    A[idx_barra[i], l] = +1
    A[idx_barra[j], l] = -1
Mh = np.zeros((N, H))
for u, b in enumerate(uhe_barra):
    Mh[idx_barra[b], u] = 1
Mt = np.zeros((N, G))
for g, b in enumerate(ute_barra):
    Mt[idx_barra[b], g] = 1
U = np.zeros((H, H))
for u, mont in enumerate(uhe_montante):
    if mont is not None:
        U[u, uhe.index(mont)] = 1
bar_t = [idx_barra[b] for b in ute_barra]

# custos das artificiais: bem acima do custo de qualquer coluna / de qualquer MWh
M_CONV = 10.0 * float(np.max((ute_cvu * ute_pot + ute_custo_partida) * T))
M_EXC = 100.0 * c_deficit


####################### BLOCOS DO MODELO #######################
def bloco_hidro_rede(m, uch_binaria):
    """Hidraulica + transmissao + deficit (tudo, menos a 1a lei)."""
    v = SimpleNamespace()
    v.g_uhe = m.addMVar((H, T), lb=0, ub=rep(uhe_pot), name='g_uhe')
    v.uch = (m.addMVar((H, T), vtype=GRB.BINARY, name='uch') if uch_binaria
             else m.addMVar((H, T), lb=0, ub=1, name='uch'))
    v.turb = m.addMVar((H, T), lb=0, name='turb_uhe')
    v.vert = m.addMVar((H, T), lb=0, name='vert_uhe')
    v.defl = m.addMVar((H, T), lb=rep(defl_min), name='defl_uhe')
    v.arm = m.addMVar((H, T), lb=0, ub=rep(uhe_arm_cap), name='uhe_arm')
    v.f = m.addMVar((L, T), lb=-rep(capacidade_linhas), ub=rep(capacidade_linhas), name='fluxo')
    v.theta = m.addMVar((N, T), lb=-GRB.INFINITY, name='theta')
    v.deficit = m.addMVar((N, T), lb=0, name='deficit')

    m.addConstr(v.f == B @ A.T @ v.theta, name='KVL')
    m.addConstr(v.theta[barra_ref, :] == np.zeros(T), name='barra_referencia')
    m.addConstr(v.turb + v.vert == v.defl, name='balanco_hidrico')
    m.addConstr(v.g_uhe == np.diag(prod_uhe) @ v.turb, name='producao_hidro')
    m.addConstr(v.arm[:, 0] == uhe_arm_cap / 2 + vaz_incr[:, 0] - v.turb[:, 0] - v.vert[:, 0]
                + U @ v.defl[:, 0], name='balanco_armazenamento_inicial')
    if T > 1:
        m.addConstr(v.arm[:, 1:] == v.arm[:, :-1] + vaz_incr[:, 1:] - v.turb[:, 1:]
                    - v.vert[:, 1:] + U @ v.defl[:, 1:], name='balanco_armazenamento')
        m.addConstr(v.defl[:, 1:] - v.defl[:, :-1] <= rep(0.1 * uhe_pot)[:, 1:], name='rampa_up_hidro')
        m.addConstr(v.defl[:, :-1] - v.defl[:, 1:] <= rep(0.1 * uhe_pot)[:, 1:], name='rampa_down_hidro')
    m.addConstr(v.arm[:, T - 1] >= uhe_arm_cap / 2, name='armazenamento_final')
    m.addConstr(v.g_uhe <= rep(uhe_pot) * v.uch, name='capacidade_hidro')
    m.addConstr(v.g_uhe >= rep(uhe_g_min) * v.uch, name='geracao_minima_hidro')
    return v


def bloco_termico(m, g, p=None):
    """X^g: despacho, comprometimento, partida/parada, tempos minimos e rampa da UTE g."""
    if p is None:
        p = m.addMVar(T, lb=0, name=f'p_{g}')
    u = m.addMVar(T, vtype=GRB.BINARY, name=f'u_{g}')
    y = m.addMVar(T, vtype=GRB.BINARY, name=f'y_{g}')
    z = m.addMVar(T, vtype=GRB.BINARY, name=f'z_{g}')
    m.addConstr(p <= ute_pot[g] * u, name=f'capacidade_ute_{g}')
    m.addConstr(p >= ute_g_min[g] * u, name=f'geracao_minima_ute_{g}')
    m.addConstr(u[0] - uct_inicial[g] == y[0] - z[0], name=f'logica_uct_inicial_{g}')
    if T > 1:
        m.addConstr(u[1:] - u[:-1] == y[1:] - z[1:], name=f'logica_uct_{g}')
        m.addConstr(p[1:] - p[:-1] <= ute_rampa[g], name=f'rampa_up_ute_{g}')
        m.addConstr(p[:-1] - p[1:] <= ute_rampa[g], name=f'rampa_down_ute_{g}')
    m.addConstr(y + z <= 1, name=f'partida_ou_desligamento_{g}')
    for t in range(t_on[g] - 1, T):
        m.addConstr(y[t - t_on[g] + 1:t + 1].sum() <= u[t], name=f'tempo_min_ligado_{g}_{t}')
    for t in range(t_off[g] - 1, T):
        m.addConstr(z[t - t_off[g] + 1:t + 1].sum() <= 1 - u[t], name=f'tempo_min_desligado_{g}_{t}')
    return SimpleNamespace(p=p, u=u, y=y, z=z)


def modelo_compacto():
    """O modelo original completo (o mesmo de branch&bound.py), para referencia."""
    m = gp.Model('compacto')
    m.Params.OutputFlag = 0
    h = bloco_hidro_rede(m, uch_binaria=True)
    g_ute = m.addMVar((G, T), lb=0, name='g_ute')
    th = [bloco_termico(m, g, g_ute[g]) for g in range(G)]
    m.addConstr(Mh @ h.g_uhe + Mt @ g_ute + h.deficit - A @ h.f == demanda, name='KCL')
    m.setObjective(gp.quicksum(ute_cvu[g] * th[g].p.sum() + ute_custo_partida[g] * th[g].y.sum()
                               for g in range(G)) + c_deficit * h.deficit.sum(), GRB.MINIMIZE)
    m.update()                   # sem isso, m.relax() copiaria um modelo vazio
    return m


def referencias(tempo=TEMPO_REF):
    """z_LP (relaxacao linear) e o MIP compacto (UB e bound) com limite de tempo."""
    m = modelo_compacto()
    lp = m.relax()
    lp.optimize()
    m.Params.TimeLimit = tempo
    m.optimize()
    return dict(z_lp=lp.ObjVal,
                ub=m.ObjVal if m.SolCount else np.nan,
                bound=m.ObjBound, tempo=m.Runtime)


####################### DECOMPOSICAO #######################
class Decomposicao:
    """Mestre restrito + um pricing por UTE. Suporta restricoes de no (branch-and-price)."""

    def __init__(self, n_pool=COLUNAS_POR_PRICING):
        M = self.M = gp.Model('mestre')
        M.Params.OutputFlag = 0
        self.h = bloco_hidro_rede(M, uch_binaria=False)
        self.exc = M.addMVar((N, T), lb=0, name='excesso')
        self.art = M.addMVar(G, lb=0, name='art_conv')
        self.kcl = M.addConstr(Mh @ self.h.g_uhe + self.h.deficit - self.exc - A @ self.h.f
                               == demanda, name='KCL')
        self.conv = M.addConstr(self.art == np.ones(G), name='convexidade')
        self.kcl_l, self.conv_l = self.kcl.tolist(), self.conv.tolist()
        # o objetivo e fixado UMA vez: setObjective depois zeraria o custo das colunas
        M.setObjective(c_deficit * self.h.deficit.sum() + M_EXC * self.exc.sum()
                       + M_CONV * self.art.sum(), GRB.MINIMIZE)

        self.sub = []
        for g in range(G):
            s = gp.Model(f'pricing_{g}')
            s.Params.OutputFlag = 0
            s.Params.MIPGap = 0
            s.Params.PoolSolutions = n_pool
            s.Params.TimeLimit = TEMPO_PRICING
            self.sub.append((s, bloco_termico(s, g)))
        self.n_pool = n_pool
        self.cols = [[] for _ in range(G)]
        self.chaves = [set() for _ in range(G)]
        self.fix_uct = {}

        # colunas iniciais: o pricing com duais zero da a programacao mais barata (desligada)
        for g in range(G):
            sols, _ = self.precificar(g, np.zeros((N, T)), 0.0)
            self.adicionar_coluna(g, *sols[0][1:])

    # ---------------- colunas ----------------
    def adicionar_coluna(self, g, p, u, y, z):
        p = np.where(np.abs(p) < 1e-9, 0.0, p)
        u, y, z = np.rint(u), np.rint(y), np.rint(z)
        chave = (tuple(np.round(p, 6)), tuple(u.astype(int)))
        if chave in self.chaves[g]:
            return False
        self.chaves[g].add(chave)
        b = bar_t[g]
        coef = [float(p[t]) for t in range(T) if p[t] != 0.0] + [1.0]
        cons = [self.kcl_l[b][t] for t in range(T) if p[t] != 0.0] + [self.conv_l[g]]
        custo = float(ute_cvu[g] * p.sum() + ute_custo_partida[g] * y.sum())
        compativel = all(u[t] == v for (gg, t), v in self.fix_uct.items() if gg == g)
        var = self.M.addVar(lb=0, ub=GRB.INFINITY if compativel else 0.0, obj=custo,
                            column=gp.Column(coef, cons), name=f'lam_{g}_{len(self.cols[g])}')
        self.cols[g].append(SimpleNamespace(p=p, u=u, y=y, z=z, custo=custo, var=var))
        return True

    def n_colunas(self):
        return sum(len(c) for c in self.cols)

    # ---------------- pricing ----------------
    def precificar(self, g, pi, mu):
        """Resolve o pricing da UTE g. Devolve ([(rc, p, u, y, z), ...], limite de rc)
        ou None se X^g (com as fixacoes do no) for vazio."""
        s, v = self.sub[g]
        s.setObjective((ute_cvu[g] - pi[bar_t[g]]) @ v.p + ute_custo_partida[g] * v.y.sum(),
                       GRB.MINIMIZE)
        s.optimize()
        if s.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            return None
        sols = []
        for i in range(min(s.SolCount, self.n_pool)):
            s.Params.SolutionNumber = i
            sols.append((s.PoolObjVal - mu, v.p.Xn, v.u.Xn, v.y.Xn, v.z.Xn))
        return sols, s.ObjBound - mu

    # ---------------- restricoes de no (branch-and-price) ----------------
    def aplicar_no(self, fix_uct, fix_uch):
        """fix_uct {(g, t): 0/1} vai para o pricing e desliga colunas incompativeis;
        fix_uch {(u, t): 0/1} vira limite de variavel do mestre."""
        self.fix_uct = dict(fix_uct)
        lb, ub = np.zeros((H, T)), np.ones((H, T))
        for (u, t), v in fix_uch.items():
            lb[u, t] = ub[u, t] = v
        self.h.uch.LB, self.h.uch.UB = lb, ub
        for g, (s, v) in enumerate(self.sub):
            lo, hi = np.zeros(T), np.ones(T)
            for (gg, t), val in self.fix_uct.items():
                if gg == g:
                    lo[t] = hi[t] = val
            v.u.LB, v.u.UB = lo, hi
            for c in self.cols[g]:
                ok = all(c.u[t] == val for (gg, t), val in self.fix_uct.items() if gg == g)
                c.var.UB = GRB.INFINITY if ok else 0.0

    # ---------------- geracao de colunas ----------------
    def gerar_colunas(self, ub=np.inf, max_iter=MAX_ITER_CG, log=False, t0=None):
        """Gera colunas ate o RMP ficar otimo. Para antes se o limite de Lagrange
        passar de ub (no podado). Devolve dict(status, z, lb, it, hist)."""
        t0 = time.time() if t0 is None else t0
        lb, hist = -np.inf, []
        if log:
            print(f"{'it':>4} {'z_RMP':>16} {'LB Lagrange':>16} {'gap':>9} {'novas':>6} "
                  f"{'colunas':>8} {'t(s)':>7}")
        for it in range(max_iter):
            self.M.optimize()
            if self.M.Status != GRB.OPTIMAL:
                raise RuntimeError(f'RMP com status {self.M.Status}')
            z = self.M.ObjVal
            pi, mu = self.kcl.Pi, self.conv.Pi
            lb_it, novas = z, 0
            for g in range(G):
                res = self.precificar(g, pi, mu[g])
                if res is None:
                    return dict(status='inviavel', z=np.inf, lb=np.inf, it=it, hist=hist)
                sols, rc_min = res
                lb_it += min(0.0, rc_min)
                for rc, *col in sols:
                    if rc < -TOL_RC:
                        novas += self.adicionar_coluna(g, *col)
            if novas == 0:
                lb_it = z                       # nenhuma coluna melhora: RMP = mestre
            lb = max(lb, lb_it)
            gap = (z - lb) / max(abs(z), 1.0)
            hist.append((it, z, lb, novas, self.n_colunas(), time.time() - t0))
            if log:
                print(f"{it:>4} {z:>16,.2f} {lb:>16,.2f} {gap:>9.2e} {novas:>6} "
                      f"{self.n_colunas():>8} {time.time() - t0:>7.1f}")
            if lb >= ub - 1e-6 * max(abs(ub), 1.0):
                return dict(status='podado', z=z, lb=lb, it=it + 1, hist=hist)
            if novas == 0 or gap <= GAP_CG:
                return dict(status='otimo', z=z, lb=lb, it=it + 1, hist=hist)
        return dict(status='limite_iter', z=z, lb=lb, it=max_iter, hist=hist)

    # ---------------- solucoes ----------------
    def artificiais(self):
        return float(self.art.X.sum() + self.exc.X.sum())

    def solucao(self):
        """Solucao do RMP em termos das variaveis originais (UTEs agregadas pelas colunas)."""
        sol = dict(g_ute=np.zeros((G, T)), uct=np.zeros((G, T)), part_ute=np.zeros((G, T)),
                   desl_ute=np.zeros((G, T)))
        for g in range(G):
            for c in self.cols[g]:
                lam = c.var.X
                if lam > 1e-12:
                    sol['g_ute'][g] += lam * c.p
                    sol['uct'][g] += lam * c.u
                    sol['part_ute'][g] += lam * c.y
                    sol['desl_ute'][g] += lam * c.z
        for k in ('g_uhe', 'uch', 'turb', 'vert', 'defl', 'arm', 'f', 'theta', 'deficit'):
            sol[k] = getattr(self.h, k).X.copy()
        sol['custo_total'] = self.M.ObjVal
        return sol

    def completar_hidro(self, uct, tempo=TEMPO_HEUR):
        """Com o comprometimento termico INTEIRO uct (G, T): resolve o mestre com uch
        binaria e so as colunas desse comprometimento (lam continua). Como todas as
        colunas ativas de g tem o mesmo u, p agregado e viavel: a solucao e viavel
        para o problema original. Respeita os limites de uch do no."""
        lam = [c.var for cs in self.cols for c in cs]
        ub_lam = [v.UB for v in lam]
        for g in range(G):
            for c in self.cols[g]:
                if not np.array_equal(c.u, uct[g]):
                    c.var.UB = 0.0
        self.h.uch.VType = GRB.BINARY
        self.art.UB, self.exc.UB = np.zeros(G), np.zeros((N, T))
        self.M.Params.TimeLimit = tempo
        self.M.optimize()
        sol = self.solucao() if self.M.SolCount else None
        self.M.Params.TimeLimit = GRB.INFINITY
        self.h.uch.VType = GRB.CONTINUOUS
        self.M.setAttr('UB', lam, ub_lam)
        self.art.UB, self.exc.UB = np.full(G, GRB.INFINITY), np.full((N, T), GRB.INFINITY)
        return sol

    def heuristica_inteira(self, tempo=TEMPO_HEUR):
        """Mestre restrito INTEIRO com todas as colunas (lam e uch binarias, artificiais
        proibidas): toda solucao dele e viavel para o problema original."""
        lam = [c.var for cs in self.cols for c in cs]
        ub_lam = [v.UB for v in lam]
        lb_uch, ub_uch = self.h.uch.LB.copy(), self.h.uch.UB.copy()
        self.M.setAttr('UB', lam, [GRB.INFINITY] * len(lam))
        self.M.setAttr('VType', lam, [GRB.BINARY] * len(lam))
        self.h.uch.LB, self.h.uch.UB = np.zeros((H, T)), np.ones((H, T))
        self.h.uch.VType = GRB.BINARY
        self.art.UB, self.exc.UB = np.zeros(G), np.zeros((N, T))
        self.M.Params.TimeLimit = tempo
        self.M.optimize()
        sol = self.solucao() if self.M.SolCount else None
        if sol is not None:
            sol['custo_total'] = self.M.ObjVal
        # volta ao RMP linear com as restricoes do no
        self.M.Params.TimeLimit = GRB.INFINITY
        self.M.setAttr('VType', lam, [GRB.CONTINUOUS] * len(lam))
        self.M.setAttr('UB', lam, ub_lam)
        self.h.uch.VType = GRB.CONTINUOUS
        self.h.uch.LB, self.h.uch.UB = lb_uch, ub_uch
        self.art.UB, self.exc.UB = np.full(G, GRB.INFINITY), np.full((N, T), GRB.INFINITY)
        return sol


####################### EXECUCAO #######################
def main():
    print(f"Sistema: N={N} barras, L={L} linhas, H={H} UHE, G={G} UTE, T={T} etapas")
    print(f"Mestre: hidraulica + rede + deficit | {G} pricings (um por UTE)\n")

    ref = referencias()
    print(f"[ref] relaxacao linear  z_LP = {ref['z_lp']:,.2f}")
    print(f"[ref] MIP compacto      z*  <= {ref['ub']:,.2f}  (bound {ref['bound']:,.2f}, "
          f"{ref['tempo']:.1f} s)\n")

    t0 = time.time()
    dec = Decomposicao()
    res = dec.gerar_colunas(log=True, t0=t0)
    z_dw = res['lb'] if res['status'] != 'otimo' else res['z']
    t_cg = time.time() - t0
    print(f"\ngeracao de colunas: {res['status']} em {res['it']} iteracoes, {t_cg:.1f} s, "
          f"{dec.n_colunas()} colunas | artificiais = {dec.artificiais():.2e}")

    sol_lp = dec.solucao()
    frac_uct = int(np.sum(np.minimum(sol_lp['uct'], 1 - sol_lp['uct']) > 1e-6))
    frac_uch = int(np.sum(np.minimum(sol_lp['uch'], 1 - sol_lp['uch']) > 1e-6))
    print(f"solucao do mestre: {frac_uct} uct agregadas fracionarias, {frac_uch} uch fracionarias")

    heur = dec.heuristica_inteira()
    ub = heur['custo_total'] if heur else np.nan

    print("\n" + "=" * 62)
    print("RESUMO")
    print("=" * 62)
    print(f"  z_LP  (relaxacao linear)        = {ref['z_lp']:>16,.2f}")
    print(f"  z_DW  (Dantzig-Wolfe)           = {z_dw:>16,.2f}")
    print(f"  UB    (mestre restrito inteiro) = {ub:>16,.2f}")
    print(f"  z*    (MIP compacto)           <= {ref['ub']:>16,.2f}  (bound {ref['bound']:,.2f})")
    print(f"  ganho z_DW - z_LP               = {z_dw - ref['z_lp']:>16,.2f}")
    melhor_ub = np.nanmin([ub, ref['ub']])
    print(f"  gap DW   (UB - z_DW)/UB         = {100 * (melhor_ub - z_dw) / melhor_ub:>15.4f}%")
    print(f"  gap LP   (UB - z_LP)/UB         = {100 * (melhor_ub - ref['z_lp']) / melhor_ub:>15.4f}%")

    h = np.array(res['hist'])
    np.savez('dantzig_wolfe.npz', iteracao=h[:, 0], z_rmp=h[:, 1], lb_lagrange=h[:, 2],
             colunas_novas=h[:, 3], colunas=h[:, 4], tempo=h[:, 5],
             z_lp=ref['z_lp'], z_dw=z_dw, ub_heur=ub, ub_mip=ref['ub'], bound_mip=ref['bound'],
             uct_lp=sol_lp['uct'], g_ute_lp=sol_lp['g_ute'], uch_lp=sol_lp['uch'],
             **({f'{k}_heur': v for k, v in heur.items()} if heur else {}))
    print("\ndantzig_wolfe.npz gerado.")


if __name__ == '__main__':
    main()
