"""
Relax-and-cut ATRASADO (delayed) do despacho hidrotermico, baseado em lagrange.py.

Mesma relaxacao lagrangeana (rampas e tempo minimo dualizados, subgradiente de
Polyak). A novidade sao RODADAS: em cada rodada o conjunto de multiplicadores e
fixo e o subgradiente roda ate o fim; no fim da rodada separam-se CORTES DE RAMPA
violados pela solucao lagrangeana de melhor L, eles sao dualizados (multiplicador
inicial 0) e a rodada seguinte recomeca com o contador de nao-melhora e o rho
REINICIADOS (os multiplicadores antigos sao mantidos: partida a quente).

Familia de cortes (validos para o problema original; usam a binaria uct):

    cup[g,t]: g_ute[g,t]   - g_ute[g,t-1] - R_g * uct[g,t]   <= 0   (t = 1..T-1)
    cdn[g,t]: g_ute[g,t-1] - g_ute[g,t]   - R_g * uct[g,t-1] <= 0

Se a unidade esta desligada no periodo "de chegada" (cup) / "de saida" (cdn), a
geracao dela e zero e o lado esquerdo e <= 0; se esta ligada, viram a rampa
original. Sao mais fortes que a rampa quando uct e fracionario.

Todo L(lam) com lam >= 0 e QUALQUER subconjunto de cortes dualizados e limite
inferior valido (usa-se o ObjBound), entao melhor_lb vale entre rodadas.

    python lagrange_delayed.py
"""

import gurobipy as gp
import pandas as pd
import numpy as np
from types import SimpleNamespace

###################### PARAMETROS DO SUBGRADIENTE ######################
MAX_ITER = 40           # iteracoes do subgradiente POR RODADA
RHO_0 = 2.0             # rho inicial do passo de Polyak (classico: 2)
RHO_MIN = 1e-3          # abaixo disso o passo e ruido
PACIENCIA = 4           # iteracoes sem melhora do LB antes de halvear rho
MAX_RODADAS = 5         # rodadas de relax-and-cut (1 = subgradiente simples)
RHO_REINICIO = RHO_0    # rho no inicio de cada rodada
TOL_CORTE = 1e-4        # violacao minima para dualizar um corte
MAX_CORTES_RODADA = 200 # maximo de cortes dualizados por rodada
TOL_VIAB = 1e-6         # ||s||^2 abaixo disso => x e viavel => L = z*

GAP_SUB = 1e-3          # MIPGap do subproblema lagrangeano
TEMPO_SUB = 60          # s por subproblema
TEMPO_UB = 120          # s para achar o incumbente (UB) com o modelo completo
TEMPO_LP = 300          # s para a relaxacao linear de referencia

DUALIZAR_RAMPA_HIDRO = True   # False => dualiza so as rampas termicas

CROP_ETAPAS=0

####################### DADOS #######################
barras = [f'B{i}' for i in range(1, 15)]

# Ramos do IEEE 14 barras (20 linhas). Arcos direcionados: (de, para)
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


vaz_incr = ler_matriz('vazao', uhe)      # Vazao incremental de cada UHE (H x T)
demanda = ler_matriz('demanda', barras)  # Demanda em cada barra (N x T)
if vaz_incr.shape[1] != demanda.shape[1]:
    raise ValueError('vazao e demanda com horizontes diferentes em dados.xlsx')

if CROP_ETAPAS>0:
    vaz_incr = vaz_incr[:,:CROP_ETAPAS]
    demanda = demanda[:,:CROP_ETAPAS]

etapas = range(demanda.shape[1])
N, L, H, G, T = len(barras), len(linhas), len(uhe), len(ute), len(etapas)

### Parametros ###
S_BASE = 100.
x_linhas = np.array([0.05917, 0.22304, 0.19797, 0.17632, 0.17388,
                     0.17103, 0.04211, 0.20912, 0.55618, 0.25202,
                     0.19890, 0.25581, 0.13027, 0.17615, 0.11001,
                     0.08450, 0.27038, 0.19207, 0.19988, 0.34802])
B = np.diag(S_BASE / x_linhas)

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
ute_g_min = np.array([45., 30., 22., 15., 5.])
ute_rampa = np.array([22., 25., 30., 40., 45.])
ute_barra = ['B3', 'B6', 'B8', 'B2', 'B13']
ute_custo_partida = np.array([9000., 4500., 2600., 1300., 450.])
t_on = np.array([6, 4, 3, 2, 2])
t_off = np.array([5, 3, 2, 2, 2])
uct_inicial = np.array([1., 0., 0., 0., 0.])


def rep(v):
    """Replica um vetor (n,) em uma matriz (n, T)."""
    return np.repeat(np.asarray(v, dtype=float)[:, None], T, axis=1)


### Matrizes de incidencia ###
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

################ ESTRUTURA DAS RESTRICOES DUALIZADAS ################
# RHS das rampas, ja no formato (., T-1) usado pelos subgradientes
R_UTE = rep(ute_rampa)[:, 1:]           # (G, T-1)
R_HID = rep(0.1 * uhe_pot)[:, 1:]       # (H, T-1)

# Janelas do tempo minimo: (g, t, inicio) com a restricao sobre [inicio, t]
jan_on = [(g, t, t - t_on[g] + 1) for g in range(G) for t in range(t_on[g] - 1, T)]
jan_off = [(g, t, t - t_off[g] + 1) for g in range(G) for t in range(t_off[g] - 1, T)]

# Cada bloco dualizado e um array; (g,t) identifica unicamente cada restricao,
# entao os multiplicadores de tempo minimo moram em matrizes (G, T) esparsas.
FORMATOS = dict(rup=(G, T - 1), rdn=(G, T - 1),
                hup=(H, T - 1), hdn=(H, T - 1),
                on=(G, T), off=(G, T))
if not DUALIZAR_RAMPA_HIDRO:
    FORMATOS.pop('hup')
    FORMATOS.pop('hdn')

# Cortes de rampa (relax-and-cut): comecam todos inativos (multiplicador 0 e fora
# do subgradiente); a mascara marca os que ja foram dualizados.
CORTES = ('cup', 'cdn')
for k in CORTES:
    FORMATOS[k] = (G, T - 1)
ATIVO = {k: np.zeros((G, T - 1), dtype=bool) for k in CORTES}


####################### MODELO #######################
def construir_modelo(incluir_dualizadas, nome):
    """Monta o modelo. Com incluir_dualizadas=False, omite rampas e tempo minimo
    (que passam a ser termos do lagrangeano)."""
    m = gp.Model(nome)
    m.Params.OutputFlag = 0
    m.Params.Threads = 1

    ### Variaveis ###
    g_uhe = m.addMVar((H, T), name="g_uhe", lb=0, ub=rep(uhe_pot))
    uch = m.addMVar((H, T), vtype=gp.GRB.BINARY, name="uch")
    turb_uhe = m.addMVar((H, T), name="turb_uhe", lb=0)
    vert_uhe = m.addMVar((H, T), name="vert_uhe", lb=0)
    defl_uhe = m.addMVar((H, T), name="defl_uhe", lb=rep(defl_min))
    uhe_arm = m.addMVar((H, T), name="uhe_arm", lb=0, ub=rep(uhe_arm_cap))

    g_ute = m.addMVar((G, T), name="g_ute", lb=0)
    uct = m.addMVar((G, T), vtype=gp.GRB.BINARY, name="uct")
    part_ute = m.addMVar((G, T), vtype=gp.GRB.BINARY, name="part_ute")
    desl_ute = m.addMVar((G, T), vtype=gp.GRB.BINARY, name="desl_ute")

    f = m.addMVar((L, T), name="fluxo", lb=-rep(capacidade_linhas), ub=rep(capacidade_linhas))
    theta = m.addMVar((N, T), name="theta", lb=-gp.GRB.INFINITY)
    deficit = m.addMVar((N, T), name="deficit", lb=0)

    ### Restricoes que ficam em X ###
    kcl = m.addConstr(Mh @ g_uhe + Mt @ g_ute + deficit - A @ f == demanda, name='KCL')
    m.addConstr(f == B @ A.T @ theta, name='KVL')
    m.addConstr(theta[barra_ref, :] == np.zeros(T), name='barra_referencia')

    m.addConstr(turb_uhe + vert_uhe == defl_uhe, name='balanco_hidrico')
    m.addConstr(g_uhe == np.diag(prod_uhe) @ turb_uhe, name='producao_hidro')
    m.addConstr(uhe_arm[:, 0] == uhe_arm_cap / 2 + vaz_incr[:, 0] - turb_uhe[:, 0]
                - vert_uhe[:, 0] + U @ defl_uhe[:, 0], name='balanco_armazenamento_inicial')
    m.addConstr(uhe_arm[:, 1:] == uhe_arm[:, :-1] + vaz_incr[:, 1:] - turb_uhe[:, 1:]
                - vert_uhe[:, 1:] + U @ defl_uhe[:, 1:], name='balanco_armazenamento')
    m.addConstr(uhe_arm[:, T - 1] >= uhe_arm_cap / 2, name='armazenamento_final')

    m.addConstr(g_uhe <= rep(uhe_pot) * uch, name='capacidade_hidro')
    m.addConstr(g_uhe >= rep(uhe_g_min) * uch, name='geracao_minima_hidro')

    m.addConstr(g_ute <= rep(ute_pot) * uct, name='capacidade_ute')
    m.addConstr(g_ute >= rep(ute_g_min) * uct, name='geracao_minima_ute')

    # Logica de estado: FICA em X (nao e restricao de tempo minimo)
    m.addConstr(uct[:, 0] - uct_inicial == part_ute[:, 0] - desl_ute[:, 0], name='logica_uct_inicial')
    m.addConstr(uct[:, 1:] - uct[:, :-1] == part_ute[:, 1:] - desl_ute[:, 1:], name='logica_uct')
    m.addConstr(part_ute + desl_ute <= np.ones((G, T)), name='partida_ou_desligamento')

    ### Restricoes dualizadas (so no modelo completo) ###
    if incluir_dualizadas:
        m.addConstr(g_ute[:, 1:] - g_ute[:, :-1] <= R_UTE, name='rampa_up_ute')
        m.addConstr(g_ute[:, :-1] - g_ute[:, 1:] <= R_UTE, name='rampa_down_ute')
        m.addConstr(defl_uhe[:, 1:] - defl_uhe[:, :-1] <= R_HID, name='rampa_up_hidro')
        m.addConstr(defl_uhe[:, :-1] - defl_uhe[:, 1:] <= R_HID, name='rampa_down_hidro')
        for g in range(G):
            for t in range(t_on[g] - 1, T):
                m.addConstr(part_ute[g, t - t_on[g] + 1:t + 1].sum() <= uct[g, t],
                            name=f'tempo_min_ligado_{g}_{t}')
            for t in range(t_off[g] - 1, T):
                m.addConstr(desl_ute[g, t - t_off[g] + 1:t + 1].sum() <= 1 - uct[g, t],
                            name=f'tempo_min_desligado_{g}_{t}')
    elif not DUALIZAR_RAMPA_HIDRO:
        # rampa hidraulica nao dualizada => continua em X
        m.addConstr(defl_uhe[:, 1:] - defl_uhe[:, :-1] <= R_HID, name='rampa_up_hidro')
        m.addConstr(defl_uhe[:, :-1] - defl_uhe[:, 1:] <= R_HID, name='rampa_down_hidro')

    ### FO original ###
    m.setObjective((ute_cvu @ g_ute).sum()
                   + (ute_custo_partida @ part_ute).sum()
                   + c_deficit * deficit.sum(), sense=gp.GRB.MINIMIZE)
    m.update()

    v = SimpleNamespace(g_uhe=g_uhe, uch=uch, turb_uhe=turb_uhe, vert_uhe=vert_uhe,
                        defl_uhe=defl_uhe, uhe_arm=uhe_arm, g_ute=g_ute, uct=uct,
                        part_ute=part_ute, desl_ute=desl_ute, fluxo=f, theta=theta,
                        deficit=deficit, kcl=kcl)
    return m, v


################ LAGRANGEANO: SUBGRADIENTE E OBJETIVO ################
def viol_cortes(gu, u):
    """Violacao BRUTA (sem mascara) dos cortes de rampa na solucao (gu, u)."""
    return dict(cup=gu[:, 1:] - gu[:, :-1] - R_UTE * u[:, 1:],
                cdn=gu[:, :-1] - gu[:, 1:] - R_UTE * u[:, :-1])


def violacoes(gu, dfl, prt, dsl, u):
    """s = A x - b de cada bloco dualizado, avaliado na solucao do subproblema.

    Positivo = restricao violada (o multiplicador sobe na proxima iteracao).
    """
    s = dict(
        rup=gu[:, 1:] - gu[:, :-1] - R_UTE,
        rdn=gu[:, :-1] - gu[:, 1:] - R_UTE,
        on=np.zeros((G, T)),
        off=np.zeros((G, T)))
    if DUALIZAR_RAMPA_HIDRO:
        s['hup'] = dfl[:, 1:] - dfl[:, :-1] - R_HID
        s['hdn'] = dfl[:, :-1] - dfl[:, 1:] - R_HID
    for k, v in viol_cortes(gu, u).items():
        s[k] = np.where(ATIVO[k], v, 0.0)       # inativo: fora do subgradiente
    for g, t, ini in jan_on:
        s['on'][g, t] = prt[g, ini:t + 1].sum() - u[g, t]
    for g, t, ini in jan_off:
        s['off'][g, t] = dsl[g, ini:t + 1].sum() + u[g, t] - 1.0
    return s


def objetivo_lagrangeano(v, lam):
    """c(x) + lam'(Ax - b) reescrito como custos lineares perturbados + constante."""
    # --- rampas: lam entra como diferenca para frente/para tras ---
    d_ute = lam['rup'] - lam['rdn']                 # (G, T-1)
    c_gu = np.zeros((G, T))
    c_gu[:, 1:] += d_ute
    c_gu[:, :-1] -= d_ute
    const = float((-(lam['rup'] + lam['rdn']) * R_UTE).sum())

    c_df = np.zeros((H, T))
    if DUALIZAR_RAMPA_HIDRO:
        d_hid = lam['hup'] - lam['hdn']
        c_df[:, 1:] += d_hid
        c_df[:, :-1] -= d_hid
        const += float((-(lam['hup'] + lam['hdn']) * R_HID).sum())

    # --- tempo minimo: cada multiplicador espalha pela sua janela ---
    c_prt = np.zeros((G, T))
    c_dsl = np.zeros((G, T))
    c_uct = np.zeros((G, T))
    for g, t, ini in jan_on:
        c_prt[g, ini:t + 1] += lam['on'][g, t]
        c_uct[g, t] -= lam['on'][g, t]
    for g, t, ini in jan_off:
        c_dsl[g, ini:t + 1] += lam['off'][g, t]
        c_uct[g, t] += lam['off'][g, t]
    const -= float(lam['off'].sum())

    # --- cortes de rampa dualizados (rhs = 0, entao nao ha constante) ---
    c_gu[:, 1:] += lam['cup']
    c_gu[:, :-1] -= lam['cup']
    c_uct[:, 1:] -= R_UTE * lam['cup']
    c_gu[:, :-1] += lam['cdn']
    c_gu[:, 1:] -= lam['cdn']
    c_uct[:, :-1] -= R_UTE * lam['cdn']

    return (((rep(ute_cvu) + c_gu) * v.g_ute).sum()
            + ((rep(ute_custo_partida) + c_prt) * v.part_ute).sum()
            + c_deficit * v.deficit.sum()
            + (c_df * v.defl_uhe).sum()
            + (c_dsl * v.desl_ute).sum()
            + (c_uct * v.uct).sum()
            + const)


####################### EXECUCAO #######################
print(f"Sistema: N={N} barras, L={L} linhas, H={H} UHE, G={G} UTE, T={T} etapas")
n_dual = sum(int(np.prod(f)) for k, f in FORMATOS.items() if k not in CORTES)
print(f"Restricoes dualizadas: {n_dual} "
      f"(rampas {'termica+hidro' if DUALIZAR_RAMPA_HIDRO else 'termica'} + tempo minimo)\n")

### 1) Referencias: relaxacao linear e incumbente do MIP completo ###
m_full, v_full = construir_modelo(True, "completo")

m_lp = m_full.relax()
m_lp.Params.OutputFlag = 0
m_lp.Params.TimeLimit = TEMPO_LP
m_lp.optimize()
z_lp = m_lp.ObjVal if m_lp.SolCount else float('nan')
print(f"[ref] relaxacao linear  z_LP  = {z_lp:,.2f}")

m_full.Params.TimeLimit = TEMPO_UB
m_full.optimize()
if m_full.SolCount:
    UB = m_full.ObjVal
    print(f"[ref] incumbente MIP    UB    = {UB:,.2f}  (bound {m_full.ObjBound:,.2f}, "
          f"gap {100 * m_full.MIPGap:.2f}%)\n")
else:
    UB = None
    print("[ref] MIP completo sem incumbente no tempo dado -- "
          "o passo usa alvo estimado\n")

### 2) Subproblema lagrangeano (sem as dualizadas) ###
m_rel, v_rel = construir_modelo(False, "lagrangeano")
m_rel.Params.MIPGap = GAP_SUB
m_rel.Params.TimeLimit = TEMPO_SUB

### 3) Relax-and-cut atrasado: rodadas de subgradiente + separacao ###
lam = {k: np.zeros(f) for k, f in FORMATOS.items()}
melhor_lb = -np.inf
melhor_lam = {k: a.copy() for k, a in lam.items()}
hist = []
resumo_rodadas = []
viavel = False

for rodada in range(MAX_RODADAS):
    lam = {k: a.copy() for k, a in melhor_lam.items()}   # partida a quente
    rho = RHO_REINICIO
    sem_melhora = 0                                      # contador REINICIADO
    lb_rodada, x_rodada = -np.inf, None
    n_ativos = int(sum(ATIVO[k].sum() for k in CORTES))

    print(f"\n=== rodada {rodada}  (cortes de rampa dualizados: {n_ativos}) ===")
    print(f"{'it':>3} {'L(lam)':>14} {'melhor LB':>14} {'||s||':>10} "
          f"{'max viol':>10} {'rho':>7} {'alpha':>10}")
    print("-" * 74)

    for it in range(MAX_ITER):
        m_rel.setObjective(objetivo_lagrangeano(v_rel, lam), gp.GRB.MINIMIZE)
        m_rel.optimize()
        if m_rel.SolCount == 0:
            print(f"{it:>3}  subproblema sem solucao (status {m_rel.status}) -- parando")
            break

        L_val = m_rel.ObjBound
        gu, dfl, prt = v_rel.g_ute.X, v_rel.defl_uhe.X, v_rel.part_ute.X
        dsl, uc = v_rel.desl_ute.X, v_rel.uct.X
        s = violacoes(gu, dfl, prt, dsl, uc)
        norma2 = float(sum((a ** 2).sum() for a in s.values()))
        max_viol = float(max(a.max() for a in s.values()))

        if L_val > lb_rodada:                            # solucao p/ separar os cortes
            lb_rodada, x_rodada = L_val, (gu.copy(), uc.copy())

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
            viavel = True
            break

        alvo = UB if UB is not None else melhor_lb + 0.1 * max(abs(melhor_lb), 1.0)
        alpha = max(rho * (alvo - L_val) / norma2, 0.0)

        print(f"{it:>3} {L_val:>14,.2f} {melhor_lb:>14,.2f} {np.sqrt(norma2):>10.4f} "
              f"{max_viol:>10.4f} {rho:>7.4f} {alpha:>10.3e}")
        hist.append((rodada, it, L_val, melhor_lb, np.sqrt(norma2), max_viol, rho, alpha))

        for k in lam:
            lam[k] = np.maximum(0.0, lam[k] + alpha * s[k])

        if rho <= RHO_MIN:
            print("\nrho no minimo -- fim da rodada.")
            break

    # --- separacao (atrasada): so apos a rodada inteira ---
    novos = 0
    if not viavel and x_rodada is not None and rodada < MAX_RODADAS - 1:
        cand = []
        for k, viol in viol_cortes(*x_rodada).items():
            for g, t in zip(*np.nonzero((viol > TOL_CORTE) & ~ATIVO[k])):
                cand.append((viol[g, t], k, g, t))
        cand.sort(reverse=True)
        for _, k, g, t in cand[:MAX_CORTES_RODADA]:
            ATIVO[k][g, t] = True
            novos += 1
        print(f"\nrodada {rodada}: LB da rodada {lb_rodada:,.2f} | cortes violados "
              f"{len(cand)} | dualizados agora {novos}")
    resumo_rodadas.append((rodada, n_ativos, lb_rodada, melhor_lb))
    if viavel or novos == 0:
        break

### 4) Resumo ###
z_ld = melhor_lb
print("\n" + "=" * 62)
print("RESUMO")
print("=" * 62)
print(f"  {'rodada':>6} {'cortes':>7} {'melhor L da rodada':>20} {'melhor LB':>16}")
for r, nc, lbr, lbg in resumo_rodadas:
    print(f"  {r:>6} {nc:>7} {lbr:>20,.2f} {lbg:>16,.2f}")
print(f"\n  z_LP  (relaxacao linear)      = {z_lp:>16,.2f}")
print(f"  z_LD  (relax-and-cut)         = {z_ld:>16,.2f}")
if UB is not None:
    print(f"  UB    (incumbente MIP)        = {UB:>16,.2f}")
    print(f"  gap lagrangeano  (UB-z_LD)/UB = {100 * (UB - z_ld) / abs(UB):>15.2f}%")
    print(f"  gap linear       (UB-z_LP)/UB = {100 * (UB - z_lp) / abs(UB):>15.2f}%")
if np.isfinite(z_lp) and np.isfinite(z_ld):
    print(f"  ganho sobre LP                = {z_ld - z_lp:>16,.2f}")

if hist:
    arr = np.array(hist)
    np.savez('lagrange_delayed.npz',
             rodada=arr[:, 0], iteracao=arr[:, 1], L=arr[:, 2], melhor_lb=arr[:, 3],
             norma_subgradiente=arr[:, 4], max_violacao=arr[:, 5], rho=arr[:, 6],
             alpha=arr[:, 7], z_lp=z_lp, z_ld=z_ld, ub=UB if UB is not None else np.nan,
             **{f'lam_{k}': a for k, a in melhor_lam.items()},
             **{f'ativo_{k}': a for k, a in ATIVO.items()})
    print("\nlagrange_delayed.npz gerado.")
