import gurobipy as gp
import pandas as pd
import numpy as np

####################### MODELO #######################
FLAG_PRINT = False
m = gp.Model("Despacho Hidrotérmico - IEEE 14 barras")
m.Params.Threads = 1
m.Params.Presolve = 0
m.Params.Heuristics = 0
m.Params.Cuts = 0
m.Params.MIPGap = 1e-3
# m.Params.GomoryPasses = 0
# m.Params.MIRCuts = 1
# m.Params.FlowCoverCuts = 0
# m.Params.FlowPathCuts = 0
# m.Params.RLTCuts = 0
# m.Params.RelaxLiftCuts = 0
# m.Params.ImpliedCuts = 0
# m.Params.InfProofCuts = 0
### Conjuntos ###
barras = [f'B{i}' for i in range(1, 15)]

# Ramos do IEEE 14 barras (20 linhas). Arcos direcionados: (de, para)
linhas = [('B1', 'B2'), ('B1', 'B5'), ('B2', 'B3'), ('B2', 'B4'), ('B2', 'B5'),
          ('B3', 'B4'), ('B4', 'B5'), ('B4', 'B7'), ('B4', 'B9'), ('B5', 'B6'),
          ('B6', 'B11'), ('B6', 'B12'), ('B6', 'B13'), ('B7', 'B8'), ('B7', 'B9'),
          ('B9', 'B10'), ('B9', 'B14'), ('B10', 'B11'), ('B12', 'B13'), ('B13', 'B14')]

uhe = ['UHE1', 'UHE2', 'UHE3', 'UHE4']
ute = ['UTE1', 'UTE2', 'UTE3', 'UTE4', 'UTE5']


def ler_matriz(sheet, indices):
    """Lê a planilha e devolve a matriz (len(indices), T), independente da orientação."""
    df = pd.read_excel('dados.xlsx', sheet_name=sheet)
    df.columns = [str(c) for c in df.columns]
    if set(indices).issubset(df.columns):
        return df[indices].to_numpy(dtype=float).T
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    return df.loc[indices].to_numpy(dtype=float)


vaz_incr = ler_matriz('vazao', uhe)  # Vazão incremental de cada UHE em cada etapa (H x T)
demanda = ler_matriz('demanda', barras)  # Demanda em cada barra em cada etapa (N x T)
if vaz_incr.shape[1] != demanda.shape[1]:
    raise ValueError('vazao e demanda com horizontes diferentes em dados.xlsx')

# Horizonte definido pelos dados (ajuste SEMANAS em sistema.py e rode gerar_dados.py)
etapas = range(demanda.shape[1])

N, L, H, G, T = len(barras), len(linhas), len(uhe), len(ute), len(etapas)

### Parametros ###
S_BASE = 100.  # Potência base (MVA) para converter reatância pu em MW/rad

# Reatâncias série do IEEE 14 barras (pu na base de 100 MVA), na ordem de 'linhas'
x_linhas = np.array([0.05917, 0.22304, 0.19797, 0.17632, 0.17388,
                     0.17103, 0.04211, 0.20912, 0.55618, 0.25202,
                     0.19890, 0.25581, 0.13027, 0.17615, 0.11001,
                     0.08450, 0.27038, 0.19207, 0.19988, 0.34802])

# [ALTERADO] Susceptância real das linhas em MW/rad (era identidade no código original).
# Com B = S_BASE / x, o fluxo f = B A' theta já sai em MW e a 2a lei de Kirchhoff
# distribui os fluxos pelas malhas conforme as reatâncias — que é o ponto de usar o IEEE 14.
B = np.diag(S_BASE / x_linhas)

capacidade_linhas = np.array([190., 100., 120., 100., 90.,
                              80., 130., 90., 45., 110.,
                              55., 40., 48., 55., 100.,
                              55., 38., 50., 30., 32.])

c_deficit = 3000.  # Custo de déficit de energia
barra_ref = 0  # Índice da barra de referência (ângulo fixado em zero)

# --- Hidrelétricas: cascata UHE1 -> UHE2 -> UHE3 -> UHE4 ---
uhe_pot = np.array([70., 95., 110., 120.])  # Potência máxima de cada UHE
uhe_g_min = 0.2 * uhe_pot  # Geração mínima de cada UHE (20% da potência máxima)
uhe_arm_cap = np.array([900., 460., 300., 170.])  # Armazenamento máximo de cada UHE
uhe_barra = ['B1', 'B2', 'B6', 'B7']  # Barra onde cada UHE está localizada
prod_uhe = np.array([1., 1., 1., 1.])  # Produtibilidade de cada UHE (queda d'água constante)
defl_min = np.array([6., 10., 14., 18.])  # Defluência mínima de cada UHE
uhe_montante = [None, 'UHE1', 'UHE2', 'UHE3']  # UHE montante de cada UHE (None se não houver)

# --- Termelétricas: da base barata ao peaker caro ---
ute_pot = np.array([110., 85., 70., 60., 45.])  # Potência máxima de cada UTE
ute_cvu = np.array([42., 85., 130., 210., 450.])  # Custo variável unitário de cada UTE
ute_g_min = np.array([45., 30., 22., 15., 5.])  # Geração mínima de cada UTE
ute_rampa = np.array([22., 25., 30., 40., 45.])  # Rampa de subida/descida de cada UTE
ute_barra = ['B3', 'B6', 'B8', 'B2', 'B13']  # Barra onde cada UTE está localizada
ute_custo_partida = np.array([9000., 4500., 2600., 1300., 450.])  # Custo de cada partida
t_on = np.array([6, 4, 3, 2, 2])  # Tempo mínimo de operação de cada UTE
t_off = np.array([5, 3, 2, 2, 2])  # Tempo mínimo de desligamento de cada UTE
uct_inicial = np.array([1., 0., 0., 0., 0.])  # Estado de operação na etapa anterior ao horizonte


def rep(v):
    """Replica um vetor (n,) em uma matriz (n, T)."""
    return np.repeat(np.asarray(v, dtype=float)[:, None], T, axis=1)


### Matrizes de incidência ###
idx_barra = {b: k for k, b in enumerate(barras)}

A = np.zeros((N, L))  # Barra x linha: +1 na origem, -1 no destino
for l, (i, j) in enumerate(linhas):
    A[idx_barra[i], l] = +1
    A[idx_barra[j], l] = -1

Mh = np.zeros((N, H))  # Barra x UHE
for u, b in enumerate(uhe_barra):
    Mh[idx_barra[b], u] = 1

Mt = np.zeros((N, G))  # Barra x UTE
for g, b in enumerate(ute_barra):
    Mt[idx_barra[b], g] = 1

U = np.zeros((H, H))  # Cascata: U[u, v] = 1 se v é montante de u
for u, mont in enumerate(uhe_montante):
    if mont is not None:
        U[u, uhe.index(mont)] = 1

### Variáveis de decisão ###
# Hidro #
g_uhe = m.addMVar((H, T), name="g_uhe", lb=0, ub=rep(uhe_pot))  # Geração de cada UHE em cada etapa
uch = m.addMVar((H, T), vtype=gp.GRB.BINARY, name="uch")  # Estado de operação de cada UHE em cada etapa
turb_uhe = m.addMVar((H, T), name="turb_uhe", lb=0)  # Turbinamento de cada UHE em cada etapa
vert_uhe = m.addMVar((H, T), name="vert_uhe", lb=0)  # Vertimento de cada UHE em cada etapa
defl_uhe = m.addMVar((H, T), name="defl_uhe", lb=rep(defl_min))  # Defluência de cada UHE em cada etapa
uhe_arm = m.addMVar((H, T), name="uhe_arm", lb=0, ub=rep(uhe_arm_cap))  # Armazenamento de cada UHE em cada etapa

# Term #
g_ute = m.addMVar((G, T), name="g_ute", lb=0)  # Geração de cada UTE em cada etapa
uct = m.addMVar((G, T), vtype=gp.GRB.BINARY, name="uct")  # Estado de operação de cada UTE em cada etapa
part_ute = m.addMVar((G, T), vtype=gp.GRB.BINARY, name="part_ute")  # Partida de cada UTE em cada etapa
desl_ute = m.addMVar((G, T), vtype=gp.GRB.BINARY, name="desl_ute")  # Desligamento de cada UTE em cada etapa

# Transmissao #
f = m.addMVar((L, T), name="fluxo", lb=-rep(capacidade_linhas), ub=rep(capacidade_linhas))  # Fluxo de cada linha em cada etapa
theta = m.addMVar((N, T), name="theta", lb=-gp.GRB.INFINITY)  # Ângulo de fase em cada barra em cada etapa
deficit = m.addMVar((N, T), name="deficit", lb=0)  # Déficit de energia em cada barra em cada etapa

### Restrições ###
# Atendimento da demanda (1a lei de Kirchhoff) #
kcl = m.addConstr(Mh @ g_uhe + Mt @ g_ute + deficit - A @ f == demanda, name='KCL')

# 2a lei de Kirchhoff #
m.addConstr(f == B @ A.T @ theta, name='KVL')
m.addConstr(theta[barra_ref, :] == np.zeros(T), name='barra_referencia')


# Balanco Hídrico #
m.addConstr(turb_uhe + vert_uhe == defl_uhe, name='balanco_hidrico')
m.addConstr(g_uhe == np.diag(prod_uhe) @ turb_uhe, name='producao_hidro')

m.addConstr(uhe_arm[:, 0] == uhe_arm_cap / 2 + vaz_incr[:, 0] - turb_uhe[:, 0] - vert_uhe[:, 0]
            + U @ defl_uhe[:, 0], name='balanco_armazenamento_inicial')
m.addConstr(uhe_arm[:, 1:] == uhe_arm[:, :-1] + vaz_incr[:, 1:] - turb_uhe[:, 1:] - vert_uhe[:, 1:]
            + U @ defl_uhe[:, 1:], name='balanco_armazenamento')
m.addConstr(uhe_arm[:, T - 1] >= uhe_arm_cap / 2, name='armazenamento_final')

# Comprometimento das UHEs: desligada gera zero e toda a defluência vira vertimento #
m.addConstr(g_uhe <= rep(uhe_pot) * uch, name='capacidade_hidro')
m.addConstr(g_uhe >= rep(uhe_g_min) * uch, name='geracao_minima_hidro')

# Rampa hidráulica: limita a variação da defluência (restrição de deplecionamento) #
m.addConstr(defl_uhe[:, 1:] - defl_uhe[:, :-1] <= rep(0.1 * uhe_pot)[:, 1:], name='rampa_up_hidro')
m.addConstr(defl_uhe[:, :-1] - defl_uhe[:, 1:] <= rep(0.1 * uhe_pot)[:, 1:], name='rampa_down_hidro')


# Operação das UTEs #
m.addConstr(g_ute <= rep(ute_pot) * uct, name='capacidade_ute')
m.addConstr(g_ute >= rep(ute_g_min) * uct, name='geracao_minima_ute')
m.addConstr(g_ute[:, 1:] - g_ute[:, :-1] <= rep(ute_rampa)[:, 1:], name='rampa_up_ute')
m.addConstr(g_ute[:, 1:] - g_ute[:, :-1] >= -rep(ute_rampa)[:, 1:], name='rampa_down_ute')

# Tempo mínimo ligado/desligado (janelas contíguas) #
m.addConstr(uct[:, 0] - uct_inicial == part_ute[:, 0] - desl_ute[:, 0], name='logica_uct_inicial')
m.addConstr(uct[:, 1:] - uct[:, :-1] == part_ute[:, 1:] - desl_ute[:, 1:], name='logica_uct')
m.addConstr(part_ute + desl_ute <= np.ones((G, T)), name='partida_ou_desligamento')
for g in range(G):
    for t in range(t_on[g] - 1, T):
        m.addConstr(part_ute[g, t - t_on[g] + 1:t + 1].sum() <= uct[g, t], name=f'tempo_min_ligado_{g}_{t}')
    for t in range(t_off[g] - 1, T):
        m.addConstr(desl_ute[g, t - t_off[g] + 1:t + 1].sum() <= 1 - uct[g, t], name=f'tempo_min_desligado_{g}_{t}')

### FO ###
m.setObjective((ute_cvu @ g_ute).sum()
               + (ute_custo_partida @ part_ute).sum()
               + c_deficit * deficit.sum(), sense=gp.GRB.MINIMIZE)

m.optimize()

print("Status:", m.status)
if m.status == gp.GRB.OPTIMAL:
    if FLAG_PRINT:
        print("Objective Value:", m.objVal)
        print("Geração UHE:")
        print(pd.DataFrame(g_uhe.X, index=uhe))
        print("Turbinamento UHE:")
        print(pd.DataFrame(turb_uhe.X, index=uhe))
        print("Vertimento UHE:")
        print(pd.DataFrame(vert_uhe.X, index=uhe))
        print("Defluência UHE:")
        print(pd.DataFrame(defl_uhe.X, index=uhe))
        print("Armazenamento UHE:")
        print(pd.DataFrame(uhe_arm.X, index=uhe))
        print("Geração UTE:")
        print(pd.DataFrame(g_ute.X, index=ute))
        print("Déficit de energia:")
        print(pd.DataFrame(deficit.X, index=barras))
        print("Fluxo de linhas:")
        print(pd.DataFrame(f.X, index=[f"{i}-{j}" for i, j in linhas]))

    ### Custo marginal de operação ###
    # O CMO é o dual da 1a lei de Kirchhoff, que só existe em problema contínuo.
    resultados = dict(
        g_uhe=g_uhe.X, uch=uch.X, turb_uhe=turb_uhe.X, vert_uhe=vert_uhe.X,
        defl_uhe=defl_uhe.X,
        uhe_arm=uhe_arm.X, g_ute=g_ute.X, uct=uct.X, part_ute=part_ute.X,
        desl_ute=desl_ute.X, fluxo=f.X, theta=theta.X, deficit=deficit.X,
        demanda=demanda, custo_total=m.objVal)

    binarias = [v for v in m.getVars() if v.VType != gp.GRB.CONTINUOUS]
    valores = m.getAttr('X', binarias)
    for v, val in zip(binarias, valores):
        v.VType = gp.GRB.CONTINUOUS
        v.LB = v.UB = round(val)
    m.Params.OutputFlag = 0
    m.optimize()
    resultados['cmo'] = kcl.Pi if m.status == gp.GRB.OPTIMAL else np.zeros((N, T))

    np.savez('resultados.npz',
             barras=np.array(barras), uhe=np.array(uhe), ute=np.array(ute),
             linhas=np.array([f"{i}-{j}" for i, j in linhas]),
             linha_de=np.array([i for i, _ in linhas]),
             linha_para=np.array([j for _, j in linhas]),
             capacidade_linhas=capacidade_linhas, uhe_barra=np.array(uhe_barra),
             ute_barra=np.array(ute_barra), uhe_arm_cap=uhe_arm_cap,
             uhe_pot=uhe_pot, ute_pot=ute_pot, ute_cvu=ute_cvu, **resultados)
    print("resultados.npz gerado.")
    
    # Saidas em excel
    with pd.ExcelWriter("resultados.xlsx") as writer:
        for chave, valor in resultados.items():
            if np.isscalar(valor):
                pd.DataFrame([valor], columns=[chave]).to_excel(writer, sheet_name=chave)
            else:
                pd.DataFrame(valor).to_excel(writer, sheet_name=chave)
    print("resultados.xlsx gerado.")
         
elif m.status == gp.GRB.INFEASIBLE:
    m.computeIIS()
    m.write("modelo.ilp")
