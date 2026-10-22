import gurobipy as gp
import pandas as pd
import numpy as np
import scipy.sparse as sp
from functools import partial

###################### MIR AGREGADO ######################
TOL = 1e-6
ESCALAS = (1.0, 2.0, 0.5, 4.0, 0.25, 8.0, 0.125)
DEBUG_AGG = True  # loga cada entrada em MIPNODE e cada corte adicionado


def dados_matriciais(model):
    """Linhas do modelo normalizadas para '<=', mais bounds e tipos das variaveis."""
    model.update()
    A = model.getA().tocsr()
    restricoes = model.getConstrs()
    rhs = np.array(model.getAttr('RHS', restricoes), dtype=float)
    sentido = np.array(model.getAttr('Sense', restricoes))
    variaveis = model.getVars()
    # '>' vira '<=' multiplicando por -1; '=' serve nos dois sentidos
    sinal = np.where(sentido == '>', -1.0, 1.0)
    return dict(
        A=(sp.diags(sinal) @ A).tocsr(),
        rhs=sinal * rhs,
        igualdade=(sentido == '='),
        lb=np.array(model.getAttr('LB', variaveis), dtype=float),
        ub=np.array(model.getAttr('UB', variaveis), dtype=float),
        inteira=np.array([v.VType in (gp.GRB.BINARY, gp.GRB.INTEGER) for v in variaveis]))


def funcao_mir(a, b, inteira):
    """Aplica a funcao MIR a uma linha ja transladada (todas as variaveis >= 0)."""
    f = b - np.floor(b)
    if f < TOL or f > 1.0 - TOL:
        return None
    fj = a - np.floor(a)
    coef = np.where(inteira,
                    np.floor(a) + np.maximum(fj - f, 0.0) / (1.0 - f),
                    np.minimum(a, 0.0) / (1.0 - f))
    return coef, np.floor(b)


def agregar(dados, linha0, xr, max_agg):
    """Combina linhas para eliminar continuas fora dos bounds (proxy de variavel basica)."""
    A, rhs, lb, ub, inteira = dados['A'], dados['rhs'], dados['lb'], dados['ub'], dados['inteira']
    a = np.asarray(A[linha0].todense()).ravel()
    b = rhs[linha0]
    usadas = {linha0}
    for _ in range(max_agg):
        # continua, com coeficiente nao nulo e estritamente entre os bounds
        cand = np.nonzero((np.abs(a) > TOL) & ~inteira
                          & (np.minimum(xr - lb, ub - xr) > 1e-4))[0]
        if cand.size == 0:
            break
        j = int(cand[np.argmax(np.abs(a[cand]))])
        coluna = A[:, j].tocoo()
        outras = [(int(i), v) for i, v in zip(coluna.row, coluna.data)
                  if int(i) not in usadas and abs(v) > TOL]
        if not outras:
            break
        i, aij = max(outras, key=lambda t: abs(t[1]))
        lam = -a[j] / aij
        # so pode inverter o sentido de linha de igualdade
        if lam < 0 and not dados['igualdade'][i]:
            break
        a = a + lam * np.asarray(A[i].todense()).ravel()
        b = b + lam * rhs[i]
        usadas.add(i)
    return a, b


def transladar(a, b, xr, lb, ub):
    """Leva cada variavel ao bound mais proximo do ponto relaxado; devolve None se for livre."""
    sup = np.nonzero(np.abs(a) > TOL)[0]
    perto_lb = (xr[sup] - lb[sup]) <= (ub[sup] - xr[sup])
    complementar = ~perto_lb & np.isfinite(ub[sup])
    desloca = perto_lb & np.isfinite(lb[sup])
    if not np.all(complementar | desloca):  # variavel livre (ex.: theta) impede o corte
        return None
    a_lin = np.where(complementar, -a[sup], a[sup])
    ub_c = np.where(complementar, ub[sup], 0.0)
    lb_c = np.where(complementar, 0.0, lb[sup])
    b_lin = b - np.sum(np.where(complementar, a[sup] * ub_c, a[sup] * lb_c))
    return sup, a_lin, b_lin, complementar


def destransladar(sup, coef, r, complementar, lb, ub, n):
    """Volta da linha transladada para as variaveis originais."""
    orig = np.zeros(n)
    orig[sup] = np.where(complementar, -coef, coef)
    # zera os bounds infinitos do ramo nao usado (evita 0 * inf = nan no np.where)
    ub_c = np.where(complementar, ub[sup], 0.0)
    lb_c = np.where(complementar, 0.0, lb[sup])
    r = r + np.sum(np.where(complementar, -coef * ub_c, coef * lb_c))
    return orig, r


def mir(model, where, dados=None, max_agg=10, max_nos=10, max_cortes=20, viol_min=1e-4):
    if where != gp.GRB.Callback.MIPNODE:
        return
    status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)
    no = int(model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT))
    if status != gp.GRB.OPTIMAL:
        return
    if no >= max_nos:                       # so separa nos max_nos primeiros nos
        return
    if DEBUG_AGG:
        print(f"[mir] MIPNODE no={no} status={status}", flush=True)

    A, rhs, lb, ub, inteira = dados['A'], dados['rhs'], dados['lb'], dados['ub'], dados['inteira']
    xr = np.array(model.cbGetNodeRel(model._vars))
    n = xr.size
    if not hasattr(model, '_mir_cache'):
        model._mir_cache = set()

    # linhas candidatas: as que tocam alguma inteira fracionaria no ponto relaxado
    frac = np.nonzero(inteira & (np.abs(xr - np.round(xr)) > 1e-4))[0]
    if frac.size == 0:
        if DEBUG_AGG:
            print(f"[mir] no={no}: nenhum corte (sem inteira fracionaria)", flush=True)
        return
    linhas = np.unique(A[:, frac].tocsc().tocoo().row)

    cortes = 0
    pior_viol = -np.inf
    for linha0 in linhas:
        if cortes >= max_cortes:
            break
        a, b = agregar(dados, int(linha0), xr, max_agg)
        base = transladar(a, b, xr, lb, ub)
        if base is None:
            continue
        sup, a_lin, b_lin, complementar = base

        melhor = None
        for escala in ESCALAS:
            saida = funcao_mir(escala * a_lin, escala * b_lin, inteira[sup])
            if saida is None:
                continue
            coef, r = destransladar(sup, saida[0], saida[1], complementar, lb, ub, n)
            viol = float(coef @ xr - r)
            nz = np.abs(coef[np.abs(coef) > 1e-10])
            if nz.size == 0 or nz.max() / nz.min() > 1e7:  # descarta corte mal condicionado
                continue
            # violacao relativa: escala com o RHS para nao aceitar corte numericamente irrelevante
            viol_rel = viol / max(1.0, abs(r))
            pior_viol = max(pior_viol, viol_rel)
            if viol_rel > viol_min and (melhor is None or viol_rel > melhor[0]):
                melhor = (viol_rel, coef, r, viol)

        if melhor is not None:
            viol_rel, coef, r, viol_abs = melhor
            idx = np.nonzero(np.abs(coef) > 1e-10)[0]
            # dedupe: assinatura do corte normalizado pelo maior coeficiente
            esc = float(np.max(np.abs(coef[idx])))
            assin = (tuple(int(j) for j in idx),
                     tuple(np.round(coef[idx] / esc, 5).tolist()),
                     round(float(r) / esc, 5))
            if assin in model._mir_cache:
                continue
            model._mir_cache.add(assin)
            model.cbCut(gp.quicksum(float(coef[j]) * model._vars[j] for j in idx) <= float(r))
            cortes += 1
            model._mir_total = getattr(model, '_mir_total', 0) + 1
            print(f"[mir] no {no}: corte #{model._mir_total} na linha {int(linha0)} "
                  f"| viol_rel={viol_rel:.4g} viol={viol_abs:.4g} rhs={r:.4g} nz={idx.size}",
                  flush=True)

    if DEBUG_AGG and cortes == 0:
        print(f"[mir] no={no}: nenhum corte "
              f"(maior viol_rel vista = {pior_viol:.4g}, limiar = {viol_min:.4g})", flush=True)

###################### AGREGACAO 1a LEI (KCL) ######################
# Callback de plano de corte por AGREGACAO do atendimento a demanda.
#
# Somando a 1a lei de Kirchhoff (KCL, restricao 'KCL') sobre TODAS as barras,
# o termo de rede A @ f se cancela: cada linha entra com +1 na origem e -1 no
# destino, entao a coluna de A soma zero. Sobra, em cada etapa t, o balanco de
# energia do sistema:
#
#     sum_u g_uhe[u,t] + sum_g g_ute[g,t] + S_t = D_t
#     S_t = sum_n deficit[n,t]        D_t = sum_n demanda[n,t]
#
# Com os limites de comprometimento  g_uhe[u,t] <= uhe_pot[u] * uch[u,t]  e
# g_ute[g,t] <= ute_pot[g] * uct[g,t], obtem-se a mochila valida
#
#     sum_j p_j z_j + S_t >= D_t ,   z_j in {0,1} (uch/uct),  S_t >= 0        (*)
#
# onde p_j sao as potencias maximas e z_j as binarias de operacao. (*) ja e
# satisfeita pela relaxacao continua; a forca vem de aplicar arredondamento
# inteiro misto (MIR) usando a integralidade das binarias. Complementando
# z_j = 1 - zbar_j e passando a '<=':
#
#     sum_j (lam p_j) zbar_j - lam S_t <= lam (P - D_t) =: b ,   P = sum_j p_j
#
# MIR de b: f0 = b - floor(b);  abar_j = floor(a_j) + (frac(a_j) - f0)^+/(1-f0)
# para as inteiras; coef -lam/(1-f0) para o continuo S_t; RHS floor(b).
# Voltando a z_j, o corte separado e
#
#     sum_j abar_j z_j + (lam/(1-f0)) S_t >= sum_j abar_j - floor(b)          (CUT)
#
# Para lam = 1 e p_j inteiro isso vira  sum_j p_j z_j + S_t/(1-f0) >= ceil(D_t),
# ou seja: arredonda a demanda para cima e encarece o deficit.

ESCALAS_AGG = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 4.0, 0.25)


def _preparar_agregacao_1a_lei(model):
    """Pre-computa, por etapa, potencias, binarias e vars de deficit da KCL agregada."""
    uch_v, uct_v, def_v = uch.tolist(), uct.tolist(), deficit.tolist()
    etapas_agg = []
    for t in range(T):
        pares = ([(float(uhe_pot[u]), uch_v[u][t]) for u in range(H)]
                 + [(float(ute_pot[g]), uct_v[g][t]) for g in range(G)])
        etapas_agg.append(dict(
            p=np.array([c for c, _ in pares], dtype=float),
            z=[v for _, v in pares],
            deficit=[def_v[n][t] for n in range(N)],
            D=float(demanda[:, t].sum())))
    return etapas_agg


def _mir_agregado(p, zr, sr, D, viol_min):
    """MIR da mochila (*) no ponto relaxado (zr, sr).

    Retorna (violacao, abar, cs, rhs) do corte  abar . z + cs * S >= rhs  mais
    violado entre as escalas, ou None se nenhum corta o ponto.
    """
    P = float(p.sum())
    melhor = None
    for lam in ESCALAS_AGG:
        b = lam * (P - D)
        f0 = b - np.floor(b)
        if f0 < TOL or f0 > 1.0 - TOL:          # RHS quase inteiro -> MIR trivial
            continue
        a = lam * p
        fa = a - np.floor(a)
        abar = np.floor(a) + np.maximum(fa - f0, 0.0) / (1.0 - f0)
        cs = lam / (1.0 - f0)                   # coef do deficit total S
        rhs = float(abar.sum() - np.floor(b))
        if rhs <= viol_min:                     # 0 <= rhs: corte redundante
            continue
        viol = rhs - (float(abar @ zr) + cs * sr)
        if viol > viol_min and (melhor is None or viol > melhor[0]):
            melhor = (viol, abar, cs, rhs)
    return melhor


def _folga_agregada(p, zr, sr, D):
    """Maior violacao do corte MIR entre as escalas, ignorando o limiar (p/ debug)."""
    P = float(p.sum())
    pior = -np.inf
    for lam in ESCALAS_AGG:
        b = lam * (P - D)
        f0 = b - np.floor(b)
        if f0 < TOL or f0 > 1.0 - TOL:
            continue
        a = lam * p
        abar = np.floor(a) + np.maximum((a - np.floor(a)) - f0, 0.0) / (1.0 - f0)
        cs = lam / (1.0 - f0)
        rhs = float(abar.sum() - np.floor(b))
        pior = max(pior, rhs - (float(abar @ zr) + cs * sr))
    return pior


def agregacao_1a_lei(model, where, max_cortes=8, viol_min=1e-5):
    if where != gp.GRB.Callback.MIPNODE:
        return
    status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)
    no = int(model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT))
    if status != gp.GRB.OPTIMAL:
        return
    if no != 0:                                    # so no no raiz
        return
    if DEBUG_AGG:
        print(f"[agregacao_1a_lei] MIPNODE no={no} status={status}", flush=True)

    etapas_agg = getattr(model, '_agg1', None)
    if etapas_agg is None:
        etapas_agg = _preparar_agregacao_1a_lei(model)
        model._agg1 = etapas_agg
        model._agg1_cache = set()

    cortes = 0
    pior_viol = -np.inf
    for t in range(T):
        if cortes >= max_cortes:
            break
        dados_t = etapas_agg[t]
        vals = model.cbGetNodeRel(dados_t['z'] + dados_t['deficit'])
        nz = len(dados_t['z'])
        zr = np.array(vals[:nz], dtype=float)
        sr = float(sum(vals[nz:]))

        melhor = _mir_agregado(dados_t['p'], zr, sr, dados_t['D'], viol_min)
        if melhor is None:
            if DEBUG_AGG:
                folga = _folga_agregada(dados_t['p'], zr, sr, dados_t['D'])
                pior_viol = max(pior_viol, folga)
            continue
        _, abar, cs, rhs = melhor

        assin = (t, round(rhs, 4), round(cs, 4), tuple(np.round(abar, 4).tolist()))
        if assin in model._agg1_cache:            # nao repetir o mesmo corte
            continue

        idx = np.nonzero(np.abs(abar) > 1e-12)[0]
        if idx.size == 0:
            continue
        model._agg1_cache.add(assin)
        corte = (gp.quicksum(float(abar[j]) * dados_t['z'][j] for j in idx)
                 + cs * gp.quicksum(dados_t['deficit']))
        model.cbCut(corte >= rhs)
        cortes += 1
        model._agg1_total = getattr(model, '_agg1_total', 0) + 1
        print(f"[agregacao_1a_lei] no {no}: corte #{model._agg1_total} na etapa {t} "
              f"| viol={melhor[0]:.4g} rhs={rhs:.4g} coef_deficit={cs:.4g} nz={idx.size}",
              flush=True)

    if DEBUG_AGG and cortes == 0:
        print(f"[agregacao_1a_lei] no={no}: nenhum corte "
              f"(maior violacao vista = {pior_viol:.4g}, limiar = {viol_min:.4g})", flush=True)

####################### MODELO #######################
FLAG_PRINT = False
m = gp.Model("Despacho Hidrotérmico - IEEE 14 barras")
m.Params.Threads = 1
m.Params.Presolve = 0
m.Params.Cuts = 0
m.Params.Heuristics = 0
m.Params.PreCrush = 1
m.Params.MIPGap = 1e-3
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

# Reserva girante

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

m.update()
m._vars = m.getVars()
# m.optimize(partial(mir, dados=dados_matriciais(m)))
m.optimize(agregacao_1a_lei)

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
