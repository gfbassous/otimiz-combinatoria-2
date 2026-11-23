"""
Despacho hidrotermico (mesmo modelo de branch&bound.py) em PROGRAMACAO POR
RESTRICOES com docplex.cp (CP Optimizer).

CP Optimizer so trabalha com variaveis INTEIRAS, entao todo o modelo e escalado:

    grandeza                      unidade original  ->  unidade no CP     fator
    potencia, fluxo, deficit      MW                ->  kW                x1000
    vazao, turb., vert., defl.    MW-equivalente    ->  kW-equivalente    x1000
    armazenamento                 MWh               ->  kWh               x1000
    susceptancia B = S_base/x     MW/rad            ->  kW/rad            x1000
    custo de partida              R$                ->  mR$ (milirreais)  x1000
    CVU e custo de deficit        R$/MWh            ->  mR$/kWh           x1 (mesmo numero)
    objetivo                      R$                ->  mR$               x1000

Com isso o objetivo do CP e exatamente 1000 x o objetivo do MIP. Coeficientes
adimensionais (produtibilidade = 1) nao mudam. Cada constante escalada e
arredondada para inteiro; o script imprime o maior erro de arredondamento.

EXCECAO -- 2a LEI DE KIRCHHOFF. f = B (theta_i - theta_j) com f e theta inteiros
so tem solucao em multiplos de B: com theta em mrad (x1000) o fluxo andaria em
degraus de 0.2 a 2.4 MW e a 1a lei viraria um problema de divisibilidade. Por isso:
  * theta e medido em MICRORRADIANOS (x 1e6);
  * a igualdade vira "f e o inteiro mais proximo de B * delta_theta":
        -500000 <= 1e6 * f_kW - B_kW/rad * delta_theta_urad <= 500000
    (erro de no maximo 0.5 kW = 0.0005 MW por linha).

Variaveis sem limite no MIP (theta, deficit, vertimento, defluencia) recebem
limites VALIDOS (nao cortam solucoes), porque CP precisa de dominios finitos.

    python cp-despacho.py
"""

import heapq
import time

import numpy as np
import pandas as pd
from docplex.cp.model import CpoModel
from docplex.cp.expression import integer_var, binary_var
import docplex.cp.modeler as cp

###################### PARAMETROS ######################
CROP_ETAPAS = 24        # 0 = horizonte completo de dados.xlsx (pesado para CP)
TEMPO = 120             # s
WORKERS = 0             # 0 = todos os nucleos
ESCALA = 1000           # fator das constantes
ESCALA_THETA = 10 ** 6  # theta em microrradianos (ver docstring)
CPOPTIMIZER = "/opt/ibm/ILOG/CPLEX_Studio222/cpoptimizer/bin/x86-64_linux/cpoptimizer"

####################### DADOS (iguais a branch&bound.py) #######################
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
uhe_rampa = 0.1 * uhe_pot
uhe_montante = [None, 'UHE1', 'UHE2', 'UHE3']

ute_pot = np.array([110., 85., 70., 60., 45.])
ute_cvu = np.array([42., 85., 130., 210., 450.])
ute_g_min = np.array([22., 25., 22., 15., 5.])
ute_rampa = np.array([22., 25., 30., 40., 45.])
ute_barra = ['B3', 'B6', 'B8', 'B2', 'B13']
ute_custo_partida = np.array([9000., 4500., 2600., 1300., 450.])
t_on = np.array([6, 4, 3, 2, 2])
t_off = np.array([5, 3, 2, 2, 2])
uct_inicial = np.array([1, 0, 0, 0, 0])

####################### ESCALA PARA INTEIROS #######################
erros = {}


def inteiro(nome, v, fator=ESCALA):
    """fator * v arredondado para inteiro (int do Python), registrando o erro."""
    a = np.asarray(v, dtype=float) * fator
    r = np.rint(a)
    erros[nome] = float(np.max(np.abs(a - r)) / fator) if a.size else 0.0
    return r.astype(np.int64)


D = inteiro('demanda', demanda)                    # kW
Q = inteiro('vazao', vaz_incr)                     # kW-eq
CAP_L = inteiro('capacidade_linhas', capacidade_linhas)
B_L = inteiro('susceptancia', S_BASE / x_linhas)   # kW/rad
C_DEF = inteiro('c_deficit', c_deficit, 1)         # mR$/kWh
P_H = inteiro('uhe_pot', uhe_pot)
PMIN_H = inteiro('uhe_g_min', uhe_g_min)
V_CAP = inteiro('uhe_arm_cap', uhe_arm_cap)
V_INI = inteiro('arm_inicial', uhe_arm_cap / 2)
DMIN = inteiro('defl_min', defl_min)
R_H = inteiro('uhe_rampa', uhe_rampa)
PROD = inteiro('prod_uhe', prod_uhe, 1)            # adimensional
P_T = inteiro('ute_pot', ute_pot)
PMIN_T = inteiro('ute_g_min', ute_g_min)
R_T = inteiro('ute_rampa', ute_rampa)
CVU = inteiro('ute_cvu', ute_cvu, 1)               # mR$/kWh
C_PART = inteiro('ute_custo_partida', ute_custo_partida)  # mR$
if np.any(np.abs(PROD - prod_uhe) > 0):
    raise ValueError('produtibilidade nao inteira: g = prod * turb deixaria de ser exata')

idx_barra = {b: k for k, b in enumerate(barras)}
de = [idx_barra[i] for i, _ in linhas]
para = [idx_barra[j] for _, j in linhas]
bar_h = [idx_barra[b] for b in uhe_barra]
bar_t = [idx_barra[b] for b in ute_barra]
mont = [uhe.index(m) if m else None for m in uhe_montante]

# ---- limites validos para variaveis sem limite no MIP ----
# |theta_n| <= menor soma de |delta theta|max = cap*x/S_base ao longo de um caminho ate a referencia
peso = np.ceil(capacidade_linhas * x_linhas / S_BASE * ESCALA_THETA).astype(np.int64)
adj = {n: [] for n in range(N)}
for l in range(L):
    adj[de[l]].append((para[l], int(peso[l])))
    adj[para[l]].append((de[l], int(peso[l])))
TH_MAX = [None] * N
fila = [(0, barra_ref)]
while fila:
    d, n = heapq.heappop(fila)
    if TH_MAX[n] is not None:
        continue
    TH_MAX[n] = d
    for k, w in adj[n]:
        if TH_MAX[k] is None:
            heapq.heappush(fila, (d + w, k))
# deficit: r_n = D_n + (saida liquida) - geracao <= D_n + soma das capacidades das linhas da barra
CAP_BARRA = np.zeros(N, dtype=np.int64)
for l in range(L):
    CAP_BARRA[de[l]] += CAP_L[l]
    CAP_BARRA[para[l]] += CAP_L[l]
# defluencia: d_u,t <= v_u,t-1 + q_u,t + d_montante,t <= V_cap + max vazao + limite do montante
DEFL_MAX = np.zeros(H, dtype=np.int64)
for u in range(H):                      # a ordem da lista ja e montante -> jusante
    DEFL_MAX[u] = V_CAP[u] + Q[u].max() + (DEFL_MAX[mont[u]] if mont[u] is not None else 0)

####################### MODELO CP #######################
t0 = time.time()
mdl = CpoModel(name='despacho_hidrotermico')


def por_etapa(v, n):
    """Escalar, vetor (n,) ou matriz (n, T) -> matriz (n, T)."""
    a = np.asarray(v)
    if a.ndim == 0:
        return np.full((n, T), a)
    return np.repeat(a[:, None], T, axis=1) if a.ndim == 1 else a


def matriz(n, nome, lo, hi):
    """Matriz (n, T) de variaveis inteiras com dominio [lo, hi]."""
    lo, hi = por_etapa(lo, n), por_etapa(hi, n)
    return [[integer_var(int(lo[i, t]), int(hi[i, t]), f'{nome}_{i}_{t}') for t in range(T)]
            for i in range(n)]


def binarias(n, nome):
    return [[binary_var(f'{nome}_{i}_{t}') for t in range(T)] for i in range(n)]


g_uhe = matriz(H, 'g_uhe', 0, P_H)
uch = binarias(H, 'uch')
turb = matriz(H, 'turb', 0, P_H // np.maximum(PROD, 1))
vert = matriz(H, 'vert', 0, DEFL_MAX)
defl = matriz(H, 'defl', DMIN, DEFL_MAX)
arm = matriz(H, 'arm', 0, V_CAP)

g_ute = matriz(G, 'g_ute', 0, P_T)
uct = binarias(G, 'uct')
part = binarias(G, 'part')
desl = binarias(G, 'desl')

f = matriz(L, 'f', -CAP_L, CAP_L)
theta = matriz(N, 'theta', -np.array(TH_MAX), np.array(TH_MAX))
deficit = matriz(N, 'deficit', 0, D + CAP_BARRA[:, None])

add = mdl.add
for t in range(T):
    # 1a lei de Kirchhoff: geracao + deficit - saida liquida = demanda
    for n in range(N):
        termos = [g_uhe[u][t] for u in range(H) if bar_h[u] == n] \
               + [g_ute[g][t] for g in range(G) if bar_t[g] == n] + [deficit[n][t]]
        termos += [-f[l][t] for l in range(L) if de[l] == n] + [f[l][t] for l in range(L) if para[l] == n]
        add(cp.sum(termos) == int(D[n, t]))
    # 2a lei: f = round(B * delta theta), theta em urad
    for l in range(L):
        resto = ESCALA_THETA * f[l][t] - int(B_L[l]) * (theta[de[l]][t] - theta[para[l]][t])
        add(cp.range(resto, -ESCALA_THETA // 2, ESCALA_THETA // 2))
    add(theta[barra_ref][t] == 0)

    for u in range(H):
        add(turb[u][t] + vert[u][t] == defl[u][t])
        add(g_uhe[u][t] == int(PROD[u]) * turb[u][t])
        anterior = int(V_INI[u]) if t == 0 else arm[u][t - 1]
        chegada = defl[mont[u]][t] if mont[u] is not None else 0
        add(arm[u][t] == anterior + int(Q[u, t]) - turb[u][t] - vert[u][t] + chegada)
        add(g_uhe[u][t] <= int(P_H[u]) * uch[u][t])
        add(g_uhe[u][t] >= int(PMIN_H[u]) * uch[u][t])
        if t > 0:
            add(cp.range(defl[u][t] - defl[u][t - 1], -int(R_H[u]), int(R_H[u])))

    for g in range(G):
        add(g_ute[g][t] <= int(P_T[g]) * uct[g][t])
        add(g_ute[g][t] >= int(PMIN_T[g]) * uct[g][t])
        if t > 0:
            add(cp.range(g_ute[g][t] - g_ute[g][t - 1], -int(R_T[g]), int(R_T[g])))
        anterior = int(uct_inicial[g]) if t == 0 else uct[g][t - 1]
        add(uct[g][t] - anterior == part[g][t] - desl[g][t])
        add(part[g][t] + desl[g][t] <= 1)
        if t >= t_on[g] - 1:
            add(cp.sum(part[g][t - t_on[g] + 1:t + 1]) <= uct[g][t])
        if t >= t_off[g] - 1:
            add(cp.sum(desl[g][t - t_off[g] + 1:t + 1]) <= 1 - uct[g][t])

for u in range(H):
    add(arm[u][T - 1] >= int(V_INI[u]))

custo = cp.sum([int(CVU[g]) * g_ute[g][t] + int(C_PART[g]) * part[g][t]
                for g in range(G) for t in range(T)]) \
      + int(C_DEF) * cp.sum([deficit[n][t] for n in range(N) for t in range(T)])
add(cp.minimize(custo))

n_var = T * (6 * H + 4 * G + L + 2 * N)
print(f"T={T} etapas | {n_var} variaveis inteiras | modelo montado em {time.time() - t0:.1f} s")
print("maior erro de arredondamento (unidade original): "
      + (", ".join(f"{k}={v:.2g}" for k, v in erros.items() if v > 0) or "nenhum"))

####################### SOLUCAO #######################
msol = mdl.solve(TimeLimit=TEMPO, Workers=WORKERS or None, LogPeriod=100000,
                 execfile=CPOPTIMIZER)
print(f"\nstatus: {msol.get_solve_status()}  tempo: {msol.get_solve_time():.1f} s")
if not msol:
    raise SystemExit("CP Optimizer nao encontrou solucao viavel no tempo dado.")


def val(m):
    return np.array([[msol.get_value(v) for v in linha] for linha in m], dtype=float)


obj = msol.get_objective_value()
bound = msol.get_objective_bound()
print(f"custo: {obj / ESCALA:,.3f} R$   (bound {bound / ESCALA:,.3f} R$)")

g_h, g_t, F, TH, R = val(g_uhe) / ESCALA, val(g_ute) / ESCALA, val(f) / ESCALA, \
                     val(theta) / ESCALA_THETA, val(deficit) / ESCALA
kvl = F - (S_BASE / x_linhas)[:, None] * (TH[de] - TH[para])
print(f"erro maximo da 2a lei: {np.abs(kvl).max():.2e} MW | deficit total: {R.sum():.3f} MWh")
print(f"partidas por UTE: {val(part).sum(axis=1).astype(int).tolist()}")

np.savez('cp_despacho.npz', custo_total=obj / ESCALA, bound=bound / ESCALA, T=T,
         g_uhe=g_h, g_ute=g_t, uch=val(uch), uct=val(uct), part_ute=val(part),
         desl_ute=val(desl), turb_uhe=val(turb) / ESCALA, vert_uhe=val(vert) / ESCALA,
         defl_uhe=val(defl) / ESCALA, uhe_arm=val(arm) / ESCALA, fluxo=F, theta=TH,
         deficit=R, demanda=demanda)
print("cp_despacho.npz gerado (valores de volta em MW/MWh/rad/R$).")
