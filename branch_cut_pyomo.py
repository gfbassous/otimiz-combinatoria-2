"""
Branch & Cut em PYOMO (equivalente a branch&cut.py, versao CUT-AND-BRANCH).

Pyomo nao expoe callbacks de forma portavel entre solvers (cbCut so existe com
gurobi_persistent). Para funcionar com QUALQUER solver, os cortes sao separados
em um laco no NO RAIZ, resolvendo a relaxacao linear:

    repita ate nao haver corte violado (ou MAX_RONDAS):
        resolve LP -> separa MIR agregado da 1a lei de Kirchhoff -> adiciona corte
    resolve o MIP (B&B puro, sem cortes/presolve do solver) com os cortes ja no modelo

Os cortes sao validos, entao o otimo nao muda; so o bound da raiz melhora. A
diferenca para o callback original: la os cortes tambem eram separados em nos
internos (aqui so na raiz -- no original tambem era so no no 0, `no != 0`).

CORTE (agregacao da KCL, ver branch&cut.py): somando a KCL sobre todas as barras
a rede se cancela e, com g <= Pmax * z, sai a mochila
    sum_j p_j z_j + S_t >= D_t    (z = uch/uct, S_t = deficit total)
cujo MIR (arredondamento inteiro misto) e
    sum_j abar_j z_j + lam/(1-f0) S_t >= sum_j abar_j - floor(b)

    python branch_cut_pyomo.py                 # appsi_highs
    python branch_cut_pyomo.py gurobi
"""

import numpy as np
import pyomo.environ as pyo
from modelo_pyomo import (escolher_solver, construir_modelo, relaxar, criar_solver,
                          resolver, relatorio, arr, G, H, T, N, uhe_pot, ute_pot, demanda)

TEMPO = 600
GAP = 1e-3
MAX_RONDAS = 10
MAX_CORTES_RONDA = 50
VIOL_MIN = 1e-5
TOL = 1e-6
ESCALAS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 4.0, 0.25)

SOLVER = escolher_solver()
print(f"Solver: {SOLVER}   T={T}")

p = np.concatenate([uhe_pot, ute_pot])   # potencias das H+G binarias de comprometimento
P = float(p.sum())
D = demanda.sum(axis=0)                  # demanda total por etapa


def mir_agregado(zr, sr, Dt):
    """Melhor corte (violacao, abar, cs, rhs) entre as escalas, ou None."""
    melhor = None
    for lam in ESCALAS:
        b = lam * (P - Dt)
        f0 = b - np.floor(b)
        if f0 < TOL or f0 > 1.0 - TOL:
            continue
        a = lam * p
        abar = np.floor(a) + np.maximum(a - np.floor(a) - f0, 0.0) / (1.0 - f0)
        cs = lam / (1.0 - f0)
        rhs = float(abar.sum() - np.floor(b))
        if rhs <= VIOL_MIN:
            continue
        viol = rhs - (float(abar @ zr) + cs * sr)
        if viol > VIOL_MIN and (melhor is None or viol > melhor[0]):
            melhor = (viol, abar, cs, rhs)
    return melhor


def expr_corte(mod, t, abar, cs, rhs):
    z = [mod.uch[u, t] for u in range(H)] + [mod.uct[g, t] for g in range(G)]
    S = sum(mod.deficit[n, t] for n in range(N))
    return sum(float(abar[j]) * z[j] for j in range(len(z)) if abs(abar[j]) > 1e-12) + cs * S >= rhs


m = construir_modelo()
m_lp = relaxar(m)
m.cortes = pyo.ConstraintList()
m_lp.cortes = pyo.ConstraintList()
solver_lp = criar_solver(SOLVER, TEMPO, GAP, puro=False)

total = 0
lb_inicial = None
print(f"\n{'rodada':>6} {'LP (raiz)':>16} {'cortes':>7}")
for rodada in range(MAX_RONDAS + 1):
    r = resolver(solver_lp, SOLVER, m_lp)
    if not r['tem']:
        print("LP sem solucao -- parando")
        break
    if lb_inicial is None:
        lb_inicial = r['obj']
    novos = 0
    if rodada < MAX_RONDAS:
        for t in range(T):
            if novos >= MAX_CORTES_RONDA:
                break
            zr = np.array([pyo.value(m_lp.uch[u, t]) for u in range(H)]
                          + [pyo.value(m_lp.uct[g, t]) for g in range(G)])
            sr = sum(pyo.value(m_lp.deficit[n, t]) for n in range(N))
            c = mir_agregado(zr, sr, float(D[t]))
            if c is None:
                continue
            _, abar, cs, rhs = c
            m.cortes.add(expr_corte(m, t, abar, cs, rhs))
            m_lp.cortes.add(expr_corte(m_lp, t, abar, cs, rhs))
            novos += 1
    total += novos
    print(f"{rodada:>6} {r['obj']:>16,.2f} {novos:>7}")
    if novos == 0:
        break

print(f"\nraiz: LP {lb_inicial:,.2f} -> {r['obj']:,.2f} com {total} cortes "
      f"(+{r['obj'] - lb_inicial:,.2f})")

# MIP com os cortes, sem presolve/cortes proprios do solver (como no original)
rm = resolver(criar_solver(SOLVER, TEMPO, GAP, puro=True), SOLVER, m)
relatorio('branch & cut (cut-and-branch)', rm)

if rm['tem']:
    np.savez(f'bc_pyomo_{SOLVER}.npz', custo_total=rm['obj'], bound=rm['bound'] or np.nan,
             tempo=rm['tempo'], n_cortes=total, lp_sem_cortes=lb_inicial, lp_com_cortes=r['obj'],
             uct=arr(m.uct, G), part_ute=arr(m.part, G), desl_ute=arr(m.desl, G),
             uch=arr(m.uch, H), g_ute=arr(m.g_ute, G), g_uhe=arr(m.g_uhe, H),
             deficit=arr(m.deficit, N))
    print(f"bc_pyomo_{SOLVER}.npz gerado.")
