"""
Branch & Bound "puro" em PYOMO (equivalente a branch&bound.py).

Resolve o MIP completo com presolve, cortes e heuristicas DESLIGADOS (quando o
solver permite -- ver OPCOES em modelo_pyomo.py), gap 1e-3, 1 thread.

    python branch_bound_pyomo.py                 # appsi_highs
    python branch_bound_pyomo.py gurobi
    python branch_bound_pyomo.py cbc

Edite TEMPO / GAP / PURO abaixo. PURO=False liga os recursos padrao do solver
(branch-and-cut completo) para comparar.
"""

import numpy as np
from modelo_pyomo import (escolher_solver, construir_modelo, criar_solver, resolver,
                          relatorio, arr, G, H, T, N, L)

TEMPO = 600
GAP = 1e-3
PURO = True

SOLVER = escolher_solver()
print(f"Solver: {SOLVER}   (puro={PURO})   T={T}")

m = construir_modelo()
r = resolver(criar_solver(SOLVER, TEMPO, GAP, PURO), SOLVER, m)
relatorio('branch & bound', r)

if r['tem']:
    np.savez(f'bb_pyomo_{SOLVER}.npz', custo_total=r['obj'], bound=r['bound'] or np.nan,
             tempo=r['tempo'], uct=arr(m.uct, G), part_ute=arr(m.part, G),
             desl_ute=arr(m.desl, G), uch=arr(m.uch, H), g_ute=arr(m.g_ute, G),
             g_uhe=arr(m.g_uhe, H), deficit=arr(m.deficit, N))
    print(f"bb_pyomo_{SOLVER}.npz gerado.")
