"""
BRANCH-AND-PRICE do despacho hidrotermico sobre a reformulacao de Dantzig-Wolfe de
dantzig_wolfe.py (mestre = hidraulica + rede; um pricing X^g por UTE).

Em cada no da arvore a relaxacao e o MESTRE LINEAR resolvido por geracao de
colunas; o limite do no e z_RMP na convergencia (ou o limite de Lagrange, se a
geracao for interrompida porque o no ja nao pode melhorar o incumbente).

================================ RAMIFICACAO ================================

Ramifica-se nas variaveis ORIGINAIS, nunca nos lambdas (ramificar em lambda
destruiria a estrutura do pricing):

  uct[g,t] = sum_k lam_gk u^gk_t  fracionario
      ramo 0: u_t = 0 no pricing de g; colunas com u^gk_t = 1 ficam com lam <= 0
      ramo 1: u_t = 1 no pricing de g; colunas com u^gk_t = 0 ficam com lam <= 0
  uch[u,t] fracionario (variavel do mestre)
      ramo 0 / ramo 1: limite da variavel no mestre

Prioridade: uct primeiro (e de onde vem o gap; ver dantzig_wolfe.py), depois uch;
dentro de cada grupo, a mais fracionaria.

Por que isso basta: se todo uct agregado e inteiro, todas as colunas com lam > 0
da UTE g tem o MESMO comprometimento u (combinacao convexa de 0/1 so da 0/1 se
todos concordam). Com u fixo, X^g e um poliedro, entao p agregado e viavel. Se
uch tambem e inteiro e as artificiais sao zero, a solucao do RMP e viavel para o
problema original e vira incumbente.

COMPLETACAO HIDRO: quando todo uct agregado ja e inteiro mas sobra uch fracionaria,
o mestre e resolvido com uch binaria e so as colunas daquele comprometimento
(dw.Decomposicao.completar_hidro). Se o custo atinge o limite do no, o no esta
resolvido sem ramificar em uch (a integralidade da hidraulica quase nao pesa aqui).

Selecao de no: mergulho em profundidade ate o primeiro incumbente, depois
melhor limite. A heuristica do mestre restrito inteiro roda na raiz e a cada
HEUR_FREQ nos.

    python branch_and_price.py
"""

import time
from types import SimpleNamespace

import numpy as np

import dantzig_wolfe as dw

###################### PARAMETROS ######################
# o horizonte vem de dw.CROP_ETAPAS (dantzig_wolfe.py)
TEMPO_LIMITE = 600      # s
GAP_BP = 1e-4           # para quando (UB - LB)/UB <= GAP_BP
MAX_NOS = 100000
TOL_INT = 1e-6          # tolerancia de integralidade
HEUR_FREQ = 100         # heuristica do mestre restrito inteiro a cada tantos nos
TEMPO_HEUR = 10         # s da heuristica nos nos (na raiz usa dw.TEMPO_HEUR)
LOG_FREQ = 10           # imprime um no a cada tantos
COMPLETAR_HIDRO = True  # uct inteiro e uch fracionaria: resolve uch como MIP antes de ramificar


def fracionarias(x):
    """Indices e valores das entradas fracionarias, da mais para a menos fracionaria."""
    d = np.minimum(x - np.floor(x), np.ceil(x) - x)
    idx = np.argwhere(d > TOL_INT)
    ordem = np.argsort(-d[tuple(idx.T)], kind='stable') if len(idx) else []
    return [(tuple(idx[i]), x[tuple(idx[i])]) for i in ordem]


def main():
    G, H, T = dw.G, dw.H, dw.T
    print(f"Branch-and-price: G={G} UTE, H={H} UHE, T={T} etapas | "
          f"gap alvo {GAP_BP:.0e}, limite {TEMPO_LIMITE} s\n")
    ref = dw.referencias()
    print(f"[ref] z_LP = {ref['z_lp']:,.2f} | MIP compacto: {ref['ub']:,.2f} "
          f"(bound {ref['bound']:,.2f}, {ref['tempo']:.1f} s)\n")

    t0 = time.time()
    dec = dw.Decomposicao()
    UB, incumbente = np.inf, None
    abertos = [SimpleNamespace(id=0, prof=0, lb=-np.inf, fix_uct={}, fix_uch={}, ramo='raiz')]
    prox_id = 1
    cont = dict(processados=0, podados=0, inviaveis=0, inteiros=0, heur=0, completados=0)
    hist = []
    z_raiz = lb_raiz = None

    def atualizar(sol, origem):
        nonlocal UB, incumbente
        if sol is not None and sol['custo_total'] < UB - 1e-6:
            UB, incumbente = sol['custo_total'], sol
            print(f"      * novo incumbente {UB:,.2f} ({origem})")

    print(f"{'no':>6} {'prof':>4} {'ramo':>14} {'z no':>14} {'LB global':>14} {'UB':>14} "
          f"{'gap':>8} {'frac':>5} {'cols':>6} {'t(s)':>7}")
    status = 'otimo'
    while abertos:
        if time.time() - t0 > TEMPO_LIMITE:
            status = 'limite de tempo'
            break
        if cont['processados'] >= MAX_NOS:
            status = 'limite de nos'
            break
        # selecao: profundidade ate ter incumbente, depois melhor limite
        k = len(abertos) - 1 if UB == np.inf else min(range(len(abertos)), key=lambda i: abertos[i].lb)
        no = abertos.pop(k)
        lb_global = min([no.lb] + [n.lb for n in abertos])
        if UB < np.inf and (UB - lb_global) <= GAP_BP * abs(UB):
            abertos.append(no)
            break
        if no.lb >= UB - GAP_BP * abs(UB):
            cont['podados'] += 1
            continue

        cont['processados'] += 1
        dec.aplicar_no(no.fix_uct, no.fix_uch)
        corte = UB - GAP_BP * abs(UB) if UB < np.inf else np.inf
        res = dec.gerar_colunas(ub=corte)
        if res['status'] == 'inviavel' or (res['status'] != 'podado' and dec.artificiais() > 1e-6):
            cont['inviaveis'] += 1
            continue
        lb_no = max(no.lb, res['lb'] if res['status'] != 'otimo' else res['z'])
        if no.id == 0:
            z_raiz, lb_raiz = res['z'], lb_no
        if res['status'] == 'podado' or lb_no >= UB - GAP_BP * abs(UB):
            cont['podados'] += 1
            continue

        sol = dec.solucao()
        cand = fracionarias(sol['uct'])
        tipo = 'uct'
        if not cand:
            cand, tipo = fracionarias(sol['uch']), 'uch'
        n_frac = len(fracionarias(sol['uct'])) + len(fracionarias(sol['uch']))

        if cand and tipo == 'uch' and COMPLETAR_HIDRO:
            # uct ja e inteiro: tenta fechar o no fixando o comprometimento termico
            # e resolvendo o mestre com uch binaria (exato para esse comprometimento)
            comp = dec.completar_hidro(np.rint(sol['uct']), TEMPO_HEUR)
            cont['completados'] += 1
            atualizar(comp, f'completar hidro, no {no.id}')
            if comp is not None and comp['custo_total'] <= lb_no + GAP_BP * abs(UB):
                cand = []               # a completacao atinge o limite do no: nada a ramificar

        if not cand:
            cont['inteiros'] += 1
            atualizar(sol, f'no {no.id}')
        else:
            if no.id == 0 or cont['processados'] % HEUR_FREQ == 0:
                cont['heur'] += 1
                atualizar(dec.heuristica_inteira(dw.TEMPO_HEUR if no.id == 0 else TEMPO_HEUR),
                          f'mestre restrito inteiro, no {no.id}')
                dec.aplicar_no(no.fix_uct, no.fix_uch)
            (i, t), v = cand[0]
            # o ramo do arredondamento e empilhado por ultimo: e o primeiro do mergulho
            for val in ((0, 1) if v >= 0.5 else (1, 0)):
                fu, fh = dict(no.fix_uct), dict(no.fix_uch)
                (fu if tipo == 'uct' else fh)[(i, t)] = val
                abertos.append(SimpleNamespace(id=prox_id, prof=no.prof + 1, lb=lb_no, fix_uct=fu,
                                               fix_uch=fh, ramo=f'{tipo}[{i},{t}]={val}'))
                prox_id += 1

        lb_global = min([lb_no] + [n.lb for n in abertos]) if abertos else UB
        lb_global = min(lb_global, UB)
        gap = (UB - lb_global) / abs(UB) if UB < np.inf else np.inf
        hist.append((no.id, no.prof, lb_no, res['z'], UB, lb_global, n_frac,
                     dec.n_colunas(), time.time() - t0))
        if no.id == 0 or cont['processados'] % LOG_FREQ == 0 or not cand:
            print(f"{no.id:>6} {no.prof:>4} {no.ramo:>14} {res['z']:>14,.2f} {lb_global:>14,.2f} "
                  f"{UB:>14,.2f} {gap:>8.2e} {n_frac:>5} {dec.n_colunas():>6} "
                  f"{time.time() - t0:>7.1f}")

    lb_final = min([n.lb for n in abertos] + [UB]) if abertos else UB
    t_total = time.time() - t0
    print("\n" + "=" * 66)
    print(f"RESUMO ({status})")
    print("=" * 66)
    print(f"  nos processados / podados / inviaveis / inteiros = {cont['processados']} / "
          f"{cont['podados']} / {cont['inviaveis']} / {cont['inteiros']}   (abertos: {len(abertos)})")
    print(f"  colunas geradas = {dec.n_colunas()} | heuristicas = {cont['heur']} | "
          f"completacoes hidro = {cont['completados']} | tempo = {t_total:.1f} s")
    print(f"  z_LP  (relaxacao linear)    = {ref['z_lp']:>16,.2f}")
    print(f"  z_DW  (raiz)                = {lb_raiz:>16,.2f}")
    print(f"  LB    (branch-and-price)    = {lb_final:>16,.2f}")
    print(f"  UB    (branch-and-price)    = {UB:>16,.2f}")
    print(f"  gap   (UB - LB)/UB          = {100 * (UB - lb_final) / abs(UB):>15.4f}%")
    print(f"  MIP compacto (referencia)   = {ref['ub']:>16,.2f}  (bound {ref['bound']:,.2f})")

    if incumbente is not None:
        h = np.array(hist)
        np.savez('branch_and_price.npz', no=h[:, 0], profundidade=h[:, 1], lb_no=h[:, 2],
                 z_no=h[:, 3], ub=h[:, 4], lb_global=h[:, 5], n_frac=h[:, 6], colunas=h[:, 7],
                 tempo=h[:, 8], z_lp=ref['z_lp'], z_dw=lb_raiz, lb_final=lb_final, ub_final=UB,
                 tempo_total=t_total, **{k: v for k, v in incumbente.items()})
        print("\nbranch_and_price.npz gerado.")


if __name__ == '__main__':
    main()
