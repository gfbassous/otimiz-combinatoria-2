"""Definicao do sistema IEEE 14 barras adaptado e geracao das series horarias."""
import numpy as np

S_BASE = 100.0
SEMANAS = 3                 # horizonte em semanas; T e derivado daqui
HORAS_SEMANA = 7 * 24
T = SEMANAS * HORAS_SEMANA

barras = [f'B{i}' for i in range(1, 15)]

# (de, para, reatancia pu 100 MVA, capacidade MW)  -- IEEE 14 bus branch data
ramos = [
    ('B1', 'B2', 0.05917, 190.),
    ('B1', 'B5', 0.22304, 100.),
    ('B2', 'B3', 0.19797, 120.),
    ('B2', 'B4', 0.17632, 100.),
    ('B2', 'B5', 0.17388, 90.),
    ('B3', 'B4', 0.17103, 80.),
    ('B4', 'B5', 0.04211, 130.),
    ('B4', 'B7', 0.20912, 90.),
    ('B4', 'B9', 0.55618, 45.),
    ('B5', 'B6', 0.25202, 110.),
    ('B6', 'B11', 0.19890, 55.),
    ('B6', 'B12', 0.25581, 40.),
    ('B6', 'B13', 0.13027, 48.),
    ('B7', 'B8', 0.17615, 55.),
    ('B7', 'B9', 0.11001, 100.),
    ('B9', 'B10', 0.08450, 55.),
    ('B9', 'B14', 0.27038, 38.),
    ('B10', 'B11', 0.19207, 50.),
    ('B12', 'B13', 0.19988, 30.),
    ('B13', 'B14', 0.34802, 32.),
]

# ---------------- usinas ----------------
uhe = ['UHE1', 'UHE2', 'UHE3', 'UHE4']
uhe_barra = ['B1', 'B2', 'B6', 'B7']
uhe_montante = [None, 'UHE1', 'UHE2', 'UHE3']
uhe_pot = np.array([70., 95., 110., 120.])
uhe_g_min = 0.2 * uhe_pot
uhe_arm_cap = np.array([900., 460., 300., 170.])
prod_uhe = np.array([1., 1., 1., 1.])
defl_min = np.array([6., 10., 14., 18.])

ute = ['UTE1', 'UTE2', 'UTE3', 'UTE4', 'UTE5']
ute_barra = ['B3', 'B6', 'B8', 'B2', 'B13']
ute_pot = np.array([110., 85., 70., 60., 45.])
ute_cvu = np.array([42., 85., 130., 210., 450.])
ute_g_min = np.array([22., 25., 22., 15., 5.])  # <= rampa (senao a unidade nao consegue partir/parar)
ute_rampa = np.array([22., 25., 30., 40., 45.])
t_on = np.array([10, 10, 5, 5, 2])
t_off = np.array([5, 5, 5, 5, 5])
uct_inicial = np.array([1., 0., 0., 0., 0.])
ute_custo_partida = np.array([9000., 4500., 2600., 1300., 450.])

c_deficit = 3000.

# ---------------- demanda ----------------
# participacao de cada barra (base IEEE 14) e "tipo" de curva
carga_base = {'B1': 0.0, 'B2': 21.7, 'B3': 94.2, 'B4': 47.8, 'B5': 7.6, 'B6': 11.2,
              'B7': 0.0, 'B8': 0.0, 'B9': 29.5, 'B10': 9.0, 'B11': 3.5, 'B12': 6.1,
              'B13': 13.5, 'B14': 14.9}
# 'I' industrial (achatada), 'R' residencial (pico noturno), 'M' mista
tipo_carga = {'B2': 'M', 'B3': 'I', 'B4': 'M', 'B5': 'R', 'B6': 'R', 'B9': 'R',
              'B10': 'R', 'B11': 'R', 'B12': 'M', 'B13': 'R', 'B14': 'R'}

perfil = {
    'I': np.array([.86, .84, .83, .82, .82, .84, .88, .93, .97, .99, 1.00, 1.00,
                   .98, .99, 1.00, .99, .98, .97, .96, .94, .92, .90, .89, .87]),
    'M': np.array([.66, .61, .58, .56, .56, .59, .67, .77, .86, .92, .95, .96,
                   .94, .93, .94, .95, .96, .98, 1.00, .98, .93, .86, .78, .71]),
    'R': np.array([.55, .49, .45, .43, .43, .45, .53, .64, .70, .72, .73, .74,
                   .73, .72, .73, .76, .82, .92, 1.00, .99, .93, .84, .72, .62]),
}
# fator diario base: semana util com onda de calor na quinta, fim de semana leve
fator_dia = np.array([0.94, 0.97, 1.00, 1.03, 0.98, 0.86, 0.83])

# ---- demanda aleatoria com MEDIA preservada e >= 1 ponto de ESTRESSE ----
MEDIA_ALVO = 520.0   # MW: media da demanda TOTAL no horizonte (preservada)
PICO_ALVO = 715.0    # MW: demanda total na hora mais carregada (estresse; 765 MW instalados)
SIGMA_DIA = 0.06     # desvio (lognormal) do fator aleatorio de cada dia
SIGMA_HORA = 0.018   # desvio do ruido horario de cada barra


def series_demanda(seed=20260826):
    """Demanda (N, T) aleatoria, deterministica na semente.

    1. perfil horario por tipo de barra x fator do dia da semana x fator aleatorio
       do dia (comum a todas as barras) x ruido horario por barra;
    2. escala para que a media da demanda total seja MEDIA_ALVO;
    3. "estica" cada barra em torno da PROPRIA media: D = m_b + a (D0 - m_b), com
       a escolhido para que o maximo da demanda total seja exatamente PICO_ALVO.
       A transformacao e afim, entao a media de cada barra (e do total) e mantida
       e o pico passa a existir -- a aleatoriedade so decide ONDE ele cai.
    Sem crescimento semanal: todas as semanas tem a mesma media esperada.
    """
    dias = -(-T // 24)
    rng_dia = np.random.default_rng([seed, 0])
    f_dia = np.repeat(np.array([fator_dia[d % 7] for d in range(dias)])
                      * rng_dia.lognormal(-SIGMA_DIA ** 2 / 2, SIGMA_DIA, dias), 24)[:T]
    D = np.zeros((len(barras), T))
    for k, b in enumerate(barras):
        if carga_base[b] <= 0:
            continue
        p = np.tile(perfil[tipo_carga[b]], dias)[:T]
        ruido = 1.0 + np.random.default_rng([seed, k + 1]).normal(0, SIGMA_HORA, T)
        D[k] = carga_base[b] * p * f_dia * ruido
    D *= MEDIA_ALVO / D.sum(axis=0).mean()

    tot = D.sum(axis=0)
    a = (PICO_ALVO - MEDIA_ALVO) / (tot.max() - MEDIA_ALVO)
    m_b = D.mean(axis=1, keepdims=True)
    D = m_b + a * (D - m_b)
    if D.min() < 0:
        raise ValueError(f'esticamento a={a:.2f} gerou demanda negativa; '
                         'reduza PICO_ALVO ou SIGMA_DIA')
    return np.round(D, 2)


def series_vazao(seed=7):
    """Vazoes incrementais (MW-equivalente, prod=1). Semana de estiagem: declinio."""
    rng = np.random.default_rng(seed)
    media = np.array([38., 18., 14., 9.])          # media da semana
    decl = np.linspace(1.25, 0.70, T)               # recessao hidrologica
    Q = np.zeros((len(uhe), T))
    for u in range(len(uhe)):
        ruido = 1.0 + 0.05 * np.sin(np.arange(T) * 2 * np.pi / 37 + u) \
                    + rng.normal(0, 0.02, T)
        Q[u] = media[u] * decl * ruido
    return np.round(np.maximum(Q, 0.0), 3)


def matrizes():
    N, L, H, G = len(barras), len(ramos), len(uhe), len(ute)
    idx = {b: k for k, b in enumerate(barras)}
    A = np.zeros((N, L))
    for l, (i, j, _, _) in enumerate(ramos):
        A[idx[i], l] = +1.0
        A[idx[j], l] = -1.0
    B = np.diag([S_BASE / r[2] for r in ramos])     # MW por radiano
    cap = np.array([r[3] for r in ramos])
    Mh = np.zeros((N, H))
    for u, b in enumerate(uhe_barra):
        Mh[idx[b], u] = 1.0
    Mt = np.zeros((N, G))
    for g, b in enumerate(ute_barra):
        Mt[idx[b], g] = 1.0
    U = np.zeros((H, H))
    for u, mont in enumerate(uhe_montante):
        if mont is not None:
            U[u, uhe.index(mont)] = 1.0
    return A, B, cap, Mh, Mt, U
