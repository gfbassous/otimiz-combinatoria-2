"""
Gera 'dados.xlsx' para o modelo de despacho hidrotermico no sistema IEEE 14 barras.

Abas escritas:
  demanda  -> T linhas (horas) x 14 colunas (B1..B14), MW
  vazao    -> T linhas (horas) x  4 colunas (UHE1..UHE4), MW-equivalente
  uhe / ute / linhas -> parametros (documentacao; nao sao lidos pelo main.py)

As series sao deterministicas (sementes fixas): rodar de novo reproduz os mesmos dados.
O horizonte e definido por SEMANAS em sistema.py (T = SEMANAS * 168 horas).
Para retunar o estresse do sistema, mexa em sistema.py (PICO_ALVO, medias de vazao,
capacidades das linhas) e rode este script novamente.
"""
import numpy as np
import pandas as pd
import sistema as S

D = S.series_demanda()          # (14, T)
Q = S.series_vazao()            # (4, T)
T = S.T

horas = np.arange(T)
dias = ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']

df_dem = pd.DataFrame(D.T, columns=S.barras)
df_dem.insert(0, 'hora', horas)

df_vaz = pd.DataFrame(Q.T, columns=S.uhe)
df_vaz.insert(0, 'hora', horas)

df_uhe = pd.DataFrame({
    'usina': S.uhe, 'barra': S.uhe_barra, 'montante': [m or '' for m in S.uhe_montante],
    'pot_MW': S.uhe_pot, 'g_min_MW': S.uhe_g_min, 'arm_cap_MWh': S.uhe_arm_cap,
    'produtibilidade': S.prod_uhe,
    'defl_min': S.defl_min, 'rampa_MW_h': 0.1 * S.uhe_pot,
    'vazao_media_MW': Q.mean(axis=1).round(2)})

df_ute = pd.DataFrame({
    'usina': S.ute, 'barra': S.ute_barra, 'pot_MW': S.ute_pot, 'cvu': S.ute_cvu,
    'g_min_MW': S.ute_g_min, 'rampa_MW_h': S.ute_rampa, 't_on_h': S.t_on,
    't_off_h': S.t_off, 'custo_partida': S.ute_custo_partida,
    'estado_inicial': S.uct_inicial})

df_lin = pd.DataFrame({
    'de': [r[0] for r in S.ramos], 'para': [r[1] for r in S.ramos],
    'x_pu': [r[2] for r in S.ramos],
    'susceptancia_MW_rad': [S.S_BASE / r[2] for r in S.ramos],
    'capacidade_MW': [r[3] for r in S.ramos]})

with pd.ExcelWriter('dados.xlsx', engine='openpyxl') as xw:
    df_dem.to_excel(xw, sheet_name='demanda', index=False)
    df_vaz.to_excel(xw, sheet_name='vazao', index=False)
    df_uhe.to_excel(xw, sheet_name='uhe', index=False)
    df_ute.to_excel(xw, sheet_name='ute', index=False)
    df_lin.to_excel(xw, sheet_name='linhas', index=False)

tot = D.sum(axis=0)
h_pico = int(tot.argmax())
print(f"horizonte: {S.SEMANAS} semana(s) = {T} horas")
print(f"demanda: pico {tot.max():.1f} MW (h{h_pico}, semana {h_pico // 168 + 1}, "
      f"{dias[(h_pico // 24) % 7]}), "
      f"minimo {tot.min():.1f} MW, media {tot.mean():.1f} MW, energia {tot.sum()/1000:.1f} GWh")
print(f"vazoes : media do horizonte (MW-eq) {dict(zip(S.uhe, Q.mean(axis=1).round(1)))}")
print("dados.xlsx gerado.")
