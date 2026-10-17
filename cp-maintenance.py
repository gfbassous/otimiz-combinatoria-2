import pandas as pd
from docplex.cp.model import *

df_units = pd.read_csv("data/rts79_units_plain.csv")
df_load = pd.read_csv("data/rts79_weekly_load_plain.csv")
df_params = pd.read_csv("data/rts79_params_plain.csv")

crews = 7
margin = 0.15
cap_tot = 3405.

mdl:CpoModel = CpoModel(name="maintenance")

units = df_units['unit_name'].to_list()

unit_vars = {unit:mdl.interval_var(size=df_units.query('unit_name == @unit')['maint_weeks'].values[0],
                                    start = (df_units.query('unit_name == @unit')['earliest_week'].values[0],
                                            df_units.query('unit_name == @unit')['latest_week'].values[0] - \
                                                df_units.query('unit_name == @unit')['maint_weeks'].values[0]),
                                    end = (df_units.query('unit_name == @unit')['earliest_week'].values[0] + \
                                          df_units.query('unit_name == @unit')['maint_weeks'].values[0],
                                          df_units.query('unit_name == @unit')['latest_week'].values[0]),
                                    name = unit) for unit in units}

mdl.add(mdl.cumul_range(sum(mdl.pulse(unit_vars[unit], df_units.query('unit_name == @unit')['crew_req'].values[0]) for unit in units),0,crews))

mw_offline = sum(mdl.pulse(unit_vars[unit], df_units.query('unit_name == @unit')['cap_mw'].values[0]) for unit in units)
for t in range(52):
    mdl.add(mdl.always_in(mw_offline,(t,t+1),0,int(cap_tot - (1+margin)*df_load.iloc[t]['peak_mw'])))

for _bus in df_units['bus'].unique():
    _units_in_bus = df_units.query('bus == @_bus')['unit_name'].values
    _unit_var_in_bus = [unit_vars[v] for v in unit_vars.keys() if v in _units_in_bus]
    if len(_unit_var_in_bus) > 1:
        mdl.add(mdl.no_overlap(_unit_var_in_bus))

peak_offline = mdl.integer_var(0,int(cap_tot), name = 'peak_offline')
mdl.add(mdl.cumul_range(mw_offline, 0, peak_offline))
mdl.add(mdl.minimize(peak_offline))

msol = mdl.solve(TimeLimit=60,execfile="/opt/ibm/ILOG/CPLEX_Studio222/cpoptimizer/bin/x86-64_linux/cpoptimizer")
for u in unit_vars.values():
    v = msol.get_var_solution(u)
    print(f"  {u.get_name()}: weeks {v.get_start()}-{v.get_end()}")
print()