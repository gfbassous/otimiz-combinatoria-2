import pandas as pd
from docplex.cp.model import CpoModel

df_units = pd.read_csv("data/rts79_units_plain.csv")
df_load = pd.read_csv("data/rts79_weekly_load_plain.csv")
df_params = pd.read_csv("data/rts79_params_plain.csv")

crews = 10

mdl = CpoModel(name="maintenance")

units = df_units['unit_name'].to_list()

unit_vars = {unit:mdl.interval_var(size=df_units.query('unit_name == @unit')['maint_weeks'].values[0],
                                   start = df_units.query('unit_name == @unit')['earliest_week'].values[0],
                                   end = df_units.query('unit_name == @unit')['latest_week'].values[0]) for unit in units}




msol = mdl.solve(TimeLimit=60)
print()